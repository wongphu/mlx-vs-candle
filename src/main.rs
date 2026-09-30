//! Micro benchmark: candle (Metal) vs mlx-rs (MLX) on Apple Silicon.
//!
//! Each iteration runs one op and blocks until the GPU finishes
//! (candle: `Device::synchronize`, MLX: `Array::eval`), so timings include
//! dispatch overhead. Reported: median over `iters` after warmup.
//!
//! Shapes are taken from Z-Image-Turbo at 1024x1024 where possible
//! (4096 image tokens, hidden 3840, 30 heads x 128, VAE 512ch @ 128x128).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use mlx_rs::{Array, Dtype};
use std::time::Instant;

// ---------------------------------------------------------------- timing

fn median_ms(warmup: usize, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..warmup {
        f()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f()?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

/// How to express throughput for a benchmark.
#[derive(Clone, Copy)]
enum Work {
    Flops(f64),
    Bytes(f64),
    None,
}

impl Work {
    fn fmt(self, ms: f64) -> String {
        match self {
            Work::Flops(f) => format!("{:7.2} TFLOP/s", f / (ms * 1e-3) / 1e12),
            Work::Bytes(b) => format!("{:7.1} GB/s   ", b / (ms * 1e-3) / 1e9),
            Work::None => "               ".into(),
        }
    }
}

struct Row {
    name: String,
    candle_ms: f64,
    mlx_ms: f64,
    work: Work,
}

// ---------------------------------------------------------------- helpers

fn c_randn(shape: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 1.0, shape, dev)?.to_dtype(dtype)?)
}

fn m_randn(shape: &[i32], dtype: Dtype) -> Result<Array> {
    let a = mlx_rs::random::normal::<f32>(shape, None, None, None)?.as_dtype(dtype)?;
    a.eval()?;
    Ok(a)
}

fn to_mlx_dtype(d: DType) -> Dtype {
    match d {
        DType::F32 => Dtype::Float32,
        DType::BF16 => Dtype::Bfloat16,
        DType::F16 => Dtype::Float16,
        _ => unimplemented!(),
    }
}

fn i32s(s: &[usize]) -> Vec<i32> {
    s.iter().map(|&x| x as i32).collect()
}

/// Copies a candle tensor into an MLX array (via f32 on the host).
fn c_to_m(t: &Tensor) -> Result<Array> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(Array::from_slice(&v, &i32s(t.dims())))
}

fn m_to_vec(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    a.eval()?;
    Ok(a.as_slice::<f32>().to_vec())
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

// ---------------------------------------------------------------- correctness

/// Runs the same inputs through both libraries (f32) and compares outputs.
fn check(dev: &Device) -> Result<()> {
    println!("Correctness (f32, identical inputs, max |diff| / max |x|):");
    let report = |name: &str, c: &Tensor, m: &Array| -> Result<()> {
        let cv = c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let err = max_rel_err(&cv, &m_to_vec(m)?);
        let ok = if err < 1e-3 { "ok" } else { "MISMATCH" };
        println!("  {name:<10} {err:.2e}  {ok}");
        Ok(())
    };

    let a = c_randn(&[256, 512], DType::F32, dev)?;
    let b = c_randn(&[512, 128], DType::F32, dev)?;
    report("matmul", &a.matmul(&b)?, &c_to_m(&a)?.matmul(c_to_m(&b)?)?)?;

    let x = c_randn(&[8, 1000], DType::F32, dev)?;
    report(
        "softmax",
        &candle_nn::ops::softmax_last_dim(&x)?,
        &mlx_rs::ops::softmax_axis(c_to_m(&x)?, -1, None)?,
    )?;

    let w = c_randn(&[1000], DType::F32, dev)?;
    report(
        "rms_norm",
        &candle_nn::ops::rms_norm(&x, &w, 1e-6)?,
        &mlx_rs::fast::rms_norm(c_to_m(&x)?, Some(&c_to_m(&w)?), 1e-6)?,
    )?;

    let (q, k, v) = (
        c_randn(&[1, 4, 64, 64], DType::F32, dev)?,
        c_randn(&[1, 4, 64, 64], DType::F32, dev)?,
        c_randn(&[1, 4, 64, 64], DType::F32, dev)?,
    );
    let scale = 1.0 / 8.0;
    report(
        "sdpa",
        &candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?,
        &mlx_rs::fast::scaled_dot_product_attention(
            c_to_m(&q)?,
            c_to_m(&k)?,
            c_to_m(&v)?,
            scale,
            None,
            None,
        )?,
    )?;

    // conv2d: candle is NCHW / OIHW, MLX is NHWC / OHWI.
    let x = c_randn(&[1, 16, 20, 20], DType::F32, dev)?;
    let w = c_randn(&[32, 16, 3, 3], DType::F32, dev)?;
    let c_out = x.conv2d(&w, 1, 1, 1, 1)?;
    let m_out = mlx_rs::ops::conv2d(
        c_to_m(&x.permute((0, 2, 3, 1))?.contiguous()?)?,
        c_to_m(&w.permute((0, 2, 3, 1))?.contiguous()?)?,
        None,
        (1, 1),
        None,
        None,
    )?;
    report("conv2d", &c_out.permute((0, 2, 3, 1))?.contiguous()?, &m_out)?;

    let x = c_randn(&[4096], DType::F32, dev)?;
    report("silu", &x.silu()?, &mlx_rs::nn::silu(c_to_m(&x)?)?)?;
    println!();
    Ok(())
}

// ---------------------------------------------------------------- benchmarks

fn run(dev: &Device, warmup: usize, iters: usize) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    let mut bench = |name: String,
                     work: Work,
                     c: &mut dyn FnMut() -> Result<()>,
                     m: &mut dyn FnMut() -> Result<()>|
     -> Result<()> {
        let candle_ms = median_ms(warmup, iters, c)?;
        let mlx_ms = median_ms(warmup, iters, m)?;
        println!(
            "  {name:<34} candle {candle_ms:8.3} ms   mlx {mlx_ms:8.3} ms   {:>5.2}x",
            candle_ms / mlx_ms
        );
        rows.push(Row { name, candle_ms, mlx_ms, work });
        Ok(())
    };
    let sync = |_: Tensor| -> Result<()> { Ok(dev.synchronize()?) };
    let eval = |a: Array| -> Result<()> { Ok(a.eval()?) };

    // Dispatch overhead: tiny op, dominated by launch + sync latency.
    {
        let (ca, cb) = (c_randn(&[1024], DType::F32, dev)?, c_randn(&[1024], DType::F32, dev)?);
        let (ma, mb) = (m_randn(&[1024], Dtype::Float32)?, m_randn(&[1024], Dtype::Float32)?);
        bench(
            "add 1K f32 (dispatch latency)".into(),
            Work::None,
            &mut || sync((&ca + &cb)?),
            &mut || eval(ma.add(&mb)?),
        )?;
    }

    // Matmul.
    for dtype in [DType::F32, DType::BF16] {
        for n in [1024usize, 2048, 4096] {
            let (ca, cb) = (c_randn(&[n, n], dtype, dev)?, c_randn(&[n, n], dtype, dev)?);
            let md = to_mlx_dtype(dtype);
            let (ma, mb) = (m_randn(&[n as i32; 2], md)?, m_randn(&[n as i32; 2], md)?);
            bench(
                format!("matmul {n}x{n} {dtype:?}"),
                Work::Flops(2.0 * (n as f64).powi(3)),
                &mut || sync(ca.matmul(&cb)?),
                &mut || eval(ma.matmul(&mb)?),
            )?;
        }
    }

    let bf = DType::BF16;
    let mbf = Dtype::Bfloat16;

    // Transformer linear layer: 4096 tokens x 3840 -> 3840 (weight transposed, as in nn.Linear).
    {
        let (t, d) = (4096usize, 3840usize);
        let cx = c_randn(&[t, d], bf, dev)?;
        let cw = c_randn(&[d, d], bf, dev)?;
        let mx = m_randn(&[t as i32, d as i32], mbf)?;
        let mw = m_randn(&[d as i32, d as i32], mbf)?;
        bench(
            "linear 4096x3840 @ W^T bf16".into(),
            Work::Flops(2.0 * t as f64 * d as f64 * d as f64),
            &mut || sync(cx.matmul(&cw.t()?)?),
            &mut || eval(mx.matmul(mw.transpose()?)?),
        )?;
    }

    // Attention: 30 heads x 128, self-attention over 1024 / 4096 tokens.
    for s in [1024usize, 4096] {
        let (b, h, dh) = (1usize, 30usize, 128usize);
        let shape = [b, h, s, dh];
        let (cq, ck, cv) = (
            c_randn(&shape, bf, dev)?,
            c_randn(&shape, bf, dev)?,
            c_randn(&shape, bf, dev)?,
        );
        let ms = i32s(&shape);
        let (mq, mk, mv) = (m_randn(&ms, mbf)?, m_randn(&ms, mbf)?, m_randn(&ms, mbf)?);
        let scale = 1.0 / (dh as f32).sqrt();
        bench(
            format!("sdpa 30h x {s} x 128 bf16"),
            Work::Flops(4.0 * (b * h) as f64 * (s * s) as f64 * dh as f64),
            &mut || sync(candle_nn::ops::sdpa(&cq, &ck, &cv, None, false, scale, 1.0)?),
            &mut || {
                eval(mlx_rs::fast::scaled_dot_product_attention(
                    &mq, &mk, &mv, scale, None, None,
                )?)
            },
        )?;
    }

    // Softmax over attention-score-sized rows.
    {
        let shape = [30usize, 1024, 1024];
        let c = c_randn(&shape, bf, dev)?;
        let m = m_randn(&i32s(&shape), mbf)?;
        let bytes = 2.0 * 2.0 * shape.iter().product::<usize>() as f64;
        bench(
            "softmax 30x1024x1024 bf16".into(),
            Work::Bytes(bytes),
            &mut || sync(candle_nn::ops::softmax_last_dim(&c)?),
            &mut || eval(mlx_rs::ops::softmax_axis(&m, -1, None)?),
        )?;
    }

    // Norms over 4096 tokens x 3840.
    {
        let (t, d) = (4096usize, 3840usize);
        let cx = c_randn(&[t, d], bf, dev)?;
        let cw = c_randn(&[d], bf, dev)?;
        let cb = c_randn(&[d], bf, dev)?;
        let mx = m_randn(&[t as i32, d as i32], mbf)?;
        let mw = m_randn(&[d as i32], mbf)?;
        let mb = m_randn(&[d as i32], mbf)?;
        let bytes = 2.0 * 2.0 * (t * d) as f64;
        bench(
            "rms_norm 4096x3840 bf16".into(),
            Work::Bytes(bytes),
            &mut || sync(candle_nn::ops::rms_norm(&cx, &cw, 1e-6)?),
            &mut || eval(mlx_rs::fast::rms_norm(&mx, Some(&mw), 1e-6)?),
        )?;
        bench(
            "layer_norm 4096x3840 bf16".into(),
            Work::Bytes(bytes),
            &mut || sync(candle_nn::ops::layer_norm(&cx, &cw, &cb, 1e-6)?),
            &mut || eval(mlx_rs::fast::layer_norm(&mx, &mw, &mb, 1e-6)?),
        )?;
    }

    // VAE-style 3x3 conv, 512 -> 512 channels at 128x128 (each library in its native layout).
    {
        let (c, hw) = (512usize, 128usize);
        let cx = c_randn(&[1, c, hw, hw], bf, dev)?;
        let cw = c_randn(&[c, c, 3, 3], bf, dev)?;
        let mx = m_randn(&[1, hw as i32, hw as i32, c as i32], mbf)?;
        let mw = m_randn(&[c as i32, 3, 3, c as i32], mbf)?;
        bench(
            "conv2d 3x3 512ch 128x128 bf16".into(),
            Work::Flops(2.0 * (c * c * 9 * hw * hw) as f64),
            &mut || sync(cx.conv2d(&cw, 1, 1, 1, 1)?),
            &mut || eval(mlx_rs::ops::conv2d(&mx, &mw, None, (1, 1), None, None)?),
        )?;
    }

    // Elementwise, memory bound: 64M elements.
    {
        let n = 64usize << 20;
        let (ca, cb) = (c_randn(&[n], bf, dev)?, c_randn(&[n], bf, dev)?);
        let (ma, mb) = (m_randn(&[n as i32], mbf)?, m_randn(&[n as i32], mbf)?);
        bench(
            "add 64M bf16".into(),
            Work::Bytes(3.0 * 2.0 * n as f64),
            &mut || sync((&ca + &cb)?),
            &mut || eval(ma.add(&mb)?),
        )?;
        bench(
            "silu 64M bf16".into(),
            Work::Bytes(2.0 * 2.0 * n as f64),
            &mut || sync(ca.silu()?),
            &mut || eval(mlx_rs::nn::silu(&ma)?),
        )?;
        // A small op chain: candle runs one kernel per op; MLX builds a lazy
        // graph and evaluates it at the end (still one kernel per op unless compiled).
        bench(
            "chain silu(a*b+a) 64M bf16".into(),
            Work::None,
            &mut || sync(((&ca * &cb)? + &ca)?.silu()?),
            &mut || eval(mlx_rs::nn::silu(ma.multiply(&mb)?.add(&ma)?)?),
        )?;
    }

    Ok(rows)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let iters: usize = std::env::args().nth(1).map(|s| s.parse()).transpose()?.unwrap_or(30);
    println!("candle 0.11.0 (Metal) vs mlx-rs 0.32.0 (MLX)  |  median of {iters} iters\n");

    check(&dev)?;

    println!("Benchmarks (candle / mlx = speedup of MLX; >1 means MLX faster):");
    let rows = run(&dev, 5, iters)?;

    println!("\n| op | candle ms | mlx ms | candle | mlx | mlx speedup |");
    println!("|---|---:|---:|---:|---:|---:|");
    for r in &rows {
        println!(
            "| {} | {:.3} | {:.3} | {} | {} | {:.2}x |",
            r.name,
            r.candle_ms,
            r.mlx_ms,
            r.work.fmt(r.candle_ms).trim(),
            r.work.fmt(r.mlx_ms).trim(),
            r.candle_ms / r.mlx_ms
        );
    }
    Ok(())
}

# mlx-vs-candle

A GPU micro benchmark of [candle](https://github.com/huggingface/candle) (Metal backend)
vs [mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings to Apple's MLX) on the ops
that dominate diffusion transformers. Shapes come from Z-Image-Turbo at 1024×1024
(4096 image tokens, hidden 3840, 30 heads × 128, VAE 512 channels at 128×128).

## Results

Apple M3 Max (96 GB), macOS 26, candle 0.11.0, mlx-rs 0.32.0. bf16 unless noted.
Median of 50 iterations after 5 warmup; each iteration runs one op and waits for
the GPU (candle `Device::synchronize`, MLX `Array::eval`), so times include dispatch
overhead. "MLX speedup" > 1 means MLX is faster; the range covers two full runs
(raw output in [`results/`](results)).

| Op | candle | MLX | MLX speedup |
|---|---:|---:|---:|
| add 1K f32 (dispatch latency) | 0.228 ms | 0.218 ms | 1.04–1.07× |
| matmul 1024² f32 | 0.828 ms | 0.693 ms | 1.01–1.19× |
| matmul 2048² f32 | 2.341 ms · 7.3 TFLOP/s | 2.349 ms · 7.3 TFLOP/s | 1.00–1.01× |
| matmul 4096² f32 | 16.67 ms · 8.3 TFLOP/s | 16.65 ms · 8.3 TFLOP/s | 1.00× |
| matmul 1024² bf16 | 0.477 ms · 4.5 TFLOP/s | 0.456 ms · 4.7 TFLOP/s | 1.05–1.06× |
| matmul 2048² bf16 | 2.201 ms · 7.8 TFLOP/s | 2.106 ms · 8.2 TFLOP/s | 1.05× |
| matmul 4096² bf16 | 15.22 ms · 9.0 TFLOP/s | 14.78 ms · 9.3 TFLOP/s | 1.03–1.06× |
| linear 4096×3840 @ Wᵀ | 14.34 ms · 8.4 TFLOP/s | 13.35 ms · 9.1 TFLOP/s | 1.07–1.08× |
| attention 30h × 1024 tok × 128 | 2.231 ms · 7.2 TFLOP/s | 2.241 ms · 7.2 TFLOP/s | 0.99–1.00× |
| attention 30h × 4096 tok × 128 | 30.84 ms · 8.4 TFLOP/s | 31.07 ms · 8.3 TFLOP/s | 0.99× |
| softmax 30×1024×1024 | 0.953 ms · 132 GB/s | 0.899 ms · 140 GB/s | 1.06–1.07× |
| rms_norm 4096×3840 | 0.447 ms · 141 GB/s | 0.443 ms · 142 GB/s | 1.01–1.07× |
| layer_norm 4096×3840 | 0.469 ms · 134 GB/s | 0.439 ms · 143 GB/s | 1.07–1.09× |
| **conv2d 3×3, 512→512 ch, 128×128** | **35.27 ms · 2.2 TFLOP/s** | **3.46 ms · 22.4 TFLOP/s\*** | **10.2×** |
| add 64M | 1.813 ms · 222 GB/s | 1.818 ms · 222 GB/s | 1.00× |
| silu 64M | 1.301 ms · 206 GB/s | 1.333 ms · 201 GB/s | 0.98× |
| chain silu(a·b + a) 64M | 4.464 ms | 4.458 ms | 1.00× |

\* Effective rate counting direct-convolution FLOPs. MLX uses a Winograd kernel for 3×3
convolutions (~2.25× fewer multiplications); candle 0.11 has no Winograd path.

### Takeaways

- **Matmul, attention, norms, softmax and elementwise ops are at parity** (MLX ahead by
  0–9%). These make up nearly all of a diffusion transformer's denoising steps.
- **3×3 convolution is the one large gap: MLX is ~10× faster.** In diffusion models this
  mostly affects the VAE (encode/decode), not the denoising loop.
- Neither library fuses op chains by default (MLX would need `compile`, not tested here).
- These are isolated ops; end-to-end model performance can differ.

### Correctness

Before timing, the same f32 inputs are run through both libraries and compared
(max |diff| / max |x|): matmul, softmax, rms_norm, sdpa, conv2d and silu all agree to
~1e-6 or better.

## Running

```bash
cargo run --release -- 50    # iterations per op (default 30)
```

Requirements for building mlx-rs (it compiles MLX from source):

- `cmake` (e.g. `brew install cmake`)
- Xcode's Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`

candle compiles its Metal kernels at runtime and needs neither.

Layouts: each library runs in its native layout (candle conv2d is NCHW, MLX is NHWC).

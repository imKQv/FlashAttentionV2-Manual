# Manual FlashAttentionV2 Benchmarks

This repository compares five FP16 attention backends with a fixed long-sequence configuration:

- Eager PyTorch
- Manual tiled PyTorch
- Manual Triton
- PyTorch SDPA Efficient Attention
- Tri Dao FlashAttentionV2

## Layout

- `models/`: backend implementations and the lazy model registry.
- `benchmarks/`: Python benchmark drivers.
- `benchmark_efficiency.sh`: efficiency entry point.
- `benchmark_accuracy.sh`: SDPA-referenced accuracy entry point.
- `outputs/`: generated Markdown reports.
- `tmp/`: archived development files and experiments.

All backends use the repository-wide layout `[B, H, T, D]`. The FlashAttentionV2 adapter converts to and from its native `[B, T, H, D]` layout.

## Configuration
Test environment: NVIDIA L40S (Ada, compute capability 8.9, 142 SMs), with 100 KiB shared memory, 65,536 registers, and up to 1,536 threads per SM. Each block can request approximately 99 KiB of opt-in shared memory; warp size is 32. The benchmark uses `B=1, H=16, D=128`, FP16, sequence lengths from 1K to 32K, and both causal and non-causal modes.

The benchmark shape is fixed to `B=1, H=16, D=128`, FP16, with sequence lengths:

`[1024, 2048, 4096, 8192, 16384, 32768]`

## Accuracy

Accuracy tests are available for the two manual implementations and use SDPA Efficient Attention as the reference:

```bash
./benchmark_accuracy.sh --models torch
./benchmark_accuracy.sh --models triton
```

The reports contain forward and backward MAE/MRE for every sequence length and are written to `outputs/accuracy_<model>.md`. Default FP16 tolerances are `rtol=atol=2e-2`.

### torch accuracy test

| Sequence length | Causal Forward | Causal Backward | Non-causal Forward | Non-causal Backward | Result |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | Output: 6.316e-05 / 5.726e-03 | dQ: 1.312e-04 / 7.509e-02<br>dK: 1.141e-04 / 1.423e-02<br>dV: 1.162e-04 / 1.613e-02 | Output: 4.453e-05 / 6.472e-03 | dQ: 9.208e-05 / 1.233e-02<br>dK: 9.462e-05 / 1.670e-02<br>dV: 9.383e-05 / 1.774e-02 | PASS |
| 2K | Output: 5.207e-05 / 6.245e-03 | dQ: 1.074e-04 / 3.502e-02<br>dK: 9.560e-05 / 1.704e-02<br>dV: 9.613e-05 / 1.744e-02 | Output: 3.786e-05 / 7.325e-03 | dQ: 8.660e-05 / 1.299e-02<br>dK: 9.172e-05 / 2.325e-02<br>dV: 9.078e-05 / 2.334e-02 | PASS |
| 4K | Output: 4.378e-05 / 6.934e-03 | dQ: 9.158e-05 / 4.429e-02<br>dK: 8.340e-05 / 2.093e-02<br>dV: 8.364e-05 / 2.079e-02 | Output: 3.352e-05 / 8.007e-03 | dQ: 6.689e-05 / 1.418e-02<br>dK: 7.007e-05 / 2.381e-02<br>dV: 6.940e-05 / 2.334e-02 | PASS |
| 8K | Output: 3.803e-05 / 7.820e-03 | dQ: 7.508e-05 / 2.272e-02<br>dK: 6.903e-05 / 2.252e-02<br>dV: 6.925e-05 / 2.269e-02 | Output: 3.091e-05 / 9.395e-03 | dQ: 5.363e-05 / 1.599e-02<br>dK: 5.516e-05 / 2.539e-02<br>dV: 5.459e-05 / 2.499e-02 | PASS |
| 16K | Output: 3.380e-05 / 9.098e-03 | dQ: 6.182e-05 / 2.518e-02<br>dK: 5.764e-05 / 2.462e-02<br>dV: 5.740e-05 / 2.467e-02 | Output: 2.928e-05 / 1.138e-02 | dQ: 4.464e-05 / 1.875e-02<br>dK: 4.478e-05 / 2.793e-02<br>dV: 4.438e-05 / 2.752e-02 | PASS |
| 32K | Output: 3.136e-05 / 1.110e-02 | dQ: 5.149e-05 / 2.197e-02<br>dK: 4.908e-05 / 2.749e-02<br>dV: 4.886e-05 / 2.745e-02 | Output: 2.911e-05 / 1.452e-02 | dQ: 3.902e-05 / 2.218e-02<br>dK: 3.835e-05 / 3.199e-02<br>dV: 3.802e-05 / 3.223e-02 | PASS |

### triton accuracy test

| Sequence length | Causal Forward | Causal Backward | Non-causal Forward | Non-causal Backward | Result |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | Output: 4.032e-06 / 5.148e-04 | dQ: 2.363e-05 / 2.659e-03<br>dK: 1.918e-05 / 2.319e-03<br>dV: 1.428e-07 / 2.064e-05 | Output: 2.513e-06 / 4.801e-04 | dQ: 1.406e-05 / 2.656e-03<br>dK: 1.388e-05 / 2.557e-03<br>dV: 2.128e-07 / 4.690e-05 | PASS |
| 2K | Output: 2.999e-06 / 4.382e-04 | dQ: 1.761e-05 / 2.620e-03<br>dK: 1.420e-05 / 2.417e-03<br>dV: 1.651e-07 / 2.639e-05 | Output: 1.490e-06 / 3.859e-04 | dQ: 1.012e-05 / 2.510e-03<br>dK: 9.985e-06 / 2.531e-03<br>dV: 2.258e-07 / 6.024e-05 | PASS |
| 4K | Output: 2.047e-06 / 4.000e-04 | dQ: 1.310e-05 / 2.530e-03<br>dK: 1.046e-05 / 2.378e-03<br>dV: 1.767e-07 / 4.415e-05 | Output: 8.573e-07 / 2.979e-04 | dQ: 7.270e-06 / 2.517e-03<br>dK: 7.171e-06 / 2.473e-03<br>dV: 2.163e-07 / 7.363e-05 | PASS |
| 8K | Output: 1.346e-06 / 3.296e-04 | dQ: 9.545e-06 / 2.507e-03<br>dK: 7.582e-06 / 2.384e-03<br>dV: 1.783e-07 / 6.126e-05 | Output: 4.855e-07 / 2.312e-04 | dQ: 5.298e-06 / 2.508e-03<br>dK: 5.228e-06 / 2.470e-03<br>dV: 1.969e-07 / 9.263e-05 | PASS |
| 16K | Output: 8.513e-07 / 2.606e-04 | dQ: 7.116e-06 / 2.575e-03<br>dK: 5.657e-06 / 2.501e-03<br>dV: 1.679e-07 / 7.954e-05 | Output: 2.710e-07 / 1.759e-04 | dQ: 4.256e-06 / 2.769e-03<br>dK: 4.215e-06 / 2.759e-03<br>dV: 1.812e-07 / 1.179e-04 | PASS |
| 32K | Output: 5.252e-07 / 2.085e-04 | dQ: 5.663e-06 / 2.915e-03<br>dK: 4.616e-06 / 3.061e-03<br>dV: 1.564e-07 / 1.070e-04 | Output: 1.518e-07 / 1.343e-04 | dQ: 4.365e-06 / 3.899e-03<br>dK: 4.338e-06 / 3.891e-03<br>dV: 1.770e-07 / 1.593e-04 | PASS |


## Efficiency
Each efficiency run produces four tables: causal forward, causal backward, non-causal forward, and non-causal backward.

Run one backend per command:

```bash
./benchmark_efficiency.sh --models eager
./benchmark_efficiency.sh --models torch
./benchmark_efficiency.sh --models triton
./benchmark_efficiency.sh --models sdpa
./benchmark_efficiency.sh --models flash-attentionv2
```

Reports are written to `outputs/efficiency_<model>.md`.

Use a specific runtime or GPU when needed:

```bash
PYTHON_BIN=python ./benchmark_efficiency.sh --models triton --device 0
```

The FlashAttentionV2 adapter uses the optional `flash-attn` package; the consolidated reference results use version 2.8.3.

### causal forward

| Sequence length | Eager | Torch Manual | Triton Manual | SDPA (Efficient Attention) | FlashAttentionV2 (2.8.3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | 0.16 ms (1.00×)<br>0.09 GiB (100.00%) | 37.07 ms (0.00×)<br>0.03 GiB (33.33%) | 0.12 ms (1.33×)<br>0.02 GiB (22.22%) | 0.07 ms (2.29×)<br>0.02 GiB (22.22%) | 0.09 ms (1.78×)<br>0.02 GiB (22.22%) |
| 2K | 1.70 ms (1.00×)<br>0.30 GiB (100.00%) | 143.36 ms (0.01×)<br>0.05 GiB (16.67%) | 0.25 ms (6.80×)<br>0.04 GiB (13.33%) | 0.19 ms (8.95×)<br>0.04 GiB (13.33%) | 0.14 ms (12.14×)<br>0.04 GiB (13.33%) |
| 4K | 6.63 ms (1.00×)<br>1.10 GiB (100.00%) | 580.54 ms (0.01×)<br>0.09 GiB (8.18%) | 0.89 ms (7.45×)<br>0.08 GiB (7.27%) | 0.59 ms (11.24×)<br>0.08 GiB (7.27%) | 0.42 ms (15.79×)<br>0.08 GiB (7.27%) |
| 8K | 27.99 ms (1.00×)<br>4.23 GiB (100.00%) | 2270.80 ms (0.01×)<br>0.17 GiB (4.02%) | 3.55 ms (7.88×)<br>0.16 GiB (3.78%) | 2.10 ms (13.33×)<br>0.16 GiB (3.78%) | 1.48 ms (18.91×)<br>0.16 GiB (3.78%) |
| 16K | 111.04 ms (1.00×)<br>16.57 GiB (100.00%) | 9069.72 ms (0.01×)<br>0.32 GiB (1.93%) | 13.72 ms (8.09×)<br>0.31 GiB (1.87%) | 7.71 ms (14.40×)<br>0.31 GiB (1.87%) | 5.32 ms (20.87×)<br>0.31 GiB (1.87%) |
| 32K | 444.16 ms\* (1.00×)<br>64.62 GiB\* (100.00%) | 36397.94 ms (0.01×)<br>0.64 GiB (0.99%) | 53.55 ms (8.29×)<br>0.63 GiB (0.97%) | 32.74 ms (13.57×)<br>0.62 GiB (0.96%) | 22.54 ms (19.71×)<br>0.63 GiB (0.97%) |

### causal backward

| Sequence length | Eager | Torch Manual | Triton Manual | SDPA (Efficient Attention) | FlashAttentionV2 (2.8.3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | 0.52 ms (1.00×)<br>0.17 GiB (100.00%) | 46.20 ms (0.01×)<br>0.05 GiB (29.41%) | 0.57 ms (0.91×)<br>0.06 GiB (35.29%) | 0.43 ms (1.21×)<br>0.05 GiB (29.41%) | 0.32 ms (1.63×)<br>0.04 GiB (23.53%) |
| 2K | 3.52 ms (1.00×)<br>0.57 GiB (100.00%) | 182.17 ms (0.02×)<br>0.09 GiB (15.79%) | 2.02 ms (1.74×)<br>0.12 GiB (21.05%) | 1.01 ms (3.49×)<br>0.10 GiB (17.54%) | 0.35 ms (10.06×)<br>0.08 GiB (14.04%) |
| 4K | 13.99 ms (1.00×)<br>2.13 GiB (100.00%) | 726.32 ms (0.02×)<br>0.16 GiB (7.51%) | 7.68 ms (1.82×)<br>0.23 GiB (10.80%) | 3.25 ms (4.30×)<br>0.18 GiB (8.45%) | 1.20 ms (11.66×)<br>0.16 GiB (7.51%) |
| 8K | 52.61 ms (1.00×)<br>8.27 GiB (100.00%) | 2865.09 ms (0.02×)<br>0.30 GiB (3.63%) | 29.49 ms (1.78×)<br>0.47 GiB (5.68%) | 11.57 ms (4.55×)<br>0.36 GiB (4.35%) | 4.36 ms (12.07×)<br>0.31 GiB (3.75%) |
| 16K | 210.80 ms (1.00×)<br>32.64 GiB (100.00%) | 11536.02 ms (0.02×)<br>0.58 GiB (1.78%) | 113.65 ms (1.85×)<br>0.94 GiB (2.88%) | 45.21 ms (4.66×)<br>0.70 GiB (2.14%) | 15.78 ms (13.36×)<br>0.63 GiB (1.93%) |
| 32K | 843.20 ms\* (1.00×)<br>129.50 GiB\* (100.00%) | 45928.31 ms (0.02×)<br>1.14 GiB (0.88%) | 441.34 ms (1.91×)<br>1.88 GiB (1.45%) | 179.29 ms (4.70×)<br>1.39 GiB (1.07%) | 61.75 ms (13.66×)<br>1.25 GiB (0.97%) |

### non-causal forward

| Sequence length | Eager | Torch Manual | Triton Manual | SDPA (Efficient Attention) | FlashAttentionV2 (2.8.3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | 0.14 ms (1.00×)<br>0.10 GiB (100.00%) | 25.83 ms (0.01×)<br>0.04 GiB (40.00%) | 0.12 ms (1.17×)<br>0.02 GiB (20.00%) | 0.08 ms (1.75×)<br>0.02 GiB (20.00%) | 0.09 ms (1.56×)<br>0.02 GiB (20.00%) |
| 2K | 1.43 ms (1.00×)<br>0.30 GiB (100.00%) | 101.97 ms (0.01×)<br>0.06 GiB (20.00%) | 0.24 ms (5.96×)<br>0.04 GiB (13.33%) | 0.27 ms (5.30×)<br>0.04 GiB (13.33%) | 0.17 ms (8.41×)<br>0.04 GiB (13.33%) |
| 4K | 5.44 ms (1.00×)<br>1.09 GiB (100.00%) | 398.83 ms (0.01×)<br>0.10 GiB (9.17%) | 0.85 ms (6.40×)<br>0.08 GiB (7.34%) | 1.04 ms (5.23×)<br>0.08 GiB (7.34%) | 0.63 ms (8.63×)<br>0.08 GiB (7.34%) |
| 8K | 21.20 ms (1.00×)<br>4.17 GiB (100.00%) | 1594.15 ms (0.01×)<br>0.17 GiB (4.08%) | 3.51 ms (6.04×)<br>0.16 GiB (3.84%) | 3.87 ms (5.48×)<br>0.16 GiB (3.84%) | 2.39 ms (8.87×)<br>0.16 GiB (3.84%) |
| 16K | 80.68 ms (1.00×)<br>16.33 GiB (100.00%) | 6360.37 ms (0.01×)<br>0.33 GiB (2.02%) | 13.53 ms (5.96×)<br>0.31 GiB (1.90%) | 15.84 ms (5.09×)<br>0.31 GiB (1.90%) | 8.90 ms (9.07×)<br>0.31 GiB (1.90%) |
| 32K | 322.72 ms\* (1.00×)<br>64.62 GiB\* (100.00%) | 25410.45 ms (0.01×)<br>0.64 GiB (0.99%) | 52.85 ms (6.11×)<br>0.63 GiB (0.97%) | 64.73 ms (4.99×)<br>0.62 GiB (0.96%) | 38.36 ms (8.41×)<br>0.63 GiB (0.97%) |

### non-causal backward

| Sequence length | Eager | Torch Manual | Triton Manual | SDPA (Efficient Attention) | FlashAttentionV2 (2.8.3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | 0.48 ms (1.00×)<br>0.16 GiB (100.00%) | 34.29 ms (0.01×)<br>0.05 GiB (31.25%) | 0.57 ms (0.84×)<br>0.06 GiB (37.50%) | 0.58 ms (0.83×)<br>0.05 GiB (31.25%) | 0.49 ms (0.98×)<br>0.04 GiB (25.00%) |
| 2K | 2.63 ms (1.00×)<br>0.56 GiB (100.00%) | 133.45 ms (0.02×)<br>0.09 GiB (16.07%) | 1.99 ms (1.32×)<br>0.12 GiB (21.43%) | 1.60 ms (1.64×)<br>0.10 GiB (17.86%) | 0.59 ms (4.46×)<br>0.08 GiB (14.29%) |
| 4K | 10.57 ms (1.00×)<br>2.11 GiB (100.00%) | 526.63 ms (0.02×)<br>0.16 GiB (7.58%) | 7.64 ms (1.38×)<br>0.23 GiB (10.90%) | 6.07 ms (1.74×)<br>0.18 GiB (8.53%) | 2.12 ms (4.99×)<br>0.16 GiB (7.58%) |
| 8K | 39.68 ms (1.00×)<br>8.20 GiB (100.00%) | 2109.01 ms (0.02×)<br>0.30 GiB (3.66%) | 29.53 ms (1.34×)<br>0.47 GiB (5.73%) | 23.00 ms (1.73×)<br>0.36 GiB (4.39%) | 7.95 ms (4.99×)<br>0.31 GiB (3.78%) |
| 16K | 152.26 ms (1.00×)<br>32.39 GiB (100.00%) | 8406.64 ms (0.02×)<br>0.58 GiB (1.79%) | 113.73 ms (1.34×)<br>0.94 GiB (2.90%) | 90.54 ms (1.68×)<br>0.70 GiB (2.16%) | 30.14 ms (5.05×)<br>0.63 GiB (1.95%) |
| 32K | 609.04 ms\* (1.00×)<br>129.50 GiB\* (100.00%) | 33489.93 ms (0.02×)<br>1.14 GiB (0.88%) | 441.30 ms (1.38×)<br>1.88 GiB (1.45%) | 358.40 ms (1.70×)<br>1.39 GiB (1.07%) | 122.93 ms (4.95×)<br>1.25 GiB (0.97%) |


## Manual Torch vs. Manual Triton capabilities

| Capability | Manual Torch | Manual Triton |
|:---:|:---:|:---:|
| Q layout | `[B,H,Tq,dk]` | `[B,H,Tq,D]` |
| K layout | `[B,H,Tk,dk]` | `[B,H,Tk,D]` |
| V layout | `[B,H,Tk,dv]` | `[B,H,Tk,D]` |
| Output layout | `[B,H,Tq,dv]` | `[B,H,Tq,D]` |
| Batch/head compatibility | Q/K/V must match | Q/K/V must match; validated explicitly |
| `Tq != Tk`, non-causal | Supported | Supported |
| `Tq != Tk`, causal | Unsupported (top-left alignment only) | Unsupported; requires `Tq == Tk` and raises an error otherwise |
| Sequence lengths not divisible by 16 | Supported | Supported |
| Partial non-causal tail tile | Supported | Supported with a padding mask |
| Partial causal tail tile | Supported | Supported with a padding mask |
| `dk != dv` | Supported | Unsupported; requires `dk == dv` |
| Head dimension | Standard dimensions supported by PyTorch matmul | Only `16/32/64/128`; forward and backward verified |
| FP32 | Supported | Supported |
| FP16 | Supported | Supported with FP32 accumulation inside the kernel |
| BF16 | Supported | Supported with FP32 accumulation inside the kernel |
| CPU | Supported | Unsupported |
| CUDA | Supported | Supported |
| Non-contiguous Q/K/V | Generally supported | Forward and backward verified for ordinary strided slices |
| Q tile | Configurable at call time | Selected from in-code candidates by autotune |
| K tile | Configurable at call time | Selected from in-code candidates by autotune |
| Tile candidates | No Triton constraint | Current autotune candidates use `32`, `64`, or `128` |
| Custom scale | Unsupported | Unsupported |
| Attention mask | Causal switch only | Causal switch only |
| Attention bias | Unsupported | Unsupported |
| Dropout | Unsupported | Unsupported |
| GQA/MQA | Unsupported | Unsupported |
| Variable-length/packed input | Unsupported | Unsupported |
| KV cache/decode | Unsupported | Unsupported |
| `dQ/dK/dV` | Supported | Supported |
| Gradient determinism | Primarily determined by PyTorch/CUDA operators | `dQ` uses `atomic_add` and may be slightly nondeterministic |
| FP32 dot precision | Determined by the PyTorch matmul precision setting | `tl.dot` currently uses TF32 by default |
| Memory complexity | Tiled; does not materialize the full `T×T` matrix | Tiled; does not materialize the full `T×T` matrix |
| Execution | Python nested loops and many small CUDA operations | Fused Triton kernel |

## Future optimization opportunities

- **Causal pruning**: Skip future key tiles and apply a mask only to diagonal tiles. Bound the K-loop by the Query tile and separate the main region from the diagonal tile.

- **Split backward kernels**: Compute dQ and dK/dV separately to remove `atomic_add` contention and nondeterminism. Assign each Query or Key tile to one owning program.

- **Use exp2**: Adopt a consistent base-2 scale in forward and backward to reduce softmax-loop overhead. Multiply the scale by `log2(e)` and update LSE storage and probability reconstruction.

- **Add features after optimization**: Extend the stable core with GQA, variable-length input, dropout, and bias, including head mapping, packed offsets, RNG state, and structured bias generation.

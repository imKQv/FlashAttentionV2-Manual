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

The main speed bottleneck of the torch implementation is the massive number of loop iterations, but it also saves GPU memory. The causal Triton implementation is about half as fast as SDPA, while the non-causal implementation is on the same order of magnitude as SDPA. This is because we did not prune the roughly 50% of blocks that are masked in the causal case. The official CUDA implementation of FlashAttentionV2 is the fastest; it can use larger tiles and greater parallelism, and has more optimizations.

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

# Efficiency benchmark: Triton Manual

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Warmup: 2

## Causal Forward

| Sequence length | Triton Manual |
|:---:|:---:|
| 1K | 0.12 ms<br>0.02 GiB |
| 2K | 0.25 ms<br>0.04 GiB |
| 4K | 0.89 ms<br>0.08 GiB |
| 8K | 3.55 ms<br>0.16 GiB |
| 16K | 13.72 ms<br>0.31 GiB |
| 32K | 53.55 ms<br>0.63 GiB |

## Causal Backward

| Sequence length | Triton Manual |
|:---:|:---:|
| 1K | 0.57 ms<br>0.06 GiB |
| 2K | 2.02 ms<br>0.12 GiB |
| 4K | 7.68 ms<br>0.23 GiB |
| 8K | 29.49 ms<br>0.47 GiB |
| 16K | 113.65 ms<br>0.94 GiB |
| 32K | 441.34 ms<br>1.88 GiB |

## Non-causal Forward

| Sequence length | Triton Manual |
|:---:|:---:|
| 1K | 0.12 ms<br>0.02 GiB |
| 2K | 0.24 ms<br>0.04 GiB |
| 4K | 0.85 ms<br>0.08 GiB |
| 8K | 3.51 ms<br>0.16 GiB |
| 16K | 13.53 ms<br>0.31 GiB |
| 32K | 52.85 ms<br>0.63 GiB |

## Non-causal Backward

| Sequence length | Triton Manual |
|:---:|:---:|
| 1K | 0.57 ms<br>0.06 GiB |
| 2K | 1.99 ms<br>0.12 GiB |
| 4K | 7.64 ms<br>0.23 GiB |
| 8K | 29.53 ms<br>0.47 GiB |
| 16K | 113.73 ms<br>0.94 GiB |
| 32K | 441.30 ms<br>1.88 GiB |

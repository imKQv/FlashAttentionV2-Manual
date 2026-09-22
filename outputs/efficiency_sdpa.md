# Efficiency benchmark: SDPA (Efficient Attention)

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Warmup: 2

## Causal Forward

| Sequence length | SDPA (Efficient Attention) |
|:---:|:---:|
| 1K | 0.07 ms<br>0.02 GiB |
| 2K | 0.19 ms<br>0.04 GiB |
| 4K | 0.59 ms<br>0.08 GiB |
| 8K | 2.10 ms<br>0.16 GiB |
| 16K | 7.71 ms<br>0.31 GiB |
| 32K | 32.74 ms<br>0.62 GiB |

## Causal Backward

| Sequence length | SDPA (Efficient Attention) |
|:---:|:---:|
| 1K | 0.43 ms<br>0.05 GiB |
| 2K | 1.01 ms<br>0.10 GiB |
| 4K | 3.25 ms<br>0.18 GiB |
| 8K | 11.57 ms<br>0.36 GiB |
| 16K | 45.21 ms<br>0.70 GiB |
| 32K | 179.29 ms<br>1.39 GiB |

## Non-causal Forward

| Sequence length | SDPA (Efficient Attention) |
|:---:|:---:|
| 1K | 0.08 ms<br>0.02 GiB |
| 2K | 0.27 ms<br>0.04 GiB |
| 4K | 1.04 ms<br>0.08 GiB |
| 8K | 3.87 ms<br>0.16 GiB |
| 16K | 15.84 ms<br>0.31 GiB |
| 32K | 64.73 ms<br>0.62 GiB |

## Non-causal Backward

| Sequence length | SDPA (Efficient Attention) |
|:---:|:---:|
| 1K | 0.58 ms<br>0.05 GiB |
| 2K | 1.60 ms<br>0.10 GiB |
| 4K | 6.07 ms<br>0.18 GiB |
| 8K | 23.00 ms<br>0.36 GiB |
| 16K | 90.54 ms<br>0.70 GiB |
| 32K | 358.40 ms<br>1.39 GiB |

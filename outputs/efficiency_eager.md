# Efficiency benchmark: Eager

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Warmup: 2

## Causal Forward

| Sequence length | Eager |
|:---:|:---:|
| 1K | 0.16 ms<br>0.09 GiB |
| 2K | 1.70 ms<br>0.30 GiB |
| 4K | 6.63 ms<br>1.10 GiB |
| 8K | 27.99 ms<br>4.23 GiB |
| 16K | 111.04 ms<br>16.57 GiB |
| 32K | OOM (estimated 64.62 GiB) |

## Causal Backward

| Sequence length | Eager |
|:---:|:---:|
| 1K | 0.52 ms<br>0.17 GiB |
| 2K | 3.52 ms<br>0.57 GiB |
| 4K | 13.99 ms<br>2.13 GiB |
| 8K | 52.61 ms<br>8.27 GiB |
| 16K | 210.80 ms<br>32.64 GiB |
| 32K | OOM (estimated 129.50 GiB) |

## Non-causal Forward

| Sequence length | Eager |
|:---:|:---:|
| 1K | 0.14 ms<br>0.10 GiB |
| 2K | 1.43 ms<br>0.30 GiB |
| 4K | 5.44 ms<br>1.09 GiB |
| 8K | 21.20 ms<br>4.17 GiB |
| 16K | 80.68 ms<br>16.33 GiB |
| 32K | OOM (estimated 64.62 GiB) |

## Non-causal Backward

| Sequence length | Eager |
|:---:|:---:|
| 1K | 0.48 ms<br>0.16 GiB |
| 2K | 2.63 ms<br>0.56 GiB |
| 4K | 10.57 ms<br>2.11 GiB |
| 8K | 39.68 ms<br>8.20 GiB |
| 16K | 152.26 ms<br>32.39 GiB |
| 32K | OOM (estimated 129.50 GiB) |

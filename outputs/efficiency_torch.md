# Efficiency benchmark: Torch Manual

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Warmup: 2

## Causal Forward

| Sequence length | Torch Manual |
|:---:|:---:|
| 1K | 37.07 ms<br>0.03 GiB |
| 2K | 143.36 ms<br>0.05 GiB |
| 4K | 580.54 ms<br>0.09 GiB |
| 8K | 2270.80 ms<br>0.17 GiB |
| 16K | 9069.72 ms<br>0.32 GiB |
| 32K | 36397.94 ms<br>0.64 GiB |

## Causal Backward

| Sequence length | Torch Manual |
|:---:|:---:|
| 1K | 46.20 ms<br>0.05 GiB |
| 2K | 182.17 ms<br>0.09 GiB |
| 4K | 726.32 ms<br>0.16 GiB |
| 8K | 2865.09 ms<br>0.30 GiB |
| 16K | 11536.02 ms<br>0.58 GiB |
| 32K | 45928.31 ms<br>1.14 GiB |

## Non-causal Forward

| Sequence length | Torch Manual |
|:---:|:---:|
| 1K | 25.83 ms<br>0.04 GiB |
| 2K | 101.97 ms<br>0.06 GiB |
| 4K | 398.83 ms<br>0.10 GiB |
| 8K | 1594.15 ms<br>0.17 GiB |
| 16K | 6360.37 ms<br>0.33 GiB |
| 32K | 25410.45 ms<br>0.64 GiB |

## Non-causal Backward

| Sequence length | Torch Manual |
|:---:|:---:|
| 1K | 34.29 ms<br>0.05 GiB |
| 2K | 133.45 ms<br>0.09 GiB |
| 4K | 526.63 ms<br>0.16 GiB |
| 8K | 2109.01 ms<br>0.30 GiB |
| 16K | 8406.64 ms<br>0.58 GiB |
| 32K | 33489.93 ms<br>1.14 GiB |

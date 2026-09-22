# Efficiency benchmark: FlashAttentionV2 2.8.3

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Warmup: 2

## Causal Forward

| Sequence length | FlashAttentionV2 |
|:---:|:---:|
| 1K | 0.09 ms<br>0.02 GiB |
| 2K | 0.14 ms<br>0.04 GiB |
| 4K | 0.42 ms<br>0.08 GiB |
| 8K | 1.48 ms<br>0.16 GiB |
| 16K | 5.32 ms<br>0.31 GiB |
| 32K | 22.54 ms<br>0.63 GiB |

## Causal Backward

| Sequence length | FlashAttentionV2 |
|:---:|:---:|
| 1K | 0.32 ms<br>0.04 GiB |
| 2K | 0.35 ms<br>0.08 GiB |
| 4K | 1.20 ms<br>0.16 GiB |
| 8K | 4.36 ms<br>0.31 GiB |
| 16K | 15.78 ms<br>0.63 GiB |
| 32K | 61.75 ms<br>1.25 GiB |

## Non-causal Forward

| Sequence length | FlashAttentionV2 |
|:---:|:---:|
| 1K | 0.09 ms<br>0.02 GiB |
| 2K | 0.17 ms<br>0.04 GiB |
| 4K | 0.63 ms<br>0.08 GiB |
| 8K | 2.39 ms<br>0.16 GiB |
| 16K | 8.90 ms<br>0.31 GiB |
| 32K | 38.36 ms<br>0.63 GiB |

## Non-causal Backward

| Sequence length | FlashAttentionV2 |
|:---:|:---:|
| 1K | 0.49 ms<br>0.04 GiB |
| 2K | 0.59 ms<br>0.08 GiB |
| 4K | 2.12 ms<br>0.16 GiB |
| 8K | 7.95 ms<br>0.31 GiB |
| 16K | 30.14 ms<br>0.63 GiB |
| 32K | 122.93 ms<br>1.25 GiB |

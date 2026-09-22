# Accuracy benchmark: Torch Manual vs SDPA

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Reference: SDPA (Efficient Attention)
- Tolerance: rtol=0.02, atol=0.02
- Error cell: MAE / MRE

## Causal Forward

| Sequence length | Output MAE / MRE | Result |
|:---:|:---:|:---:|
| 1K | 6.316e-05 / 5.726e-03 | PASS |
| 2K | 5.207e-05 / 6.245e-03 | PASS |
| 4K | 4.378e-05 / 6.934e-03 | PASS |
| 8K | 3.803e-05 / 7.820e-03 | PASS |
| 16K | 3.380e-05 / 9.098e-03 | PASS |
| 32K | 3.136e-05 / 1.110e-02 | PASS |

## Causal Backward

| Sequence length | dQ MAE / MRE | dK MAE / MRE | dV MAE / MRE | Result |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 1.312e-04 / 7.509e-02 | 1.141e-04 / 1.423e-02 | 1.162e-04 / 1.613e-02 | PASS |
| 2K | 1.074e-04 / 3.502e-02 | 9.560e-05 / 1.704e-02 | 9.613e-05 / 1.744e-02 | PASS |
| 4K | 9.158e-05 / 4.429e-02 | 8.340e-05 / 2.093e-02 | 8.364e-05 / 2.079e-02 | PASS |
| 8K | 7.508e-05 / 2.272e-02 | 6.903e-05 / 2.252e-02 | 6.925e-05 / 2.269e-02 | PASS |
| 16K | 6.182e-05 / 2.518e-02 | 5.764e-05 / 2.462e-02 | 5.740e-05 / 2.467e-02 | PASS |
| 32K | 5.149e-05 / 2.197e-02 | 4.908e-05 / 2.749e-02 | 4.886e-05 / 2.745e-02 | PASS |

## Non-causal Forward

| Sequence length | Output MAE / MRE | Result |
|:---:|:---:|:---:|
| 1K | 4.453e-05 / 6.472e-03 | PASS |
| 2K | 3.786e-05 / 7.325e-03 | PASS |
| 4K | 3.352e-05 / 8.007e-03 | PASS |
| 8K | 3.091e-05 / 9.395e-03 | PASS |
| 16K | 2.928e-05 / 1.138e-02 | PASS |
| 32K | 2.911e-05 / 1.452e-02 | PASS |

## Non-causal Backward

| Sequence length | dQ MAE / MRE | dK MAE / MRE | dV MAE / MRE | Result |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 9.208e-05 / 1.233e-02 | 9.462e-05 / 1.670e-02 | 9.383e-05 / 1.774e-02 | PASS |
| 2K | 8.660e-05 / 1.299e-02 | 9.172e-05 / 2.325e-02 | 9.078e-05 / 2.334e-02 | PASS |
| 4K | 6.689e-05 / 1.418e-02 | 7.007e-05 / 2.381e-02 | 6.940e-05 / 2.334e-02 | PASS |
| 8K | 5.363e-05 / 1.599e-02 | 5.516e-05 / 2.539e-02 | 5.459e-05 / 2.499e-02 | PASS |
| 16K | 4.464e-05 / 1.875e-02 | 4.478e-05 / 2.793e-02 | 4.438e-05 / 2.752e-02 | PASS |
| 32K | 3.902e-05 / 2.218e-02 | 3.835e-05 / 3.199e-02 | 3.802e-05 / 3.223e-02 | PASS |

# Accuracy benchmark: Triton Manual vs SDPA

- GPU: NVIDIA L40S, CC 8.9, 142 SMs
- Shape: B=1, H=16, D=128
- Dtype: FP16
- Reference: SDPA (Efficient Attention)
- Tolerance: rtol=0.02, atol=0.02
- Error cell: MAE / MRE

## Causal Forward

| Sequence length | Output MAE / MRE | Result |
|:---:|:---:|:---:|
| 1K | 4.032e-06 / 5.148e-04 | PASS |
| 2K | 2.999e-06 / 4.382e-04 | PASS |
| 4K | 2.047e-06 / 4.000e-04 | PASS |
| 8K | 1.346e-06 / 3.296e-04 | PASS |
| 16K | 8.513e-07 / 2.606e-04 | PASS |
| 32K | 5.252e-07 / 2.085e-04 | PASS |

## Causal Backward

| Sequence length | dQ MAE / MRE | dK MAE / MRE | dV MAE / MRE | Result |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 2.363e-05 / 2.659e-03 | 1.918e-05 / 2.319e-03 | 1.428e-07 / 2.064e-05 | PASS |
| 2K | 1.761e-05 / 2.620e-03 | 1.420e-05 / 2.417e-03 | 1.651e-07 / 2.639e-05 | PASS |
| 4K | 1.310e-05 / 2.530e-03 | 1.046e-05 / 2.378e-03 | 1.767e-07 / 4.415e-05 | PASS |
| 8K | 9.545e-06 / 2.507e-03 | 7.582e-06 / 2.384e-03 | 1.783e-07 / 6.126e-05 | PASS |
| 16K | 7.116e-06 / 2.575e-03 | 5.657e-06 / 2.501e-03 | 1.679e-07 / 7.954e-05 | PASS |
| 32K | 5.663e-06 / 2.915e-03 | 4.616e-06 / 3.061e-03 | 1.564e-07 / 1.070e-04 | PASS |

## Non-causal Forward

| Sequence length | Output MAE / MRE | Result |
|:---:|:---:|:---:|
| 1K | 2.513e-06 / 4.801e-04 | PASS |
| 2K | 1.490e-06 / 3.859e-04 | PASS |
| 4K | 8.573e-07 / 2.979e-04 | PASS |
| 8K | 4.855e-07 / 2.312e-04 | PASS |
| 16K | 2.710e-07 / 1.759e-04 | PASS |
| 32K | 1.518e-07 / 1.343e-04 | PASS |

## Non-causal Backward

| Sequence length | dQ MAE / MRE | dK MAE / MRE | dV MAE / MRE | Result |
|:---:|:---:|:---:|:---:|:---:|
| 1K | 1.406e-05 / 2.656e-03 | 1.388e-05 / 2.557e-03 | 2.128e-07 / 4.690e-05 | PASS |
| 2K | 1.012e-05 / 2.510e-03 | 9.985e-06 / 2.531e-03 | 2.258e-07 / 6.024e-05 | PASS |
| 4K | 7.270e-06 / 2.517e-03 | 7.171e-06 / 2.473e-03 | 2.163e-07 / 7.363e-05 | PASS |
| 8K | 5.298e-06 / 2.508e-03 | 5.228e-06 / 2.470e-03 | 1.969e-07 / 9.263e-05 | PASS |
| 16K | 4.256e-06 / 2.769e-03 | 4.215e-06 / 2.759e-03 | 1.812e-07 / 1.179e-04 | PASS |
| 32K | 4.365e-06 / 3.899e-03 | 4.338e-06 / 3.891e-03 | 1.770e-07 / 1.593e-04 | PASS |

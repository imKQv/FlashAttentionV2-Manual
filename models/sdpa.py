"""PyTorch SDPA using the Efficient Attention backend."""

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def attention(q, k, v, is_causal=False):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=is_causal,
        )


__all__ = ["attention"]

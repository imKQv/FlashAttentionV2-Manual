"""Explicit eager PyTorch attention backend."""

import math

import torch


def attention(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        query_length = q.shape[-2]
        key_length = k.shape[-2]
        mask = torch.ones(
            (query_length, key_length),
            device=q.device,
            dtype=torch.bool,
        ).tril_()
        scores.masked_fill_(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


__all__ = ["attention"]

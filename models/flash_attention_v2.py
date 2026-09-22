"""Adapter for Tri Dao's FlashAttention-2 Python package."""

from importlib.metadata import PackageNotFoundError, version


def installed_version():
    for distribution_name in ("flash-attn", "flash_attn"):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    return None


def attention(q, k, v, is_causal=False):
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError(
            "FlashAttentionV2 requires the optional flash-attn package "
            "(benchmark reference version: 2.8.3)."
        ) from exc

    # Repository-wide layout: [B, H, T, D].
    # flash_attn_func layout: [B, T, H, D].
    output = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        softmax_scale=None,
        causal=is_causal,
    )
    return output.transpose(1, 2)


__all__ = ["attention", "installed_version"]

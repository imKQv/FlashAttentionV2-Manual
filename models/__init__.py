"""Lazy model registry for all benchmark backends."""

MODEL_NAMES = (
    "eager",
    "torch",
    "triton",
    "sdpa",
    "flash-attention-v2",
)
ACCURACY_MODEL_NAMES = ("torch", "triton")

DISPLAY_NAMES = {
    "eager": "Eager",
    "torch": "Torch Manual",
    "triton": "Triton Manual",
    "sdpa": "SDPA (Efficient Attention)",
    "flash-attention-v2": "FlashAttentionV2",
}

_ALIASES = {
    "flash_attention_v2": "flash-attention-v2",
    "flash-attention": "flash-attention-v2",
    "flash-attentionv2": "flash-attention-v2",
    "flashattn": "flash-attention-v2",
    "fa2": "flash-attention-v2",
}


def normalize_model_name(name):
    normalized = name.strip().lower()
    return _ALIASES.get(normalized, normalized)


def get_model(name):
    name = normalize_model_name(name)
    if name == "eager":
        from .eager import attention
    elif name == "torch":
        from .flashattnv2_torch import attention
    elif name == "triton":
        from .flashattnv2_triton import attention
    elif name == "sdpa":
        from .sdpa import attention
    elif name == "flash-attention-v2":
        from .flash_attention_v2 import attention
    else:
        choices = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {name!r}; choose one of: {choices}")
    return attention


def model_display_name(name):
    return DISPLAY_NAMES[normalize_model_name(name)]


def model_version(name):
    name = normalize_model_name(name)
    if name != "flash-attention-v2":
        return None
    from .flash_attention_v2 import installed_version
    return installed_version()


__all__ = [
    "ACCURACY_MODEL_NAMES",
    "DISPLAY_NAMES",
    "MODEL_NAMES",
    "get_model",
    "model_display_name",
    "model_version",
    "normalize_model_name",
]

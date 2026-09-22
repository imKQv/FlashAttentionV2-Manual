"""Shared benchmark configuration and utilities."""

import gc
from pathlib import Path

import torch


BATCH_SIZE = 1
NUM_HEADS = 16
HEAD_DIM = 128
SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
DTYPE = torch.float16
WARMUP = 2


def measured_repeats(sequence_length):
    if sequence_length <= 4096:
        return 5
    if sequence_length <= 16384:
        return 3
    return 2


def make_inputs(sequence_length, device, requires_grad):
    return tuple(
        torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            sequence_length,
            HEAD_DIM,
            device=device,
            dtype=DTYPE,
            requires_grad=requires_grad,
        )
        for _ in range(3)
    )


def reset_after_case():
    gc.collect()
    torch.cuda.empty_cache()


def sequence_label(sequence_length):
    return f"{sequence_length // 1024}K"


def gpu_description(device):
    properties = torch.cuda.get_device_properties(device)
    return (
        f"{properties.name}, CC {properties.major}.{properties.minor}, "
        f"{properties.multi_processor_count} SMs"
    )


def write_report(report, output_dir, filename):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / filename
    destination.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {destination}")
    return destination

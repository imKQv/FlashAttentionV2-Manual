#!/usr/bin/env python3
"""Benchmark one FP16 attention backend over four experiment groups."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import (
    BATCH_SIZE,
    DTYPE,
    HEAD_DIM,
    NUM_HEADS,
    SEQUENCE_LENGTHS,
    WARMUP,
    gpu_description,
    make_inputs,
    measured_repeats,
    reset_after_case,
    sequence_label,
    write_report,
)
from models import (
    MODEL_NAMES,
    get_model,
    model_display_name,
    model_version,
    normalize_model_name,
)


def eager_memory_estimate_gib(sequence_length, direction):
    score_bytes = (
        BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * sequence_length
        * torch.empty((), dtype=DTYPE).element_size()
    )
    token_bytes = (
        BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * HEAD_DIM
        * torch.empty((), dtype=DTYPE).element_size()
    )
    score_multiplier = 2 if direction == "forward" else 4
    token_multiplier = 5 if direction == "forward" else 12
    return (score_multiplier * score_bytes + token_multiplier * token_bytes) / (1024**3)


def benchmark_forward(attention, sequence_length, is_causal, device):
    q, k, v = make_inputs(sequence_length, device, requires_grad=False)
    repeats = measured_repeats(sequence_length)

    with torch.inference_mode():
        for _ in range(WARMUP):
            output = attention(q, k, v, is_causal)
        torch.cuda.synchronize(device)
        del output
        torch.cuda.reset_peak_memory_stats(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            output = attention(q, k, v, is_causal)
        end.record()
        end.synchronize()

        elapsed_ms = start.elapsed_time(end) / repeats
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        del output

    del q, k, v
    return elapsed_ms, peak_gib


def benchmark_backward(attention, sequence_length, is_causal, device):
    q, k, v = make_inputs(sequence_length, device, requires_grad=True)
    grad_output = torch.randn_like(q)
    output = attention(q, k, v, is_causal)
    torch.cuda.synchronize(device)
    repeats = measured_repeats(sequence_length)

    for _ in range(WARMUP):
        gradients = torch.autograd.grad(
            output,
            (q, k, v),
            grad_outputs=grad_output,
            retain_graph=True,
        )
        del gradients
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        gradients = torch.autograd.grad(
            output,
            (q, k, v),
            grad_outputs=grad_output,
            retain_graph=True,
        )
        del gradients
    end.record()
    end.synchronize()

    elapsed_ms = start.elapsed_time(end) / repeats
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    del output, grad_output, q, k, v
    return elapsed_ms, peak_gib


def run_case(model_name, attention, sequence_length, is_causal, direction, device):
    if model_name == "eager":
        estimate = eager_memory_estimate_gib(sequence_length, direction)
        total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if estimate > 0.8 * total_gib:
            return {"status": "oom", "estimated_gib": estimate}

    try:
        if direction == "forward":
            elapsed_ms, peak_gib = benchmark_forward(
                attention, sequence_length, is_causal, device
            )
        else:
            elapsed_ms, peak_gib = benchmark_backward(
                attention, sequence_length, is_causal, device
            )
        return {"status": "ok", "ms": elapsed_ms, "peak_gib": peak_gib}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "oom"}
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {' '.join(str(exc).split())}",
        }
    finally:
        reset_after_case()


def format_result(result):
    if result["status"] == "oom":
        if "estimated_gib" in result:
            return f"OOM (estimated {result['estimated_gib']:.2f} GiB)"
        return "OOM"
    if result["status"] == "error":
        return f"ERROR ({result['message'][:120]})"
    return f"{result['ms']:.2f} ms<br>{result['peak_gib']:.2f} GiB"


def build_report(model_name, attention, device):
    display_name = model_display_name(model_name)
    installed_version = model_version(model_name)
    version_text = f" {installed_version}" if installed_version else ""

    lines = [
        f"# Efficiency benchmark: {display_name}{version_text}",
        "",
        (
            f"- GPU: {gpu_description(device)}\n"
            f"- Shape: B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}\n"
            f"- Dtype: FP16\n"
            f"- Warmup: {WARMUP}"
        ),
        "",
    ]

    experiments = (
        ("Causal Forward", True, "forward"),
        ("Causal Backward", True, "backward"),
        ("Non-causal Forward", False, "forward"),
        ("Non-causal Backward", False, "backward"),
    )
    for title, is_causal, direction in experiments:
        lines.extend(
            [
                f"## {title}",
                "",
                f"| Sequence length | {display_name} |",
                "|:---:|:---:|",
            ]
        )
        for sequence_length in SEQUENCE_LENGTHS:
            print(
                f"[{model_name}] {title}, T={sequence_length}",
                file=sys.stderr,
                flush=True,
            )
            result = run_case(
                model_name,
                attention,
                sequence_length,
                is_causal,
                direction,
                device,
            )
            lines.append(
                f"| {sequence_label(sequence_length)} | {format_result(result)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        required=True,
        help="One backend: eager, torch, triton, sdpa, or flash-attention-v2",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.models)
    if model_name not in MODEL_NAMES:
        raise SystemExit(
            f"--models must be one of: {', '.join(MODEL_NAMES)}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    attention = get_model(model_name)
    report = build_report(model_name, attention, device)
    filename = f"efficiency_{model_name}.md"
    write_report(report, args.output_dir, filename)


if __name__ == "__main__":
    main()

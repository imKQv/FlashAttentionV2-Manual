#!/usr/bin/env python3
"""Compare one manual FP16 backend with SDPA over all sequence lengths."""

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
    gpu_description,
    make_inputs,
    reset_after_case,
    sequence_label,
    write_report,
)
from models import (
    ACCURACY_MODEL_NAMES,
    get_model,
    model_display_name,
    normalize_model_name,
)


def error_metrics(actual, reference, rtol, atol):
    actual_fp32 = actual.detach().float()
    reference_fp32 = reference.detach().float()
    difference = (actual_fp32 - reference_fp32).abs()
    mean_absolute = difference.mean().item()
    mean_relative = (
        difference / reference_fp32.abs().clamp_min(1e-6)
    ).mean().item()
    passed = torch.allclose(
        actual_fp32,
        reference_fp32,
        rtol=rtol,
        atol=atol,
    )
    return {
        "mae": mean_absolute,
        "mre": mean_relative,
        "passed": passed,
    }


def run_case(attention, reference_attention, sequence_length, is_causal, device, rtol, atol):
    torch.manual_seed(42 + sequence_length + int(is_causal))
    torch.cuda.manual_seed_all(42 + sequence_length + int(is_causal))

    q, k, v = make_inputs(sequence_length, device, requires_grad=True)
    grad_output = torch.randn_like(q)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    try:
        output = attention(q, k, v, is_causal)
        gradients = torch.autograd.grad(
            output,
            (q, k, v),
            grad_outputs=grad_output,
        )

        reference_output = reference_attention(
            q_ref,
            k_ref,
            v_ref,
            is_causal,
        )
        reference_gradients = torch.autograd.grad(
            reference_output,
            (q_ref, k_ref, v_ref),
            grad_outputs=grad_output,
        )
        torch.cuda.synchronize(device)

        return {
            "status": "ok",
            "forward": error_metrics(
                output, reference_output, rtol, atol
            ),
            "backward": tuple(
                error_metrics(actual, reference, rtol, atol)
                for actual, reference in zip(gradients, reference_gradients)
            ),
        }
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "oom"}
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {' '.join(str(exc).split())}",
        }
    finally:
        del q, k, v, q_ref, k_ref, v_ref, grad_output
        reset_after_case()


def metric_cell(metric):
    return f"{metric['mae']:.3e} / {metric['mre']:.3e}"


def status_cell(result, metrics_key):
    if result["status"] == "oom":
        return "OOM"
    if result["status"] == "error":
        return f"ERROR ({result['message'][:100]})"
    metrics = (
        (result["forward"],)
        if metrics_key == "forward"
        else result["backward"]
    )
    return "PASS" if all(item["passed"] for item in metrics) else "FAIL"


def build_report(model_name, attention, reference_attention, device, rtol, atol):
    display_name = model_display_name(model_name)
    lines = [
        f"# Accuracy benchmark: {display_name} vs SDPA",
        "",
        (
            f"- GPU: {gpu_description(device)}\n"
            f"- Shape: B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}\n"
            f"- Dtype: FP16\n"
            f"- Reference: SDPA (Efficient Attention)\n"
            f"- Tolerance: rtol={rtol}, atol={atol}\n"
            "- Error cell: MAE / MRE"
        ),
        "",
    ]

    all_passed = True
    for mode_name, is_causal in (("Causal", True), ("Non-causal", False)):
        results = []
        for sequence_length in SEQUENCE_LENGTHS:
            print(
                f"[{model_name}] {mode_name}, T={sequence_length}",
                file=sys.stderr,
                flush=True,
            )
            result = run_case(
                attention,
                reference_attention,
                sequence_length,
                is_causal,
                device,
                rtol,
                atol,
            )
            results.append((sequence_length, result))
            if result["status"] != "ok":
                all_passed = False
            elif not result["forward"]["passed"]:
                all_passed = False
            elif not all(metric["passed"] for metric in result["backward"]):
                all_passed = False

        lines.extend(
            [
                f"## {mode_name} Forward",
                "",
                "| Sequence length | Output MAE / MRE | Result |",
                "|:---:|:---:|:---:|",
            ]
        )
        for sequence_length, result in results:
            if result["status"] == "ok":
                metric = result["forward"]
                lines.append(
                    f"| {sequence_label(sequence_length)} | "
                    f"{metric_cell(metric)} | "
                    f"{status_cell(result, 'forward')} |"
                )
            else:
                status = status_cell(result, "forward")
                lines.append(
                    f"| {sequence_label(sequence_length)} | — | {status} |"
                )
        lines.append("")

        lines.extend(
            [
                f"## {mode_name} Backward",
                "",
                "| Sequence length | dQ MAE / MRE | dK MAE / MRE | dV MAE / MRE | Result |",
                "|:---:|:---:|:---:|:---:|:---:|",
            ]
        )
        for sequence_length, result in results:
            if result["status"] == "ok":
                dq, dk, dv = result["backward"]
                lines.append(
                    f"| {sequence_label(sequence_length)} | "
                    f"{metric_cell(dq)} | {metric_cell(dk)} | "
                    f"{metric_cell(dv)} | {status_cell(result, 'backward')} |"
                )
            else:
                status = status_cell(result, "backward")
                lines.append(
                    f"| {sequence_label(sequence_length)} | — | — | — | {status} |"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", all_passed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        required=True,
        help="One manual backend: torch or triton",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.models)
    if model_name not in ACCURACY_MODEL_NAMES:
        raise SystemExit(
            f"--models must be one of: {', '.join(ACCURACY_MODEL_NAMES)}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    attention = get_model(model_name)
    reference_attention = get_model("sdpa")

    report, all_passed = build_report(
        model_name,
        attention,
        reference_attention,
        device,
        args.rtol,
        args.atol,
    )
    filename = f"accuracy_{model_name}.md"
    write_report(report, args.output_dir, filename)
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

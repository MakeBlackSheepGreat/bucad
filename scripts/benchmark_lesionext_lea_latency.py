"""Benchmark frozen model-only inference latency for the LesioNeXt-LEA manuscript."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import load_classifier
from src.utils.config import load_project_config
from src.utils.runtime import optional_import


torch = optional_import("torch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "paper_evidence" / "lesionext_lea_latency_benchmark.json",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def benchmark_model(model, image, *, device, warmup: int, repetitions: int) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(image)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings_ms: list[float] = []
        for _ in range(repetitions):
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            _ = model(image)
            torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(repetitions),
        "latency_ms_mean": float(statistics.fmean(timings_ms)),
        "latency_ms_median": float(statistics.median(timings_ms)),
        "latency_ms_p95": float(np.percentile(timings_ms, 95)),
        "latency_ms_min": float(min(timings_ms)),
        "latency_ms_max": float(max(timings_ms)),
        "peak_memory_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# LesioNeXt-LEA Model-Only Latency Benchmark",
        "",
        "## Protocol",
        "",
        "- Inference mode: `torch.inference_mode()`.",
        "- Input: one synthetic `3 x 224 x 224` FP32 image on the measured device.",
        "- Scope: model-only forward pass. Image loading, CLAHE, device transfer, and post-processing are excluded.",
        "- Timing: CUDA synchronization before and after every timed forward pass.",
        f"- Warm-up: `{report['protocol']['warmup_iterations']}` iterations; timed repetitions: `{report['protocol']['timed_iterations']}`.",
        "",
        "## Environment",
        "",
        f"- Device: `{report['environment']['device_name']}`.",
        f"- PyTorch: `{report['environment']['torch_version']}`; CUDA runtime: `{report['environment']['cuda_version']}`.",
        "",
        "## Results",
        "",
        "| Model | Parameters | Mean ms | Median ms | P95 ms | Peak memory MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["models"].items():
        lines.append(
            f"| {name} | {values['parameter_count']:,} | {values['latency_ms_mean']:.3f} | "
            f"{values['latency_ms_median']:.3f} | {values['latency_ms_p95']:.3f} | {values['peak_memory_mib']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is required for latency benchmarking.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for this manuscript benchmark.")
    if args.warmup < 1 or args.repetitions < 1:
        raise ValueError("--warmup and --repetitions must be positive.")

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(20_260_811)
    image = torch.randn((1, 3, 224, 224), device=device, dtype=torch.float32)
    model_specs = {
        "ConvNeXt-Tiny": {
            "config": PROJECT_ROOT / "configs" / "classifier" / "convnext_tiny_timm_recipe.yml",
            "checkpoint": PROJECT_ROOT / "artifacts" / "checkpoints" / "convnext_tiny_timm_recipe_fold1.pt",
        },
        "LesioNeXt-LEA": {
            "config": PROJECT_ROOT / "configs" / "classifier" / "lesionext_lens_v1a_evidence_only.yml",
            "checkpoint": PROJECT_ROOT / "artifacts" / "checkpoints" / "lesionext_lens_v1a_evidence_only_fold1.pt",
        },
    }
    results = {}
    for name, spec in model_specs.items():
        config, paths = load_project_config(spec["config"])
        model = load_classifier(
            dict(config["model"]),
            checkpoint_path=resolve_path(spec["checkpoint"], paths.project_root),
            map_location="cpu",
        ).to(device)
        model.eval()
        results[name] = benchmark_model(
            model,
            image,
            device=device,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "scope": "Frozen checkpoint, GPU model-only latency benchmark for manuscript reporting",
        "protocol": {
            "batch_size": 1,
            "input_shape": [1, 3, 224, 224],
            "dtype": "float32",
            "inference_context": "torch.inference_mode",
            "warmup_iterations": int(args.warmup),
            "timed_iterations": int(args.repetitions),
            "synchronization": "torch.cuda.synchronize before and after each timed forward pass",
            "included": "model forward pass",
            "excluded": "image loading, CLAHE, host-to-device transfer, and post-processing",
        },
        "environment": {
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(),
        },
        "models": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"json": str(output), "markdown": str(output.with_suffix('.md'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark torchvision matrix vs torch bool-mask GreedyNMM IOS.

This benchmark isolates the path changed by the torch mask implementation:

    GreedyNMM + IOS + class-agnostic predictions

It compares:

    numpy               Pure numpy GreedyNMM IOS baseline
    torchvision_matrix  Old torchvision path: GPU float matrix -> CPU greedy
    torch_mask          New path: GPU bool match mask -> CPU greedy
    triton              Triton packed bitset -> CPU greedy

The run is deterministic for a fixed seed and writes CSV, JSON, and Markdown
reports under benchmarks/results by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import importlib.util
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sahi.postprocess._numpy_backend import greedy_nmm_from_matrix, greedy_nmm_numpy

DEFAULT_OUTPUT_DIR = Path("benchmarks/results")
VARIANTS = ("numpy", "torchvision_matrix", "torch_mask", "triton")
DATA_SEED_OFFSETS = {"clustered": 10_000_000, "random": 20_000_000}


@dataclass(frozen=True)
class BenchRow:
    variant: str
    data: str
    boxes: int
    result_items: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    cv_pct: float
    stable: bool
    parity: bool
    samples_ms: list[float]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "git_commit": git_value(["rev-parse", "--short", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "modules": {},
        "torch": {},
    }
    for module_name in ("numpy", "torch", "torchvision", "triton"):
        info["modules"][module_name] = {
            "installed": has_module(module_name),
            "version": package_version(module_name),
        }

    if has_module("torch"):
        try:
            import torch

            info["torch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            info["torch"] = {"import_error": f"{type(exc).__name__}: {exc}"}
    return info


def print_runtime(info: dict[str, Any]) -> None:
    print("Runtime", flush=True)
    print(f"  python: {info['python']}", flush=True)
    print(f"  platform: {info['platform']}", flush=True)
    print(f"  git: {info.get('git_branch')} @ {info.get('git_commit')}", flush=True)
    for module_name, module_info in info["modules"].items():
        version = module_info["version"] or "-"
        print(f"  {module_name}: installed={module_info['installed']} version={version}", flush=True)
    for key, value in info.get("torch", {}).items():
        print(f"  torch.{key}: {value}", flush=True)


def progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def sync_cuda() -> None:
    if not has_module("torch"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def pack_predictions(
    x1y1: np.ndarray,
    x2y2: np.ndarray,
    rng: np.random.Generator,
    n: int,
    num_classes: int,
) -> np.ndarray:
    scores = rng.uniform(0.05, 0.99, size=(n, 1))
    classes = rng.integers(0, num_classes, size=(n, 1))
    boxes = np.concatenate([x1y1, x2y2], axis=1)
    return np.concatenate([boxes, scores, classes], axis=1).astype(np.float32)


def make_random_predictions(n: int, *, seed: int, num_classes: int, image_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1y1 = rng.uniform(0, image_size * 0.9, size=(n, 2))
    wh = rng.uniform(8, image_size * 0.12, size=(n, 2))
    x2y2 = np.minimum(x1y1 + wh, image_size)
    return pack_predictions(x1y1, x2y2, rng, n, num_classes)


def make_clustered_predictions(n: int, *, seed: int, num_classes: int, image_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers_count = max(1, n // 8)

    centers = rng.uniform(image_size * 0.1, image_size * 0.9, size=(centers_count, 2))
    sizes = rng.uniform(20, image_size * 0.08, size=(centers_count, 2))

    center_ids = rng.integers(0, centers_count, size=n)
    xy = centers[center_ids] + rng.normal(0, 16, size=(n, 2))
    wh = sizes[center_ids] * rng.uniform(0.75, 1.25, size=(n, 2))

    x1y1 = np.clip(xy - wh / 2, 0, image_size - 1)
    x2y2 = np.clip(xy + wh / 2, 1, image_size)
    too_small = x2y2 <= x1y1 + 1
    x2y2 = np.where(too_small, np.minimum(x1y1 + 2, image_size), x2y2)
    return pack_predictions(x1y1, x2y2, rng, n, num_classes)


def make_predictions(data: str, n: int, *, seed: int, num_classes: int, image_size: int) -> np.ndarray:
    case_seed = seed + DATA_SEED_OFFSETS[data] + n
    if data == "clustered":
        return make_clustered_predictions(n, seed=case_seed, num_classes=num_classes, image_size=image_size)
    return make_random_predictions(n, seed=case_seed, num_classes=num_classes, image_size=image_size)


def normalize_result(result: dict[int, list[int]]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple((int(k), tuple(int(x) for x in v)) for k, v in sorted(result.items()))


def result_size(result: dict[int, list[int]]) -> int:
    return len(result) + sum(len(v) for v in result.values())


def run_numpy(predictions: np.ndarray, threshold: float) -> dict[int, list[int]]:
    return greedy_nmm_numpy(predictions, "IOS", threshold)


def run_torchvision_matrix(predictions: np.ndarray, threshold: float) -> dict[int, list[int]]:
    from sahi.postprocess._torchvision_backend import _prepare_matrix_torch

    matrix, sorted_idxs = _prepare_matrix_torch(predictions, "IOS")
    return greedy_nmm_from_matrix(matrix, sorted_idxs, threshold)


def run_torch_mask(predictions: np.ndarray, threshold: float) -> dict[int, list[int]]:
    from sahi.postprocess._torchvision_backend import greedy_nmm_torchvision

    return greedy_nmm_torchvision(predictions, "IOS", threshold)


def run_triton(predictions: np.ndarray, threshold: float) -> dict[int, list[int]]:
    from sahi.postprocess._triton_backend import greedy_nmm_triton

    return greedy_nmm_triton(predictions, "IOS", threshold)


def variant_available(variant: str) -> tuple[bool, str | None]:
    if variant == "numpy":
        return True, None

    try:
        import torch  # noqa: F401
        if variant in ("torchvision_matrix", "torch_mask"):
            import torchvision  # noqa: F401
        if variant == "triton":
            import triton  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if variant == "triton" and not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    return True, None


def get_runner(variant: str) -> Callable[[np.ndarray, float], dict[int, list[int]]]:
    if variant == "numpy":
        return run_numpy
    if variant == "torchvision_matrix":
        return run_torchvision_matrix
    if variant == "torch_mask":
        return run_torch_mask
    if variant == "triton":
        return run_triton
    raise ValueError(f"Unknown variant: {variant}")


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def time_once(
    runner: Callable[[np.ndarray, float], dict[int, list[int]]],
    predictions: np.ndarray,
    threshold: float,
) -> tuple[float, dict[int, list[int]]]:
    sync_cuda()
    start = time.perf_counter()
    result = runner(predictions, threshold)
    sync_cuda()
    return (time.perf_counter() - start) * 1000, result


def benchmark_variant(
    *,
    variant: str,
    data: str,
    predictions: np.ndarray,
    threshold: float,
    warmups: int,
    repeats: int,
    stability_cv_threshold: float,
    reference: tuple[tuple[int, tuple[int, ...]], ...],
    show_progress: bool,
) -> BenchRow:
    runner = get_runner(variant)

    last_result: dict[int, list[int]] | None = None
    for i in range(warmups):
        progress(f"  warmup {i + 1}/{warmups}: variant={variant} data={data} boxes={len(predictions)}", enabled=show_progress)
        _, last_result = time_once(runner, predictions, threshold)

    gc.collect()
    times: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(repeats):
            progress(
                f"  repeat {i + 1}/{repeats}: variant={variant} data={data} boxes={len(predictions)}",
                enabled=show_progress,
            )
            elapsed_ms, last_result = time_once(runner, predictions, threshold)
            times.append(elapsed_ms)
    finally:
        if gc_was_enabled:
            gc.enable()

    if last_result is None:
        raise RuntimeError("Benchmark did not produce a result")

    mean_ms = statistics.fmean(times)
    median_ms = statistics.median(times)
    stdev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    cv_pct = (stdev_ms / mean_ms * 100) if mean_ms else 0.0
    return BenchRow(
        variant=variant,
        data=data,
        boxes=len(predictions),
        result_items=result_size(last_result),
        mean_ms=mean_ms,
        median_ms=median_ms,
        p90_ms=percentile(times, 90),
        min_ms=min(times),
        max_ms=max(times),
        stdev_ms=stdev_ms,
        cv_pct=cv_pct,
        stable=cv_pct <= stability_cv_threshold,
        parity=normalize_result(last_result) == reference,
        samples_ms=times,
    )


def format_float(value: float) -> str:
    return f"{value:.3f}"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    return "\n".join([fmt(headers), "-+-".join("-" * width for width in widths), *[fmt(row) for row in rows]])


def result_table(rows: list[BenchRow]) -> str:
    table_rows = [
        [
            row.data,
            str(row.boxes),
            row.variant,
            format_float(row.median_ms),
            format_float(row.p90_ms),
            f"{row.cv_pct:.1f}",
            "yes" if row.stable else "no",
            "yes" if row.parity else "no",
            str(row.result_items),
        ]
        for row in rows
    ]
    return format_table(
        ["data", "boxes", "variant", "median_ms", "p90_ms", "cv_pct", "stable", "parity", "result_items"],
        table_rows,
    )


def speedup_table(rows: list[BenchRow]) -> str:
    grouped: dict[tuple[str, int], dict[str, BenchRow]] = {}
    for row in rows:
        grouped.setdefault((row.data, row.boxes), {})[row.variant] = row

    table_rows: list[list[str]] = []
    for (data, boxes), by_variant in sorted(grouped.items()):
        baseline = by_variant.get("torchvision_matrix")
        mask = by_variant.get("torch_mask")
        triton_row = by_variant.get("triton")
        numpy = by_variant.get("numpy")
        matrix_to_mask = baseline.median_ms / mask.median_ms if baseline and mask and mask.median_ms else None
        matrix_to_triton = (
            baseline.median_ms / triton_row.median_ms if baseline and triton_row and triton_row.median_ms else None
        )
        mask_to_triton = mask.median_ms / triton_row.median_ms if mask and triton_row and triton_row.median_ms else None
        numpy_to_triton = (
            numpy.median_ms / triton_row.median_ms if numpy and triton_row and triton_row.median_ms else None
        )
        table_rows.append(
            [
                data,
                str(boxes),
                format_float(matrix_to_mask) if matrix_to_mask is not None else "-",
                format_float(matrix_to_triton) if matrix_to_triton is not None else "-",
                format_float(mask_to_triton) if mask_to_triton is not None else "-",
                format_float(numpy_to_triton) if numpy_to_triton is not None else "-",
                "yes" if mask and mask.parity else "no" if mask else "-",
                "yes" if triton_row and triton_row.parity else "no" if triton_row else "-",
            ]
        )
    return format_table(
        [
            "data",
            "boxes",
            "mask_vs_matrix",
            "triton_vs_matrix",
            "triton_vs_mask",
            "triton_vs_numpy",
            "mask_parity",
            "triton_parity",
        ],
        table_rows,
    )


def write_csv(path: Path, rows: list[BenchRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else [field.name for field in BenchRow.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = asdict(row)
            row_dict["samples_ms"] = json.dumps(row.samples_ms)
            writer.writerow(row_dict)


def write_json(path: Path, rows: list[BenchRow], runtime: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "runtime": runtime,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, rows: list[BenchRow], runtime: dict[str, Any]) -> None:
    lines = [
        "# Torch Mask GreedyNMM IOS Benchmark",
        "",
        "## Runtime",
        "",
        f"- python: {runtime['python']}",
        f"- platform: {runtime['platform']}",
        f"- git: {runtime.get('git_branch')} @ {runtime.get('git_commit')}",
    ]
    for module_name, module_info in runtime["modules"].items():
        lines.append(f"- {module_name}: installed={module_info['installed']} version={module_info['version'] or '-'}")
    for key, value in runtime.get("torch", {}).items():
        lines.append(f"- torch.{key}: {value}")

    lines.extend(["", "## Speedups", "", "```text", speedup_table(rows), "```", "", "## Results", "", "```text", result_table(rows), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2000, 5000, 10000])
    parser.add_argument("--data", nargs="+", choices=sorted(DATA_SEED_OFFSETS), default=["clustered"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--stability-cv-threshold", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fail-on-noisy", action="store_true")
    parser.add_argument("--fail-on-parity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = runtime_info()
    print_runtime(runtime)

    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        git_commit = runtime.get("git_commit") or "nogit"
        run_name = f"torch_mask_greedy_nmm_ios_{timestamp}_{git_commit}"

    show_progress = not args.no_progress
    rows: list[BenchRow] = []
    skipped: list[str] = []

    for data in args.data:
        for size in args.sizes:
            predictions = make_predictions(
                data,
                size,
                seed=args.seed,
                num_classes=args.num_classes,
                image_size=args.image_size,
            )
            reference = normalize_result(run_numpy(predictions, args.threshold))

            for variant in args.variants:
                available, reason = variant_available(variant)
                if not available:
                    message = f"skip variant={variant}: {reason}"
                    skipped.append(message)
                    progress(message, enabled=show_progress)
                    continue

                progress(f"case: variant={variant} data={data} boxes={size}", enabled=show_progress)
                row = benchmark_variant(
                    variant=variant,
                    data=data,
                    predictions=predictions,
                    threshold=args.threshold,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    stability_cv_threshold=args.stability_cv_threshold,
                    reference=reference,
                    show_progress=show_progress,
                )
                rows.append(row)

    if skipped:
        print("\nSkipped", flush=True)
        for message in skipped:
            print(f"  {message}", flush=True)

    print("\nSpeedups", flush=True)
    print(speedup_table(rows) or "No benchmark rows.", flush=True)
    print("\nResults", flush=True)
    print(result_table(rows) or "No benchmark rows.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{run_name}.csv"
    json_path = args.output_dir / f"{run_name}.json"
    md_path = args.output_dir / f"{run_name}.md"
    write_csv(csv_path, rows)
    write_json(json_path, rows, runtime, args)
    write_markdown(md_path, rows, runtime)

    print("\nSaved", flush=True)
    print(f"  csv: {csv_path}", flush=True)
    print(f"  json: {json_path}", flush=True)
    print(f"  markdown: {md_path}", flush=True)

    if args.fail_on_parity and any(not row.parity for row in rows):
        return 2
    if args.fail_on_noisy and any(not row.stable for row in rows):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

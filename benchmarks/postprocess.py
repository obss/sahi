"""Benchmark SAHI postprocess backends.

This script measures the raw ndarray-level postprocess functions:

    predictions[N, 6] -> kept indices or keep-to-merge mapping

It intentionally excludes model inference, image IO, and ObjectPrediction
construction so backend changes can be measured directly.

The run is deterministic for a fixed seed and writes durable reports:

    benchmarks/results/<run-name>.csv
    benchmarks/results/<run-name>.json
    benchmarks/results/<run-name>.md

Examples:
    python benchmarks/postprocess.py
    python benchmarks/postprocess.py --backends numpy numba torchvision --sizes 1000 2000 5000
    python benchmarks/postprocess.py --ops greedy_nmm --metrics IOS --class-modes agnostic
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
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sahi.postprocess.backends import set_postprocess_backend
from sahi.postprocess.combine import (
    batched_greedy_nmm,
    batched_nmm,
    batched_nms,
    greedy_nmm,
    nmm,
    nms,
)

OPS: dict[str, dict[str, Callable[..., Any]]] = {
    "nms": {"agnostic": nms, "per_class": batched_nms},
    "greedy_nmm": {"agnostic": greedy_nmm, "per_class": batched_greedy_nmm},
    "nmm": {"agnostic": nmm, "per_class": batched_nmm},
}

DATA_SEED_OFFSETS = {"clustered": 10_000_000, "random": 20_000_000}
DEFAULT_OUTPUT_DIR = Path("benchmarks/results")


@dataclass(frozen=True)
class BenchRow:
    backend: str
    op: str
    metric: str
    class_mode: str
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
    samples_ms: list[float]


@dataclass(frozen=True)
class CaseSummary:
    data: str
    class_mode: str
    op: str
    metric: str
    boxes: int
    best_backend: str
    best_median_ms: float
    best_p90_ms: float
    speedup_vs_numpy: float | None
    speedup_vs_second: float | None
    noisy_backends: list[str]


def progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


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

    for module_name in ("numpy", "numba", "torch", "torchvision", "triton"):
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
    if info.get("torch"):
        torch_info = info["torch"]
        for key, value in torch_info.items():
            print(f"  torch.{key}: {value}", flush=True)


def sync_cuda() -> None:
    """Synchronize CUDA timers when torch+CUDA is available."""
    if not has_module("torch"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def case_seed(base_seed: int, data_name: str, boxes: int) -> int:
    return base_seed + DATA_SEED_OFFSETS[data_name] + boxes


def make_random_predictions(n: int, *, seed: int, num_classes: int, image_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1y1 = rng.uniform(0, image_size * 0.9, size=(n, 2))
    wh = rng.uniform(8, image_size * 0.12, size=(n, 2))
    x2y2 = np.minimum(x1y1 + wh, image_size)
    return pack_predictions(x1y1, x2y2, rng, n, num_classes)


def make_clustered_predictions(n: int, *, seed: int, num_classes: int, image_size: int) -> np.ndarray:
    """Generate SAHI-like duplicate boxes around repeated object centers."""
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


def result_size(result: Any) -> int:
    if isinstance(result, dict):
        return len(result) + sum(len(v) for v in result.values())
    return len(result)


def normalize_result(result: Any) -> tuple[Any, ...]:
    if isinstance(result, dict):
        return tuple((int(k), tuple(int(x) for x in v)) for k, v in sorted(result.items()))
    return tuple(int(x) for x in result)


def time_once(func: Callable[..., Any], preds: np.ndarray, metric: str, threshold: float) -> tuple[float, Any]:
    sync_cuda()
    start = time.perf_counter()
    result = func(preds, match_metric=metric, match_threshold=threshold)
    sync_cuda()
    return (time.perf_counter() - start) * 1000, result


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def benchmark_case(
    *,
    backend: str,
    op: str,
    metric: str,
    class_mode: str,
    data_name: str,
    preds: np.ndarray,
    threshold: float,
    warmups: int,
    repeats: int,
    stability_cv_threshold: float,
    show_progress: bool,
) -> tuple[BenchRow | None, Any | None, str | None]:
    set_postprocess_backend(backend)
    func = OPS[op][class_mode]

    try:
        last_result = None
        for i in range(warmups):
            progress(
                f"  warmup {i + 1}/{warmups}: backend={backend} op={op} metric={metric} "
                f"class_mode={class_mode} data={data_name} boxes={len(preds)}",
                enabled=show_progress,
            )
            _, last_result = time_once(func, preds, metric, threshold)

        gc.collect()
        times: list[float] = []
        was_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            for i in range(repeats):
                progress(
                    f"  repeat {i + 1}/{repeats}: backend={backend} op={op} metric={metric} "
                    f"class_mode={class_mode} data={data_name} boxes={len(preds)}",
                    enabled=show_progress,
                )
                elapsed_ms, last_result = time_once(func, preds, metric, threshold)
                times.append(elapsed_ms)
        finally:
            if was_gc_enabled:
                gc.enable()

        assert last_result is not None
        mean_ms = statistics.fmean(times)
        stdev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        cv_pct = (stdev_ms / mean_ms * 100) if mean_ms else 0.0
        row = BenchRow(
            backend=backend,
            op=op,
            metric=metric,
            class_mode=class_mode,
            data=data_name,
            boxes=len(preds),
            result_items=result_size(last_result),
            mean_ms=mean_ms,
            median_ms=statistics.median(times),
            p90_ms=percentile(times, 90),
            min_ms=min(times),
            max_ms=max(times),
            stdev_ms=stdev_ms,
            cv_pct=cv_pct,
            stable=cv_pct <= stability_cv_threshold,
            samples_ms=[round(value, 6) for value in times],
        )
        stability = "stable" if row.stable else f"noisy cv={row.cv_pct:.1f}%"
        progress(
            f"done: backend={backend} op={op} metric={metric} class_mode={class_mode} "
            f"data={data_name} boxes={len(preds)} median_ms={row.median_ms:.3f} "
            f"p90_ms={row.p90_ms:.3f} {stability}",
            enabled=show_progress,
        )
        return row, last_result, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def case_key(row: BenchRow) -> tuple[str, str, str, str, int]:
    return (row.data, row.class_mode, row.op, row.metric, row.boxes)


def summarize(rows: list[BenchRow]) -> list[CaseSummary]:
    grouped: dict[tuple[str, str, str, str, int], list[BenchRow]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)

    summaries: list[CaseSummary] = []
    for key, group in sorted(grouped.items()):
        sorted_group = sorted(group, key=lambda item: item.median_ms)
        best = sorted_group[0]
        numpy_row = next((row for row in group if row.backend == "numpy"), None)
        second = sorted_group[1] if len(sorted_group) > 1 else None
        summaries.append(
            CaseSummary(
                data=key[0],
                class_mode=key[1],
                op=key[2],
                metric=key[3],
                boxes=key[4],
                best_backend=best.backend,
                best_median_ms=best.median_ms,
                best_p90_ms=best.p90_ms,
                speedup_vs_numpy=(numpy_row.median_ms / best.median_ms if numpy_row and best.median_ms else None),
                speedup_vs_second=(second.median_ms / best.median_ms if second and best.median_ms else None),
                noisy_backends=[row.backend for row in group if not row.stable],
            )
        )
    return summaries


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_speedup(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    row_lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def best_backend_table(summaries: list[CaseSummary]) -> str:
    rows = []
    for summary in summaries:
        noisy = ",".join(summary.noisy_backends) if summary.noisy_backends else "-"
        rows.append(
            [
                f"{summary.data}/{summary.class_mode}/{summary.op}/{summary.metric}",
                str(summary.boxes),
                summary.best_backend,
                fmt_ms(summary.best_median_ms),
                fmt_ms(summary.best_p90_ms),
                fmt_speedup(summary.speedup_vs_numpy),
                fmt_speedup(summary.speedup_vs_second),
                noisy,
            ]
        )
    return format_table(
        ["case", "boxes", "best", "median", "p90", "vs_numpy", "vs_2nd", "noisy"],
        rows,
    )


def backend_medians_table(rows: list[BenchRow], backends: list[str]) -> str:
    grouped: dict[tuple[str, str, str, str, int], list[BenchRow]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)

    table_rows = []
    for key, group in sorted(grouped.items()):
        by_backend = {row.backend: row for row in group}
        table_rows.append(
            [
                f"{key[0]}/{key[1]}/{key[2]}/{key[3]}",
                str(key[4]),
                *[fmt_ms(by_backend[backend].median_ms) if backend in by_backend else "-" for backend in backends],
            ]
        )
    return format_table(["case", "boxes", *backends], table_rows)


def row_to_csv_dict(row: BenchRow) -> dict[str, Any]:
    data = asdict(row)
    data["samples_ms"] = ";".join(f"{value:.6f}" for value in row.samples_ms)
    return data


def write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def write_csv(path: Path, rows: list[BenchRow]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(row_to_csv_dict(rows[0]).keys()) if rows else list(BenchRow.__dataclass_fields__)
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv_dict(row))
    tmp_path.replace(path)


def markdown_report(
    *,
    run_name: str,
    runtime: dict[str, Any],
    args: argparse.Namespace,
    rows: list[BenchRow],
    summaries: list[CaseSummary],
    skipped: list[str],
    parity_failures: list[str],
    backends: list[str],
) -> str:
    lines = [
        f"# SAHI Postprocess Backend Benchmark: `{run_name}`",
        "",
        "## Runtime",
        "",
        f"- Python: `{runtime['python']}`",
        f"- Platform: `{runtime['platform']}`",
        f"- Git: `{runtime.get('git_branch')}` @ `{runtime.get('git_commit')}`",
        f"- Command: `{runtime['command']}`",
        f"- Seed: `{args.seed}`",
        f"- Warmups / repeats: `{args.warmups}` / `{args.repeats}`",
        f"- Stability CV threshold: `{args.stability_cv_threshold}%`",
        "",
        "## Best Backend By Case",
        "",
        "```text",
        best_backend_table(summaries),
        "```",
        "",
        "## Backend Median Times (ms)",
        "",
        "```text",
        backend_medians_table(rows, backends),
        "```",
        "",
        "## Notes",
        "",
        "- Median is the primary timing statistic.",
        "- p90/min/max/stdev/CV and raw samples are available in the CSV/JSON outputs.",
        "- `noisy` marks rows whose coefficient of variation exceeded the configured threshold.",
    ]

    if skipped:
        lines.extend(["", "## Skipped Cases", ""])
        lines.extend(f"- {item}" for item in skipped)

    if parity_failures:
        lines.extend(["", "## Parity Failures", ""])
        lines.extend(f"- {item}" for item in parity_failures)

    return "\n".join(lines) + "\n"


def persist_results(
    *,
    output_dir: Path,
    run_name: str,
    runtime: dict[str, Any],
    args: argparse.Namespace,
    rows: list[BenchRow],
    summaries: list[CaseSummary],
    skipped: list[str],
    parity_failures: list[str],
    backends: list[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}.csv"
    json_path = output_dir / f"{run_name}.json"
    md_path = output_dir / f"{run_name}.md"

    args_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {
        "run_name": run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "args": args_payload,
        "rows": [asdict(row) for row in rows],
        "summaries": [asdict(summary) for summary in summaries],
        "skipped": skipped,
        "parity_failures": parity_failures,
    }

    write_csv(csv_path, rows)
    write_json_atomic(json_path, payload)
    write_text_atomic(
        md_path,
        markdown_report(
            run_name=run_name,
            runtime=runtime,
            args=args,
            rows=rows,
            summaries=summaries,
            skipped=skipped,
            parity_failures=parity_failures,
            backends=backends,
        ),
    )
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def default_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    commit = git_value(["rev-parse", "--short", "HEAD"]) or "nogit"
    return f"postprocess_{timestamp}_{commit}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["numpy", "numba", "torchvision"])
    parser.add_argument("--ops", nargs="+", choices=sorted(OPS), default=["nms", "greedy_nmm", "nmm"])
    parser.add_argument("--metrics", nargs="+", choices=["IOU", "IOS"], default=["IOU", "IOS"])
    parser.add_argument(
        "--class-modes",
        nargs="+",
        choices=["agnostic", "per_class"],
        default=["agnostic"],
    )
    parser.add_argument("--data", nargs="+", choices=["clustered", "random"], default=["clustered"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 1000, 2000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-parity", action="store_true")
    parser.add_argument("--stability-cv-threshold", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-save", action="store_true", help="Print report only; do not save CSV/JSON/Markdown files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-case progress logs on stderr.")
    parser.add_argument("--fail-on-noisy", action="store_true", help="Exit non-zero when any row exceeds CV threshold.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    if args.repeats < 3:
        warnings.append("repeats < 3 makes p90/stability weak; use at least 10 for PR evidence.")
    if args.warmups < 1:
        warnings.append("warmups < 1 may include import/JIT/CUDA initialization in timed samples.")
    return warnings


def main() -> int:
    args = parse_args()
    run_name = args.run_name or default_run_name()
    runtime = runtime_info()
    warnings = validate_args(args)

    print_runtime(runtime)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr, flush=True)

    rows: list[BenchRow] = []
    skipped: list[str] = []
    parity_failures: list[str] = []

    for data_name in args.data:
        make_predictions = make_clustered_predictions if data_name == "clustered" else make_random_predictions
        for n in args.sizes:
            preds = make_predictions(
                n,
                seed=case_seed(args.seed, data_name, n),
                num_classes=args.num_classes,
                image_size=args.image_size,
            )
            for op in args.ops:
                for metric in args.metrics:
                    for class_mode in args.class_modes:
                        baseline = None
                        baseline_backend = None
                        for backend in args.backends:
                            row, result, error = benchmark_case(
                                backend=backend,
                                op=op,
                                metric=metric,
                                class_mode=class_mode,
                                data_name=data_name,
                                preds=preds,
                                threshold=args.threshold,
                                warmups=args.warmups,
                                repeats=args.repeats,
                                stability_cv_threshold=args.stability_cv_threshold,
                                show_progress=not args.no_progress,
                            )
                            if error is not None:
                                skip_msg = (
                                    f"backend={backend} op={op} metric={metric} "
                                    f"class_mode={class_mode} data={data_name} boxes={n}: {error}"
                                )
                                progress(f"skip: {skip_msg}", enabled=not args.no_progress)
                                skipped.append(skip_msg)
                                continue

                            assert row is not None
                            assert result is not None
                            rows.append(row)

                            if args.check_parity:
                                normalized = normalize_result(result)
                                if baseline is None:
                                    baseline = normalized
                                    baseline_backend = backend
                                elif normalized != baseline:
                                    parity_failures.append(
                                        f"{backend} != {baseline_backend} for op={op} metric={metric} "
                                        f"class_mode={class_mode} data={data_name} boxes={n}"
                                    )

    summaries = summarize(rows)

    print("\nBest Backend By Case")
    print(best_backend_table(summaries) or "No completed benchmark rows.")
    print("\nBackend Median Times (ms)")
    print(backend_medians_table(rows, args.backends) or "No completed benchmark rows.")

    if skipped:
        print("\nSkipped Cases")
        for item in skipped:
            print(f"  {item}")

    if parity_failures:
        print("\nParity Failures")
        for item in parity_failures:
            print(f"  {item}")

    if not args.no_save:
        paths = persist_results(
            output_dir=args.output_dir,
            run_name=run_name,
            runtime=runtime,
            args=args,
            rows=rows,
            summaries=summaries,
            skipped=skipped,
            parity_failures=parity_failures,
            backends=args.backends,
        )
        print("\nSaved Results")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")

    if parity_failures:
        return 1

    if args.fail_on_noisy and any(not row.stable for row in rows):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark SAHI postprocess backends.

This script measures the raw ndarray-level postprocess functions:

    predictions[N, 6] -> kept indices or keep-to-merge mapping

It intentionally excludes model inference, image IO, and ObjectPrediction
construction so backend changes can be measured directly.

Examples:
    python benchmarks/postprocess_backends.py
    python benchmarks/postprocess_backends.py --backends numpy numba torchvision --sizes 1000 2000 5000
    python benchmarks/postprocess_backends.py --ops greedy_nmm --metrics IOS --class-modes agnostic
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
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


def progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def describe_runtime() -> None:
    print(f"python={sys.version.split()[0]} platform={platform.platform()}", flush=True)
    for module_name in ("numpy", "numba", "torch", "torchvision", "triton"):
        print(f"{module_name}_installed={has_module(module_name)}", flush=True)

    if not has_module("torch"):
        return

    try:
        import torch

        print(f"torch_version={torch.__version__}", flush=True)
        print(f"torch_cuda_available={torch.cuda.is_available()}", flush=True)
        print(f"torch_cuda_device_count={torch.cuda.device_count()}", flush=True)
        if torch.cuda.is_available():
            print(f"torch_cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:
        print(f"torch_import_error={type(exc).__name__}: {exc}", flush=True)


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
        times = []
        for i in range(repeats):
            progress(
                f"  repeat {i + 1}/{repeats}: backend={backend} op={op} metric={metric} "
                f"class_mode={class_mode} data={data_name} boxes={len(preds)}",
                enabled=show_progress,
            )
            elapsed_ms, last_result = time_once(func, preds, metric, threshold)
            times.append(elapsed_ms)

        assert last_result is not None
        row = BenchRow(
            backend=backend,
            op=op,
            metric=metric,
            class_mode=class_mode,
            data=data_name,
            boxes=len(preds),
            result_items=result_size(last_result),
            mean_ms=statistics.fmean(times),
            median_ms=statistics.median(times),
            p90_ms=percentile(times, 90),
            min_ms=min(times),
            max_ms=max(times),
        )
        progress(
            f"done: backend={backend} op={op} metric={metric} class_mode={class_mode} "
            f"data={data_name} boxes={len(preds)} median_ms={row.median_ms:.3f} "
            f"p90_ms={row.p90_ms:.3f}",
            enabled=show_progress,
        )
        return row, last_result, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def print_rows(rows: list[BenchRow]) -> None:
    if not rows:
        return

    header = (
        "backend,op,metric,class_mode,data,boxes,result_items,"
        "mean_ms,median_ms,p90_ms,min_ms,max_ms"
    )
    print(header)
    for row in rows:
        print(
            f"{row.backend},{row.op},{row.metric},{row.class_mode},{row.data},{row.boxes},"
            f"{row.result_items},{row.mean_ms:.3f},{row.median_ms:.3f},"
            f"{row.p90_ms:.3f},{row.min_ms:.3f},{row.max_ms:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["numpy", "numba", "torchvision"])
    parser.add_argument("--ops", nargs="+", choices=sorted(OPS), default=["nms", "greedy_nmm", "nmm"])
    parser.add_argument("--metrics", nargs="+", choices=["IOU", "IOS"], default=["IOU", "IOS"])
    parser.add_argument("--class-modes", nargs="+", choices=["agnostic", "per_class"], default=["agnostic"])
    parser.add_argument("--data", nargs="+", choices=["clustered", "random"], default=["clustered"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 1000, 2000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-parity", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-case progress logs on stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    describe_runtime()

    rows: list[BenchRow] = []
    skipped: list[str] = []
    parity_failures: list[str] = []

    for data_name in args.data:
        make_predictions = make_clustered_predictions if data_name == "clustered" else make_random_predictions
        for n in args.sizes:
            preds = make_predictions(n, seed=args.seed + n, num_classes=args.num_classes, image_size=args.image_size)
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
                                show_progress=not args.no_progress,
                            )
                            if error is not None:
                                skip_msg = (
                                    f"skip backend={backend} op={op} metric={metric} "
                                    f"class_mode={class_mode} data={data_name} boxes={n}: {error}"
                                )
                                progress(skip_msg, enabled=not args.no_progress)
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

    print_rows(rows)

    if skipped:
        print("\nSkipped cases:")
        for item in skipped:
            print(item)

    if parity_failures:
        print("\nParity failures:")
        for item in parity_failures:
            print(item)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark the crowded-input path of SAHI's NumPy NMM implementation.

The generated boxes intentionally occupy a small area, so many of them overlap
and ``nmm_numpy`` selects the row-streaming STRtree path. Input generation and
density estimation are excluded from the measured NMM time.

Examples:

    python benchmarks/crowded_nmm.py
    python benchmarks/crowded_nmm.py --boxes 10000 20000 33337 --repeat 3
    python benchmarks/crowded_nmm.py --boxes 4000 --profile
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import platform
import pstats
import statistics
import time
from dataclasses import dataclass

import numpy as np

from sahi.postprocess._numpy_backend import nmm_numpy
from sahi.postprocess._sparse_backend import estimate_mean_degree, should_stream_nmm, should_use_sparse


@dataclass(frozen=True)
class Result:
    boxes: int
    mean_degree: float
    estimated_intersections: int
    route: str
    median_seconds: float
    minimum_seconds: float
    keepers: int
    merged: int
    largest_group: int


def make_crowded_predictions(n: int, *, spread: float, seed: int) -> np.ndarray:
    """Create the same crowded synthetic layout used by sparse backend tests."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, spread, n)
    y1 = rng.uniform(0, spread, n)
    width = rng.uniform(1, 40, n)
    height = rng.uniform(1, 40, n)

    # Rounded scores exercise SAHI's deterministic coordinate tie-break.
    scores = np.round(rng.uniform(0, 1, n), 2)
    categories = np.zeros(n)
    predictions = np.stack([x1, y1, x1 + width, y1 + height, scores, categories], axis=1)
    return np.ascontiguousarray(predictions, dtype=np.float32)


def validate_partition(groups: dict[int, list[int]], n: int) -> None:
    """Check that every prediction occurs in exactly one NMM group."""
    keepers = set(groups)
    merged: set[int] = set()

    for keeper, members in groups.items():
        if keeper in members:
            raise AssertionError(f"keeper {keeper} appears in its own merge list")
        for member in members:
            if member in keepers:
                raise AssertionError(f"index {member} is both a keeper and a merged box")
            if member in merged:
                raise AssertionError(f"index {member} occurs in more than one merge list")
            merged.add(member)

    assigned = keepers | merged
    if len(assigned) != n:
        missing = n - len(assigned)
        raise AssertionError(f"NMM result does not cover the input ({missing} boxes missing)")


def run_case(
    predictions: np.ndarray,
    *,
    metric: str,
    threshold: float,
    repeat: int,
    profile: bool,
    profile_lines: int,
) -> tuple[Result, str | None]:
    """Time one input size and optionally collect a cProfile report."""
    boxes = predictions[:, :4]
    mean_degree = estimate_mean_degree(boxes)
    estimated_intersections = round(mean_degree * len(boxes))
    if threshold <= 0:
        route = "match-all"
    elif not should_use_sparse(len(boxes), threshold):
        route = "dense"
    else:
        route = "streaming" if should_stream_nmm(boxes) else "stored CSR"

    samples: list[float] = []
    groups: dict[int, list[int]] = {}
    profiler = cProfile.Profile() if profile else None

    for sample_index in range(repeat):
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        start = time.perf_counter()
        try:
            if profiler is not None and sample_index == 0:
                groups = profiler.runcall(nmm_numpy, predictions, metric, threshold)
            else:
                groups = nmm_numpy(predictions, metric, threshold)
        finally:
            elapsed = time.perf_counter() - start
            if gc_was_enabled:
                gc.enable()
        samples.append(elapsed)

    validate_partition(groups, len(predictions))
    merged = sum(len(members) for members in groups.values())
    largest_group = max((len(members) + 1 for members in groups.values()), default=0)

    profile_report = None
    if profiler is not None:
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(profile_lines)
        profile_report = output.getvalue()

    result = Result(
        boxes=len(predictions),
        mean_degree=mean_degree,
        estimated_intersections=estimated_intersections,
        route=route,
        median_seconds=statistics.median(samples),
        minimum_seconds=min(samples),
        keepers=len(groups),
        merged=merged,
        largest_group=largest_group,
    )
    return result, profile_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=int, nargs="+", default=[4000], help="input sizes (default: 4000)")
    parser.add_argument("--spread", type=float, default=80.0, help="coordinate spread; lower is denser (default: 80)")
    parser.add_argument("--metric", choices=("IOU", "IOS"), default="IOS")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--repeat", type=int, default=1, help="timed repetitions per size (default: 1)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", action="store_true", help="profile the first repetition of each size")
    parser.add_argument("--profile-lines", type=int, default=25, help="number of profiler rows to print")
    args = parser.parse_args()

    if any(n <= 0 for n in args.boxes):
        parser.error("--boxes values must be positive")
    if args.spread <= 0:
        parser.error("--spread must be positive")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.profile_lines <= 0:
        parser.error("--profile-lines must be positive")
    return args


def print_result(result: Result, *, repeat: int) -> None:
    # Intersections are an upper bound on thresholded matches. This estimates
    # only the final indptr/indices arrays, not temporary CSR build memory.
    estimated_csr_mib = ((result.boxes + 1 + result.estimated_intersections) * np.dtype(np.intp).itemsize) / 2**20
    print(f"\n{result.boxes:,} boxes")
    print(f"  selected route:       {result.route}")
    print(f"  mean intersections:   {result.mean_degree:,.1f} per box")
    print(f"  est. intersections:   {result.estimated_intersections:,}")
    print(f"  final CSR arrays max: {estimated_csr_mib:,.1f} MiB")
    print(f"  median of {repeat}:        {result.median_seconds:,.3f} s")
    print(f"  minimum:              {result.minimum_seconds:,.3f} s")
    print(f"  groups:               {result.keepers:,}")
    print(f"  merged boxes:         {result.merged:,}")
    print(f"  largest group:        {result.largest_group:,}")


def main() -> None:
    args = parse_args()
    print(
        f"Crowded NMM benchmark: metric={args.metric}, threshold={args.threshold}, "
        f"spread={args.spread}, seed={args.seed}"
    )
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, machine={platform.machine()}")
    print("Input generation and density estimation are not included in NMM timings.")
    if args.profile:
        print("Profiling is enabled, so the reported first repetition includes cProfile overhead.")

    for n in args.boxes:
        predictions = make_crowded_predictions(n, spread=args.spread, seed=args.seed)
        result, profile_report = run_case(
            predictions,
            metric=args.metric,
            threshold=args.threshold,
            repeat=args.repeat,
            profile=args.profile,
            profile_lines=args.profile_lines,
        )
        print_result(result, repeat=args.repeat)
        if profile_report is not None:
            print("\n  cProfile (sorted by cumulative time):")
            print(profile_report.rstrip())


if __name__ == "__main__":
    main()

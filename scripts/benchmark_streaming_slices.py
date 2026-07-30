"""Benchmark streaming slice reads against the eager whole-image decode.

Measures peak RSS and wall time for `SliceImageStream` (reads one row-band at a time)
against `slice_image` (decodes the image, then materialises every slice). Each
measurement runs in a fresh child process, because `ru_maxrss` is a high-water mark that
never comes back down: measuring both paths in one process would report the larger.

The eager path allocates the full decoded image plus every slice, so it is expected to
fail with `MemoryError` on large sizes. That is a result, not a crash, and is reported
as `OOM`.

Usage:
    python scripts/benchmark_streaming_slices.py                 # 1..256 MP sweep
    python scripts/benchmark_streaming_slices.py --mp 1 16 64    # pick sizes
    python scripts/benchmark_streaming_slices.py --image scan.jpg
    python scripts/benchmark_streaming_slices.py --keep-images   # reuse across runs

Requires `pip install sahi[bigimage]` for the streaming column to mean anything; without
libvips it silently measures the fallback, and the script says so up front.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

SLICE = 640
OVERLAP = 0.2
BATCH_SIZE = 8
DEFAULT_MP = [1, 16, 64, 144, 256]

# Run one path over one image and print "<seconds> <peak_kib>". Kept as source rather
# than imported so the child starts clean, with no sahi module already resident.
_PROBE = """
import resource, sys, time
path, mode = sys.argv[1], sys.argv[2]

def consume():
    # GIL-holding stand-in for per-batch inference. Deliberately pure Python: a real
    # model releases the GIL, so anything prefetch wins here it wins by more in practice.
    total = 0
    for _ in range({consume_iters}):
        total = (total + 1) & 255
    return total

start = time.perf_counter()
if mode.startswith("streaming"):
    from sahi.slicing import SliceImageStream
    stream = SliceImageStream(
        path, slice_height={slice}, slice_width={slice},
        overlap_height_ratio={overlap}, overlap_width_ratio={overlap},
        auto_slice_resolution=False,
    )
    for images, starts in stream.iter_batches({batch}, prefetch=mode.endswith("prefetch")):
        consume()
else:
    from sahi.slicing import slice_image
    result = slice_image(
        image=path, slice_height={slice}, slice_width={slice},
        overlap_height_ratio={overlap}, overlap_width_ratio={overlap},
        auto_slice_resolution=False,
    )
    # per batch, not per slice, so both paths pay the same simulated inference
    for _ in range(0, len(result.images), {batch}):
        consume()
print(time.perf_counter() - start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""

# Pure-Python iterations standing in for one batch of inference. 3M lands near 30 ms,
# which is the order a small detector takes on a batch of 8 slices.
CONSUME_ITERS = {0.0: 0, 30.0: 3_000_000}


def probe(path: str, mode: str, consume_ms: float) -> tuple[float, float] | None:
    """Run `mode` over `path` in a fresh process. Returns (seconds, peak MB), or None on OOM."""
    iters = CONSUME_ITERS.get(consume_ms, int(consume_ms * 100_000))
    source = _PROBE.format(slice=SLICE, overlap=OVERLAP, batch=BATCH_SIZE, consume_iters=iters)
    done = subprocess.run([sys.executable, "-c", source, path, mode], capture_output=True, text=True)
    if done.returncode != 0:
        if "MemoryError" in done.stderr or done.returncode == -9:  # -9 is the OOM killer
            return None
        raise RuntimeError(f"{mode} probe failed on {path}:\n{done.stderr}")
    seconds, max_rss = done.stdout.split()
    # ru_maxrss is KiB on Linux and bytes on macOS
    divisor = 1024 if sys.platform == "linux" else 1024**2
    return float(seconds), int(max_rss) / divisor


def write_test_image(path: Path, megapixels: int) -> None:
    """Write a square JPEG of the requested size without holding it in memory."""
    side = int(math.sqrt(megapixels * 1_000_000))
    try:
        import pyvips

        # gaussian noise, so the JPEG does not compress to nothing and the decoder does
        # real work; .copy(interpretation=...) tags it sRGB without a conversion pass
        pyvips.Image.gaussnoise(side, side, mean=128, sigma=40).bandjoin(
            [pyvips.Image.gaussnoise(side, side, mean=128, sigma=40) for _ in range(2)]
        ).cast("uchar").copy(interpretation="srgb").write_to_file(str(path), Q=90)
    except Exception:
        # not just ImportError: pyvips binds libvips through cffi at import and raises
        # OSError when the shared library is absent, which is the common case here
        import numpy as np
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8)).save(path, quality=90)


def fmt(result: tuple[float, float] | None) -> str:
    """Render one cell of the results table."""
    if result is None:
        return "OOM"
    seconds, peak_mb = result
    return f"{seconds:.2f} s / {peak_mb:.0f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="benchmark this file instead of the generated sweep")
    parser.add_argument("--mp", type=int, nargs="+", default=DEFAULT_MP, help="megapixel sizes to generate")
    parser.add_argument("--keep-images", action="store_true", help="keep generated images for reuse")
    parser.add_argument(
        "--consume-ms",
        type=float,
        default=0.0,
        help="simulated per-batch inference cost. Prefetch only pays off above roughly 30",
    )
    args = parser.parse_args()

    try:
        import pyvips

        print(f"libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)} — streaming active\n")
    except Exception as error:
        print(f"!! pyvips unusable ({error}); the streaming column measures the fallback, not libvips.")
        print("!! Install with: pip install sahi[bigimage]\n")

    if args.image:
        cases = [(Path(args.image).name, Path(args.image))]
        holder = None
    else:
        holder = Path("scripts/outputs/bench") if args.keep_images else Path(tempfile.mkdtemp())
        holder.mkdir(parents=True, exist_ok=True)
        cases = []
        for megapixels in args.mp:
            path = holder / f"noise_{megapixels}mp.jpg"
            if not path.exists():
                print(f"generating {megapixels} MP test image...", flush=True)
                write_test_image(path, megapixels)
            cases.append((f"{megapixels} MP", path))
        print()

    print(f"{SLICE}x{SLICE} slices at {OVERLAP} overlap, batch {BATCH_SIZE}, peak RSS of a fresh process")
    print(f"simulated inference: {args.consume_ms:.0f} ms/batch\n")
    print("| image | eager (slice_image) | streaming, no prefetch | streaming + prefetch |")
    print("| --- | --- | --- | --- |")
    for label, path in cases:
        eager = probe(str(path), "eager", args.consume_ms)
        serial = probe(str(path), "streaming", args.consume_ms)
        prefetched = probe(str(path), "streaming+prefetch", args.consume_ms)
        print(f"| {label} | {fmt(eager)} | {fmt(serial)} | {fmt(prefetched)} |", flush=True)

    print("\nStreaming costs a flat ~0.1 s for `import pyvips`, paid once per process. On")
    print("small images that fixed cost outweighs the saving; it is the large end that matters.")
    if args.consume_ms == 0:
        print("Prefetch has nothing to overlap with at 0 ms/batch. Try --consume-ms 30.")

    if holder is not None and not args.keep_images:
        for _, path in cases:
            path.unlink(missing_ok=True)
        holder.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

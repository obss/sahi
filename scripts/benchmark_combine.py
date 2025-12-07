"""Micro-benchmark for combine.py NMS/NMM functions.

Runs simple timings for nms, nmm and greedy_nmm (and batched counterparts) with random boxes
and prints elapsed time. Designed to be run locally by developers.
"""

import argparse
import random
import time

import torch

from sahi.postprocess.combine import (
    batched_greedy_nmm,
    batched_nmm,
    batched_nms,
    greedy_nmm,
    nmm,
    nms,
)


def random_boxes(n, classes=1):
    boxes = []
    for i in range(n):
        x1 = random.random() * 10000
        y1 = random.random() * 10000
        w = random.random() * 50
        h = random.random() * 50
        x2 = x1 + max(1.0, w)
        y2 = y1 + max(1.0, h)
        score = random.random()
        cid = random.randint(1, classes)
        boxes.append([x1, y1, x2, y2, score, cid])
    return torch.tensor(boxes, dtype=torch.float32)


def time_fn(fn, arg, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.time()
        fn(arg)
        times.append(time.time() - t0)
    return min(times), sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=[100, 500, 1000])
    parser.add_argument("--classes", type=int, default=5)
    args = parser.parse_args()

    print("Benchmarking NMS/NMM implementations")
    for n in args.sizes:
        print(f"\nInput size: {n}")
        data = random_boxes(n, classes=args.classes)

        # nms (class-agnostic)
        mn, avg = time_fn(nms, data)
        print(f"nms: min={mn:.4f}s avg={avg:.4f}s")

        # batched_nms (class-aware)
        mn, avg = time_fn(batched_nms, data)
        print(f"batched_nms: min={mn:.4f}s avg={avg:.4f}s")

        # nmm
        mn, avg = time_fn(nmm, data)
        print(f"nmm: min={mn:.4f}s avg={avg:.4f}s")

        # batched_nmm
        mn, avg = time_fn(batched_nmm, data)
        print(f"batched_nmm: min={mn:.4f}s avg={avg:.4f}s")

        # greedy_nmm
        mn, avg = time_fn(greedy_nmm, data)
        print(f"greedy_nmm: min={mn:.4f}s avg={avg:.4f}s")

        # batched_greedy_nmm
        mn, avg = time_fn(batched_greedy_nmm, data)
        print(f"batched_greedy_nmm: min={mn:.4f}s avg={avg:.4f}s")


if __name__ == "__main__":
    main()

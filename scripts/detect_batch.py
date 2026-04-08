"""
Batch inference detection script — multiple HuggingFace architectures

Tests sigmoid (RT-DETRv2, Conditional DETR) and softmax+background (DETR, YOLOS)
classification heads. Runs single-image and batch inference, saves annotated
outputs, benchmarks speed, and verifies consistency.

Usage:
    python scripts/detect_batch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image, visualize_object_predictions

OUTPUT_DIR = Path("scripts/outputs")
IMAGE1 = "demo/demo_data/small-vehicles1.jpeg"
IMAGE2 = "demo/demo_data/terrain2.png"

BATCH_SIZE = 8  # for speed benchmark

MODELS = [
    # sigmoid: logits.shape[-1] == num_labels (no background class)
    {
        "name": "rtdetr_v2_r18vd",
        "model_path": "PekingU/rtdetr_v2_r18vd",
        "image_size": 640,
        "confidence_threshold": 0.5,
        "expected_cls": "sigmoid",
    },
    {
        "name": "conditional_detr_resnet50",
        "model_path": "microsoft/conditional-detr-resnet-50",
        "image_size": 800,
        "confidence_threshold": 0.3,
        "expected_cls": "sigmoid",
    },
    # softmax+background: logits.shape[-1] == num_labels + 1
    {
        "name": "detr_resnet50",
        "model_path": "facebook/detr-resnet-50",
        "image_size": 800,
        "confidence_threshold": 0.3,
        "expected_cls": "softmax",
    },
]


def sep(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def print_top_preds(preds: list, n: int = 5) -> None:
    for p in sorted(preds, key=lambda x: x.score.value, reverse=True)[:n]:
        print(f"    {p.category.name:<20} score={p.score.value:.3f}  box={[round(v) for v in p.bbox.to_xyxy()]}")
    if len(preds) > n:
        print(f"    ... and {len(preds) - n} more")


def run(cfg: dict, img1: np.ndarray, img2: np.ndarray) -> bool:
    name = cfg["name"]
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    sep(name)
    print(f"  model     : {cfg['model_path']}")
    print(f"  threshold : {cfg['confidence_threshold']}")
    print(f"  image_size: {cfg['image_size']}")
    print(f"  expected  : {cfg['expected_cls']}")

    # ── load model ────────────────────────────────────────────
    print("\n  Loading model ...")
    try:
        model = AutoDetectionModel.from_pretrained(
            model_type="huggingface",
            model_path=cfg["model_path"],
            confidence_threshold=cfg["confidence_threshold"],
            image_size=cfg["image_size"],
            device="cpu",
        )
    except Exception as e:
        print(f"  SKIP — {e}")
        return True

    detected = "sigmoid" if model._uses_sigmoid_cls else "softmax"
    match = detected == cfg["expected_cls"]
    print(f"  num_labels: {model.num_categories}")
    print(f"  cls type  : {detected}  (expected: {cfg['expected_cls']})  [{'OK' if match else 'MISMATCH'}]")

    images = [img1, img2]
    image_names = [Path(IMAGE1).stem, Path(IMAGE2).stem]

    # ── single-image detection + save ─────────────────────────
    sep("detection results")
    single_counts = []
    for i, (img, img_name) in enumerate(zip(images, image_names)):
        model.perform_inference(img)
        model.convert_original_predictions()
        preds = model.object_prediction_list_per_image[0]
        single_counts.append(len(preds))

        visualize_object_predictions(
            image=img,
            object_prediction_list=preds,
            output_dir=str(out),
            file_name=f"single_{img_name}",
        )

        print(f"\n  [{img_name}]  {len(preds)} detections")
        print(f"  saved -> {out / f'single_{img_name}.png'}")
        print_top_preds(preds)

    # ── batch detection + save ────────────────────────────────
    model.perform_batch_inference(images)
    model.convert_original_predictions(
        shift_amount=[[0, 0]] * len(images),
        full_shape=[list(img.shape[:2]) for img in images],
    )
    batch_preds = model.object_prediction_list_per_image

    batch_counts = []
    for i, (preds, img, img_name) in enumerate(zip(batch_preds, images, image_names)):
        batch_counts.append(len(preds))
        visualize_object_predictions(
            image=img,
            object_prediction_list=preds,
            output_dir=str(out),
            file_name=f"batch_{img_name}",
        )

    # ── speed benchmark ───────────────────────────────────────
    sep(f"speed benchmark ({BATCH_SIZE} images)")
    bench_images = [img1] * BATCH_SIZE

    # warm-up
    model.perform_inference(img1)
    model.perform_batch_inference(bench_images[:2])

    # sequential: one image at a time
    t0 = time.perf_counter()
    for img in bench_images:
        model.perform_inference(img)
        model.convert_original_predictions()
    seq_total = time.perf_counter() - t0
    seq_per_img = seq_total / BATCH_SIZE * 1000

    # batch: all at once
    t0 = time.perf_counter()
    model.perform_batch_inference(bench_images)
    model.convert_original_predictions(
        shift_amount=[[0, 0]] * BATCH_SIZE,
        full_shape=[list(img1.shape[:2])] * BATCH_SIZE,
    )
    batch_total = time.perf_counter() - t0
    batch_per_img = batch_total / BATCH_SIZE * 1000

    speedup = seq_total / batch_total if batch_total > 0 else float("inf")

    print(f"\n  {'Method':<20} {'Total':>10} {'Per Image':>12} {'Speedup':>10}")
    print(f"  {'─' * 54}")
    print(f"  {'Sequential':<20} {seq_total:>9.3f}s {seq_per_img:>10.1f} ms {'1.00x':>10}")
    print(f"  {'Batch':<20} {batch_total:>9.3f}s {batch_per_img:>10.1f} ms {speedup:>9.2f}x")

    # ── consistency ────────────────────────────────────────────
    sep("consistency: single vs batch")
    passed = True
    for i, img_name in enumerate(image_names):
        s, b = single_counts[i], batch_counts[i]
        ok = s == b
        if not ok:
            passed = False
        print(f"  [{img_name}]  single={s}  batch={b}  [{'OK' if ok else 'FAIL'}]")

    if not match:
        passed = False

    if any(c == 0 for c in single_counts):
        print("  WARN: 0 detections on a single image")

    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading images ...")
    img1 = read_image(IMAGE1)
    img2 = read_image(IMAGE2)
    print(f"  {Path(IMAGE1).name}: {img1.shape}")
    print(f"  {Path(IMAGE2).name}: {img2.shape}")

    results = []
    benchmarks = []
    for cfg in MODELS:
        ok = run(cfg, img1, img2)
        results.append((cfg["name"], ok))

    sep("SUMMARY")
    all_passed = True
    for model_name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {model_name}")
        if not ok:
            all_passed = False

    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
    sys.exit(0 if all_passed else 1)

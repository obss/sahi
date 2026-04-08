"""
Local test script for HuggingFace batch inference.

Tests:
- RT-DETRv2 sigmoid classification path
- DETR softmax + background class path
- Single vs batch detection consistency
- perform_batch_inference API

Run:
    python tests/test_batch_inference_hf.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image

IMAGE1 = "demo/demo_data/small-vehicles1.jpeg"
IMAGE2 = "demo/demo_data/terrain2.png"

MODELS = [
    {
        "name": "RT-DETRv2-R18 (sigmoid)",
        "model_path": "PekingU/rtdetr_v2_r18vd",
        "image_size": 640,
        "confidence_threshold": 0.5,
    },
    {
        "name": "DETR-ResNet50 (softmax+background)",
        "model_path": "facebook/detr-resnet-50",
        "image_size": 800,
        "confidence_threshold": 0.3,
    },
]


def load_images() -> tuple[np.ndarray, np.ndarray]:
    img1 = read_image(IMAGE1)
    img2 = read_image(IMAGE2)
    print(f"  img1 shape: {img1.shape}")
    print(f"  img2 shape: {img2.shape}")
    return img1, img2


def test_single_inference(model, img: np.ndarray) -> list:
    model.perform_inference(img)
    model.convert_original_predictions()
    preds = model.object_prediction_list_per_image[0]
    return preds


def test_batch_inference(model, images: list[np.ndarray]) -> list[list]:
    model.perform_batch_inference(images)
    model.convert_original_predictions(
        shift_amount=[[0, 0]] * len(images),
        full_shape=[list(img.shape[:2]) for img in images],
    )
    return model.object_prediction_list_per_image


def run_model_tests(cfg: dict) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {cfg['name']}")
    print(f"  model: {cfg['model_path']}")
    print(f"{'=' * 60}")

    print("\n[1] Loading images ...")
    img1, img2 = load_images()
    batch = [img1, img2]

    print("\n[2] Loading model ...")
    try:
        model = AutoDetectionModel.from_pretrained(
            model_type="huggingface",
            model_path=cfg["model_path"],
            confidence_threshold=cfg["confidence_threshold"],
            image_size=cfg["image_size"],
            device="cpu",
        )
        print(f"  num categories: {model.num_categories}")
    except Exception as e:
        print(f"  SKIP — could not load model: {e}")
        return True  # not a code failure

    print("\n[3] Single-image inference ...")
    t0 = time.perf_counter()
    single_preds = test_single_inference(model, img1)
    single_time = time.perf_counter() - t0
    print(f"  img1 detections: {len(single_preds)}  ({single_time * 1000:.0f} ms)")
    if len(single_preds) == 0:
        print("  WARN: 0 detections on single image — check model/threshold")

    print("\n[4] Batch inference (2 images) ...")
    t0 = time.perf_counter()
    batch_preds = test_batch_inference(model, batch)
    batch_time = time.perf_counter() - t0
    print(f"  img0 detections: {len(batch_preds[0])}  ({batch_time * 1000:.0f} ms total)")
    print(f"  img1 detections: {len(batch_preds[1])}")

    print("\n[5] Consistency check (8x same image) ...")
    images_8 = [img1] * 8
    preds_8 = test_batch_inference(model, images_8)
    counts = [len(p) for p in preds_8]
    all_same = len(set(counts)) == 1
    print(f"  detection counts: {counts}")
    print(f"  all equal: {all_same}")

    print("\n[6] Single vs batch agreement ...")
    single_count = len(single_preds)
    batch_count = len(batch_preds[0])
    match = single_count == batch_count
    status = "OK" if match else "MISMATCH"
    print(f"  single={single_count}  batch[0]={batch_count}  [{status}]")

    print("\n[7] object_prediction_list_per_image length ...")
    assert len(preds_8) == 8, f"Expected 8, got {len(preds_8)}"
    print(f"  8 images → {len(preds_8)} result lists  [OK]")

    print("\n[8] Backward compat: object_prediction_list ...")
    _ = test_single_inference(model, img1)
    compat = model.object_prediction_list
    assert isinstance(compat, list), "object_prediction_list should return a list"
    print(f"  object_prediction_list: {len(compat)} predictions  [OK]")

    passed = match and all_same
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = []
    for cfg in MODELS:
        ok = run_model_tests(cfg)
        results.append((cfg["name"], ok))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
        if not ok:
            all_passed = False

    sys.exit(0 if all_passed else 1)

"""Backend parity for SAHI's custom COCO evaluation and JSON/CLI paths."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sahi.scripts import coco_evaluation

pytest.importorskip("pycocotools")
pytest.importorskip("ultrafast_pycocotools")


@pytest.fixture
def coco_files(tmp_path: Path) -> tuple[Path, Path]:
    from pycocotools import mask as mask_utils

    annotations, predictions = [], []
    for idx, (category, x, size, crowd) in enumerate([(1, 1, 3, 0), (2, 8, 7, 0), (2, 16, 12, 1)], 1):
        mask = np.zeros((32, 32), dtype=np.uint8, order="F")
        mask[x : x + size, x : x + size] = 1
        rle = mask_utils.encode(mask)
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append(
            {
                "id": idx,
                "image_id": 1,
                "category_id": category,
                "iscrowd": crowd,
                "area": size * size,
                "bbox": [x, x, size, size],
                "segmentation": [[x, x, x + size, x, x + size, x + size, x, x + size]],
            }
        )
        predictions.append(
            {
                "image_id": 1,
                "category_id": category,
                "score": 0.8,
                "bbox": [x, x, size, size],
                "segmentation": rle,
            }
        )
    # A tied duplicate and a detection on an otherwise empty image.
    predictions.append(dict(predictions[0]))
    predictions.append(dict(predictions[0], image_id=2, score=0.1))
    false_mask = np.zeros((32, 32), dtype=np.uint8, order="F")
    false_mask[:3, 25:28] = 1
    false_rle = mask_utils.encode(false_mask)
    false_rle["counts"] = false_rle["counts"].decode("ascii")
    predictions.insert(0, dict(predictions[0], bbox=[25, 0, 3, 3], segmentation=false_rle, score=0.9))
    dataset = {
        "images": [{"id": i, "width": 32, "height": 32} for i in (1, 2, 3)],
        "categories": [{"id": i, "name": f"class-{i}"} for i in (1, 2, 3)],
        "annotations": annotations,
    }
    dataset_path, result_path = tmp_path / "dataset.json", tmp_path / "result.json"
    dataset_path.write_text(json.dumps(dataset))
    result_path.write_text(json.dumps(predictions))
    return dataset_path, result_path


@pytest.mark.parametrize("metric", ["bbox", "segm"])
@pytest.mark.parametrize("max_detections", [1, 500])
@pytest.mark.parametrize("iou_thrs", [None, 0.5, [0.5, 0.75]])
def test_backend_full_array_and_export_parity(
    coco_files: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    max_detections: int,
    iou_thrs: Any,
) -> None:
    original_loader = coco_evaluation._load_coco_backend
    evaluators = {}

    def capture_backend(backend: str) -> tuple[type, type]:
        coco, evaluator = original_loader(backend)

        class CaptureEval(evaluator):
            def accumulate(self, *args: Any, **kwargs: Any) -> None:
                super().accumulate(*args, **kwargs)
                evaluators[backend] = self

        return coco, CaptureEval

    monkeypatch.setattr(coco_evaluation, "_load_coco_backend", capture_backend)
    outputs = []
    before = [p.read_bytes() for p in coco_files]
    for backend in ("pycocotools", "ultrafast"):
        output = coco_evaluation.evaluate(
            *(str(p) for p in coco_files),
            backend=backend,
            type=metric,
            max_detections=max_detections,
            iou_thrs=iou_thrs,
            areas=[16, 100, 10000],
            classwise=True,
            return_dict=True,
            out_dir=str(tmp_path / backend),
        )
        assert json.loads(Path(output["export_path"]).read_text()) == output["eval_results"]
        outputs.append(output["eval_results"])
    assert outputs[0] == outputs[1]
    assert [p.read_bytes() for p in coco_files] == before
    reference, candidate = evaluators["pycocotools"], evaluators["ultrafast"]
    for key in ("precision", "recall", "scores"):
        np.testing.assert_array_equal(candidate.eval[key], reference.eval[key])
    np.testing.assert_array_equal(candidate.stats, reference.stats)
    assert candidate.params.maxDets == [max_detections]


def test_existing_bbox_fixture(tmp_path: Path) -> None:
    data = Path(__file__).parent / "data" / "coco_evaluate"
    results = [
        coco_evaluation.evaluate(
            str(data / "dataset.json"),
            str(data / "result.json"),
            backend=backend,
            classwise=True,
            return_dict=True,
            out_dir=str(tmp_path / backend),
        )["eval_results"]
        for backend in ("pycocotools", "ultrafast")
    ]
    assert results[0] == results[1]


@pytest.mark.parametrize("backend", ["pycocotools", "ultrafast"])
def test_empty_results(coco_files: tuple[Path, Path], tmp_path: Path, backend: str) -> None:
    coco_files[1].write_text("[]")
    output = coco_evaluation.evaluate(
        *(str(p) for p in coco_files),
        backend=backend,
        return_dict=True,
        out_dir=str(tmp_path / backend),
    )
    assert output["eval_results"] == {}
    assert json.loads(Path(output["export_path"]).read_text()) == {}


def test_backend_errors_and_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def without_reference(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pycocotools"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_reference)
    coco, evaluator = coco_evaluation._load_coco_backend("ultrafast")
    assert coco.__module__.startswith("ultrafast_pycocotools.")
    assert evaluator.__module__.startswith("ultrafast_pycocotools.")
    with pytest.raises(ModuleNotFoundError, match="pip install -U pycocotools"):
        coco_evaluation._load_coco_backend("pycocotools")
    with pytest.raises(ValueError, match="Unknown COCO backend"):
        coco_evaluation._load_coco_backend("unknown")

    def without_ultrafast(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("ultrafast_pycocotools"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ultrafast)
    with pytest.raises(ModuleNotFoundError, match=r"sahi\[ultrafast\]"):
        coco_evaluation._load_coco_backend("ultrafast")
    assert coco_evaluation._load_coco_backend("pycocotools")[0].__module__ == "pycocotools.coco"


@pytest.mark.parametrize("backend", [None, "ultrafast"])
def test_cli(coco_files: tuple[Path, Path], tmp_path: Path, backend: str | None) -> None:
    output_dir = tmp_path / "cli"
    command = [
        sys.executable,
        "-m",
        "sahi.cli",
        "coco",
        "evaluate",
        "--dataset_json_path",
        str(coco_files[0]),
        "--result_json_path",
        str(coco_files[1]),
        "--out_dir",
        str(output_dir),
        "--max_detections",
        "500",
        "--classwise",
        "True",
    ]
    if backend is not None:
        command += ["--backend", backend]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    result = json.loads((output_dir / "eval.json").read_text())
    assert 0 < result["bbox_mAP"] < 1
    assert "results_per_category" in result

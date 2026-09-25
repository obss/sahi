from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sahi.scripts.coco_evaluation import evaluate

pytest.importorskip("pycocotools")
pytest.importorskip("ultrafast_pycocotools")

DATA = Path(__file__).parent / "data" / "coco_evaluate"


def run(tmp_path: Path, result: Path, backend: str, **kwargs: Any) -> dict:
    return evaluate(
        str(DATA / "dataset.json"),
        str(result),
        out_dir=str(tmp_path / backend),
        backend=backend,
        return_dict=True,
        **kwargs,
    )["eval_results"]


@pytest.mark.parametrize("max_detections", [1, 500])
@pytest.mark.parametrize("iou_thrs", [None, 0.5, [0.5, 0.75]])
def test_backends_match(tmp_path: Path, max_detections: int, iou_thrs: float | list[float] | None) -> None:
    kwargs = dict(max_detections=max_detections, iou_thrs=iou_thrs, classwise=True)
    assert run(tmp_path, DATA / "result.json", "pycocotools", **kwargs) == run(
        tmp_path, DATA / "result.json", "ultrafast", **kwargs
    )


@pytest.mark.parametrize("backend", ["pycocotools", "ultrafast"])
def test_empty_results(tmp_path: Path, backend: str) -> None:
    result = tmp_path / "result.json"
    result.write_text("[]")
    assert run(tmp_path, result, backend) == {}

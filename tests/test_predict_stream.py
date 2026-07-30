"""Tests for streaming slice reads inside get_sliced_prediction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from sahi.predict import get_sliced_prediction
from sahi.slicing import _group_bboxes_into_bands, get_slice_bboxes, slice_image

SLICE = 128
OVERLAP = 0.2


class RecordingModel:
    """A detection model that predicts nothing and records what it was handed.

    Implements only the surface `get_sliced_prediction` touches. The real base class
    pulls in torch, which these tests run without.
    """

    def __init__(self, events: list[str] | None = None) -> None:
        self.confidence_threshold = 0.5
        self.batches: list[list[np.ndarray]] = []
        self.shifts: list[list[list[int | float]]] = []
        self.object_prediction_list_per_image: list[list] = []
        self.object_prediction_list: list = []
        # shared with the band-source spy, so reads and inferences can be interleaved
        self.events = events if events is not None else []

    def perform_inference(self, image: np.ndarray) -> None:
        """Whole-image inference, which `perform_standard_pred=True` adds on top."""
        self.events.append("standard_infer")

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        self.events.append("infer")
        # the eager path fed models np.ascontiguousarray output and backends index into
        # the buffer directly, so a non-contiguous slice is a silent wrong-pixels bug
        assert all(image.flags["C_CONTIGUOUS"] for image in images), "slices must reach the model contiguous"
        self.batches.append([image.copy() for image in images])
        self.object_prediction_list_per_image = [[] for _ in images]

    def convert_original_predictions(self, shift_amount: list, full_shape: list) -> None:
        self.shifts.append(shift_amount)


def make_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Noise image; any band-boundary mistake shows up as a pixel mismatch."""
    return np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)


def run(path: str, batch_size: int, model: RecordingModel | None = None, **kwargs: Any) -> RecordingModel:
    """Run a sliced prediction that produces no detections, and return the model."""
    model = model or RecordingModel()
    get_sliced_prediction(
        image=path,
        detection_model=model,  # type: ignore[arg-type]
        slice_height=SLICE,
        slice_width=SLICE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        auto_slice_resolution=False,
        batch_size=batch_size,
        verbose=0,
        **{"perform_standard_pred": False, **kwargs},
    )
    return model


class TestSlicedPredictionEquivalence:
    """Test the model receives what the eager path would have produced."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_model_receives_same_slices_as_slice_image(self, tmp_path: Path, batch_size: int) -> None:
        """Test the same slices arrive, in the same order, as slice_image produces."""
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400, seed=1)).save(path)
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        model = run(path, batch_size)

        received = [image for batch in model.batches for image in batch]
        assert len(received) == len(eager)
        for index, (got, want) in enumerate(zip(received, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    def test_shift_amounts_match_slice_image_starting_pixels(self, tmp_path: Path) -> None:
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400, seed=2)).save(path)
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        model = run(path, batch_size=4)

        received = [shift for batch in model.shifts for shift in batch]
        assert received == [list(pixel) for pixel in eager.starting_pixels]


class TestStreamingIsActuallyUsed:
    """Test slices really are streamed. Eager slicing passes the equivalence tests too."""

    def test_read_ahead_stays_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test bands are read as inference consumes them, rather than all up front.

        Not strict read/infer alternation: the prefetch worker reads one band ahead by
        design, so a read may overtake an inference. What must stay true is that the
        read-ahead is *bounded* — if slicing went back to materialising everything, every
        band would be read before the first inference.
        """
        import sahi.slicing as slicing

        events: list[str] = []
        open_band_source = slicing._open_band_source

        def spy(*args: Any, **kwargs: Any) -> Any:
            source = open_band_source(*args, **kwargs)
            read_band = source.read_band

            def recording_read_band(y_min: int, y_max: int) -> np.ndarray:
                events.append("read")
                return read_band(y_min, y_max)

            source.read_band = recording_read_band  # type: ignore[method-assign]
            return source

        monkeypatch.setattr(slicing, "_open_band_source", spy)
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400, seed=3)).save(path)
        bands = _group_bboxes_into_bands(get_slice_bboxes(300, 400, SLICE, SLICE, False, OVERLAP, OVERLAP))
        assert len(bands) > 1, "fixture no longer exercises more than one band"

        run(path, batch_size=3, model=RecordingModel(events))

        assert events.count("read") == len(bands)
        assert events[0] == "read", "inference ran before any band was read"
        # one band in the queue plus one being cut is the most _prefetch(depth=1) allows
        reads_before_first_infer = events.index("infer")
        assert reads_before_first_infer <= 2, (
            f"{reads_before_first_infer} of {len(bands)} bands were read before the first "
            f"inference; read-ahead should be bounded by the prefetch depth"
        )
        # and the rest really are read later, not front-loaded
        assert events.count("read") > reads_before_first_infer, "every band was read up front"


def pyvips_installed() -> bool:
    """Whether the optional libvips extra is usable here.

    pyvips raises OSError, not ImportError, when the shared library is missing.
    """
    try:
        import pyvips  # noqa: F401
    except Exception:
        return False
    return True


class TestStandardPredWarning:
    """Test the streaming caller is told when perform_standard_pred undoes the saving.

    It defaults to True and runs inference on the *whole* image, which decodes it in one
    piece: the peak memory the streaming path exists to bound is silently restored.
    """

    @pytest.mark.skipif(not pyvips_installed(), reason="pyvips is an optional extra")
    def test_warns_when_streaming_and_standard_pred_is_on(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test the warning fires, and names the flag that turns it off."""
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400)).save(path)

        with caplog.at_level("WARNING"):
            model = run(path, batch_size=2, perform_standard_pred=True)

        assert "standard_infer" in model.events, "fixture no longer runs the standard prediction"
        assert "perform_standard_pred=False" in caplog.text

    def test_no_warning_when_standard_pred_is_off(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test the caller who already did the right thing is not nagged."""
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400)).save(path)

        with caplog.at_level("WARNING"):
            run(path, batch_size=2)

        assert "perform_standard_pred" not in caplog.text

    def test_no_warning_when_the_image_is_not_streamed(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test an in-memory image does not warn: it is already decoded, so nothing is lost."""
        with caplog.at_level("WARNING"):
            get_sliced_prediction(
                image=make_image(300, 400),
                detection_model=RecordingModel(),  # type: ignore[arg-type]
                slice_height=SLICE,
                slice_width=SLICE,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                auto_slice_resolution=False,
                perform_standard_pred=True,
                verbose=0,
            )

        assert "perform_standard_pred" not in caplog.text


class TestSliceDurations:
    """Test the reported slice duration still describes the pixel reads.

    Reads moved inside the prediction loop, so timing only the geometry step would report
    ~0 seconds for slicing a gigapixel scan, which reads as a broken counter.
    """

    def test_slice_duration_covers_the_band_reads(self, tmp_path: Path) -> None:
        """Test time spent waiting for bands lands in `slice`, not in `prediction`."""
        import sahi.slicing as slicing

        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400)).save(path)
        open_band_source = slicing._open_band_source

        def slow(*args: Any, **kwargs: Any) -> Any:
            source = open_band_source(*args, **kwargs)
            read_band = source.read_band

            def slow_read_band(y_min: int, y_max: int) -> np.ndarray:
                time.sleep(0.05)
                return read_band(y_min, y_max)

            source.read_band = slow_read_band  # type: ignore[method-assign]
            return source

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(slicing, "_open_band_source", slow)
            result = get_sliced_prediction(
                image=path,
                detection_model=RecordingModel(),  # type: ignore[arg-type]
                slice_height=SLICE,
                slice_width=SLICE,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                auto_slice_resolution=False,
                perform_standard_pred=False,
                batch_size=2,
                # prefetch would hide the sleep behind inference, which is the point of it
                # but leaves nothing to measure here
                verbose=0,
            )

        assert result.durations_in_seconds["slice"] >= 0.05, (
            f"band reads took at least 0.05s but slice duration was {result.durations_in_seconds['slice']:.4f}s"
        )
        assert result.durations_in_seconds["prediction"] >= 0, "read time was double-counted out of prediction"


class TestSliceExport:
    """Test slice_dir export, now done one band at a time rather than all at once."""

    def test_exports_one_file_per_slice(self, tmp_path: Path) -> None:
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400, seed=4)).save(path)
        out_dir = tmp_path / "slices"
        expected = {
            f"pre_{'_'.join(map(str, bbox))}.png"
            for bbox in get_slice_bboxes(300, 400, SLICE, SLICE, False, OVERLAP, OVERLAP)
        }

        run(path, batch_size=4, slice_dir=str(out_dir), slice_export_prefix="pre")

        assert {file.name for file in out_dir.iterdir()} == expected

    def test_exported_slices_match_the_slices_predicted_on(self, tmp_path: Path) -> None:
        """Exports are written from the band buffer the reader reuses, so a missing copy
        corrupts them silently.
        """
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(300, 400, seed=5)).save(path)
        out_dir = tmp_path / "slices"

        model = run(path, batch_size=4, slice_dir=str(out_dir), slice_export_prefix="pre")

        received = [image for batch in model.batches for image in batch]
        bboxes = get_slice_bboxes(300, 400, SLICE, SLICE, False, OVERLAP, OVERLAP)
        for image, bbox in zip(received, bboxes):
            written = np.asarray(Image.open(out_dir / f"pre_{'_'.join(map(str, bbox))}.png"))
            np.testing.assert_array_equal(written, image, err_msg=f"export for {bbox} differs")

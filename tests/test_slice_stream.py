"""Tests for streaming slice reads (SliceImageStream)."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sahi.slicing import (
    SliceImageStream,
    _ChunkedBandReader,
    _group_bboxes_into_bands,
    _InMemoryBandSource,
    _open_band_source,
    _prefetch,
    _VipsBandSource,
    get_slice_bboxes,
    slice_image,
)


class RecordingForwardSource:
    """Sequential decoder stand-in: serves rows forward-only and raises on a rewind."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.reads: list[tuple[int, int]] = []
        self.position = 0

    def read_chunk(self, y_min: int, rows: int) -> np.ndarray:
        if y_min < self.position:
            raise AssertionError(f"backward read: asked for row {y_min}, decoder is at {self.position}")
        self.reads.append((y_min, y_min + rows))
        self.position = y_min + rows
        return self.array[y_min : y_min + rows]


SLICE = 128
OVERLAP = 0.2


def make_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Noise image; any band-boundary mistake shows up as a pixel mismatch."""
    return np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)


def stream_all(stream: SliceImageStream, batch_size: int = 4) -> list[np.ndarray]:
    """Flatten every batch into a single ordered list of slices."""
    return [image for batch_images, _ in stream.iter_batches(batch_size) for image in batch_images]


class TestSliceImageStreamGeometry:
    """Test slice geometry, which is computed without decoding pixels."""

    def test_reports_same_geometry_as_get_slice_bboxes(self) -> None:
        """Test the stream reports the same bboxes as get_slice_bboxes."""
        image = np.zeros((300, 400, 3), dtype=np.uint8)

        stream = SliceImageStream(
            image,
            slice_height=128,
            slice_width=128,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            auto_slice_resolution=False,
        )

        expected = get_slice_bboxes(300, 400, 128, 128, False, 0.2, 0.2)
        assert stream.original_image_height == 300
        assert stream.original_image_width == 400
        assert stream.slice_bboxes == expected
        assert len(stream) == len(expected)

    @pytest.mark.parametrize("axis", ["overlap_height_ratio", "overlap_width_ratio"])
    def test_negative_overlap_is_rejected(self, axis: str) -> None:
        """Test a negative ratio is rejected, since it spaces slices apart.

        Gapped slices leave rows no band covers, which a forward-only reader cannot
        serve; catching it at the geometry step keeps the error at the caller's input.
        """
        ratios = {"overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2, axis: -0.5}

        with pytest.raises(ValueError, match=r"Overlap ratio must be in \[0.0, 1.0\)"):
            get_slice_bboxes(300, 400, 128, 128, False, **ratios)

    @pytest.mark.parametrize("size", [300, 640, 1024, 2048, 6380])
    def test_auto_slice_resolution_survives_the_ratio_bound(self, size: int) -> None:
        """The auto path still produces covering slices at every resolution tier.

        `calc_slice_and_overlap_params` uses an overlap *ratio* of 1.0 for low-resolution
        images, which the bound above would reject. It never reaches that bound: the auto
        branch converts to pixel overlaps itself, and the 1.0 tier is also the one that
        splits into a single whole-image slice. Both facts are pinned here, since routing
        auto through the ratio check would raise, and an overlap equal to the slice size
        with more than one slice would hang.
        """
        bboxes = get_slice_bboxes(size, size, auto_slice_resolution=True)

        assert bboxes, "auto slicing produced no slices"
        assert bboxes[0][:2] == [0, 0]
        assert max(x_max for _, _, x_max, _ in bboxes) == size
        assert max(y_max for _, _, _, y_max in bboxes) == size


class TestSliceImageStreamInputValidation:
    """Test bad inputs are refused at construction, where slice_image refuses them."""

    @pytest.mark.parametrize("shape", [(0, 400, 3), (300, 0, 3), (0, 0, 3)])
    def test_zero_size_image_is_rejected(self, shape: tuple[int, int, int]) -> None:
        """Test an empty image raises rather than yielding no slices.

        `slice_image` raises on the same input. Yielding nothing would surface through
        `get_sliced_prediction` as an empty result, which reads as "nothing detected"
        rather than "nothing was read".
        """
        with pytest.raises(RuntimeError, match="invalid image size"):
            SliceImageStream(np.zeros(shape, dtype=np.uint8), auto_slice_resolution=True)

    def test_batch_size_is_validated_on_the_call(self) -> None:
        """Test an invalid batch_size raises from iter_batches, not from the first next().

        `iter_batches` returns its generator rather than delegating to it precisely so
        this check is not deferred to a `for` statement somewhere else in the caller.
        """
        stream = SliceImageStream(make_image(300, 400), slice_height=SLICE, slice_width=SLICE)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            stream.iter_batches(0)


class TestPathInput:
    """Test os.PathLike inputs, which used to size fine and then fail to decode."""

    def test_path_and_str_produce_the_same_slices(self, tmp_path: Path) -> None:
        """Test a Path is accepted end to end and matches the str spelling."""
        array = make_image(300, 400, seed=7)
        path = tmp_path / "scan.png"
        Image.fromarray(array).save(str(path))
        kwargs = {
            "slice_height": SLICE,
            "slice_width": SLICE,
            "overlap_height_ratio": OVERLAP,
            "overlap_width_ratio": OVERLAP,
            "auto_slice_resolution": False,
        }

        from_path = stream_all(SliceImageStream(path, **kwargs))  # type: ignore[arg-type]
        from_str = stream_all(SliceImageStream(str(path), **kwargs))

        assert len(from_path) == len(from_str)
        for index, (got, want) in enumerate(zip(from_path, from_str)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    def test_path_still_takes_the_streaming_route(self, tmp_path: Path) -> None:
        """Test a Path is normalised early enough to stream.

        Every path check downstream tests for `str`, so a Path that reached them unchanged
        would silently fall back to a whole-image decode.
        """
        path = tmp_path / "scan.png"
        Image.fromarray(make_image(300, 400)).save(str(path))

        streams_as_str = SliceImageStream(str(path), auto_slice_resolution=True).is_streaming

        assert SliceImageStream(path, auto_slice_resolution=True).is_streaming == streams_as_str  # type: ignore[arg-type]


class TestInMemoryBandSourceShapeGuard:
    """Test the decoded array is checked against the geometry it will be cut with.

    `_VipsBandSource` cross-checks its dimensions at open; without the same check here a
    header/decode disagreement produces silently wrong slices instead of an error, since
    numpy clamps an out-of-range slice rather than raising.
    """

    @pytest.mark.parametrize("expected", [(301, 400), (300, 401), (299, 399)])
    def test_shape_disagreement_is_rejected(self, expected: tuple[int, int]) -> None:
        """Test a mismatch on either axis raises at construction."""
        with pytest.raises(ValueError, match="where the header read saw"):
            _InMemoryBandSource(make_image(300, 400), expected_shape=expected)

    def test_matching_shape_is_accepted(self) -> None:
        """Test the guard does not reject the images it is supposed to pass."""
        array = make_image(300, 400)

        source = _InMemoryBandSource(array, expected_shape=(300, 400))

        np.testing.assert_array_equal(source.read_band(0, 128), array[0:128])


class TestSliceImageStreamEquivalence:
    """Test streamed slices against slice_image."""

    def test_streamed_slices_match_slice_image_pixels(self) -> None:
        """Test streamed slices are pixel-identical to slice_image output."""
        image = make_image(300, 400)
        stream = SliceImageStream(
            image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        streamed = stream_all(stream)

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    def test_batches_stay_full_across_band_boundaries(self) -> None:
        """Test a band boundary does not cut a batch short.

        Only two slices fit across this image, so a batch that stopped at the end of a
        row-band would never reach the requested size.
        """
        image = make_image(600, 200)
        stream = SliceImageStream(
            image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        sizes = [len(images) for images, _ in stream.iter_batches(4)]

        assert sum(sizes) == len(stream)
        assert sizes[:-1] == [4] * (len(sizes) - 1), f"short batch before the last: {sizes}"

    def test_streamed_shifts_match_slice_image(self) -> None:
        """Test starting pixels match slice_image."""
        image = make_image(300, 400)
        stream = SliceImageStream(
            image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        shifts = [shift for _, batch_shifts in stream.iter_batches(4) for shift in batch_shifts]

        assert shifts == eager.starting_pixels


class TestExifOrientation:
    """Test EXIF orientation handling.

    The contract is that an oriented file never streams. `read_image_as_pil` applies every
    EXIF transform and a sequential decoder applies none, so a streamed oriented image
    would carry the right geometry and the wrong pixels. The `_VipsBandSource` assertions
    below hold that line, since once a case falls back its pixel comparison is eager
    against eager and cannot fail. Sizes are covered by TestReadImageSize.
    """

    @pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif"])
    @pytest.mark.parametrize("orientation", [2, 6])
    @pytest.mark.parametrize("exif_fix", [True, False])
    def test_oriented_image_falls_back_and_matches_slice_image(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, orientation: int, suffix: str, exif_fix: bool
    ) -> None:
        """An oriented file falls back to an eager decode, whatever exif_fix says.

        6 is a quarter turn, which transposes the geometry; 2 is a mirror, which keeps the
        shape and so returns plausible-looking slices when it is skipped. `exif_fix=False`
        is covered because PIL's TIFF plugin applies orientations 2-4 on load regardless
        of the flag, so honouring the flag in the streaming source made the two paths
        disagree on every slice of a mirrored TIFF.
        """
        path = str(tmp_path / f"rotated{suffix}")
        exif = Image.Exif()
        exif[0x0112] = orientation
        Image.fromarray(make_image(200, 300, seed=11)).save(path, exif=exif, quality=95)

        stream = SliceImageStream(
            path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
            exif_fix=exif_fix,
        )
        with caplog.at_level("WARNING"):
            source = _open_band_source(
                path,
                band_height=SLICE,
                exif_fix=exif_fix,
                expected_shape=(stream.original_image_height, stream.original_image_width),
            )
        assert not isinstance(source, _VipsBandSource), "an oriented image must not be streamed"
        assert "decoding the whole image instead" in caplog.text

        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
            exif_fix=exif_fix,
        )
        streamed = stream_all(stream)

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            assert got.shape == want.shape, f"slice {index} shape {got.shape} != {want.shape}"
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    def test_untagged_image_still_streams(self, tmp_path: Path) -> None:
        """Test the orientation guard only refuses files that carry a tag."""
        path = str(tmp_path / "plain.tif")
        Image.fromarray(make_image(200, 300, seed=13)).save(path)

        if pyvips_installed():
            assert isinstance(_open_band_source(path, band_height=SLICE), _VipsBandSource)


class TestPillowInput:
    """Test slicing an in-memory Pillow image."""

    def test_streamed_slices_from_pil_image_match_slice_image(self) -> None:
        """Test a Pillow image falls back to an eager decode and still matches slice_image."""
        pil_image = Image.fromarray(make_image(300, 400, seed=9))

        stream = SliceImageStream(
            pil_image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=pil_image,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        streamed = stream_all(stream)

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")


class TestPrefetch:
    """Test the worker thread that reads the next band while the caller works.

    Correctness here is not about pixels, it is about what a thread can get wrong:
    losing an exception, reordering results, or outliving the iteration.
    """

    def test_yields_everything_in_order(self) -> None:
        """Test the results and their order are exactly the source's."""
        source = ((["item"], [[index, 0]]) for index in range(50))

        assert [shift[0][0] for _, shift in _prefetch(source)] == list(range(50))

    def test_exception_reaches_the_caller(self) -> None:
        """Test a failure inside the worker is re-raised in the consuming thread.

        Swallowed here it would look like a short iteration: some slices silently never
        inferred on, and no error anywhere.
        """

        def explode() -> Iterator[tuple[list, list]]:
            yield ([], [[0, 0]])
            raise RuntimeError("band read failed")

        received = []
        with pytest.raises(RuntimeError, match="band read failed"):
            for batch in _prefetch(explode()):
                received.append(batch)

        assert len(received) == 1, "batches before the failure should still arrive"

    def test_abandoning_the_iteration_stops_the_worker(self) -> None:
        """Test breaking out of the loop does not leave the thread parked on a full queue.

        The worker blocks on `put` once the queue is full, so without the drain in
        `_prefetch`'s finally it would sit there for the life of the process.
        """
        import threading

        before = threading.active_count()
        for _ in _prefetch(([f"item{i}"], [[i, 0]]) for i in range(1000)):
            break  # abandon immediately, while the worker is still producing

        deadline = time.time() + 5
        while threading.active_count() > before and time.time() < deadline:
            time.sleep(0.01)
        assert threading.active_count() == before, "prefetch worker outlived the iteration"

    def test_source_is_closed_when_abandoned(self) -> None:
        """Test the underlying generator is closed, so its `finally` blocks run."""
        closed = []

        def source() -> Iterator[tuple[list, list]]:
            try:
                for index in range(1000):
                    yield ([], [[index, 0]])
            finally:
                closed.append(True)

        for _ in _prefetch(source()):
            break

        deadline = time.time() + 5
        while not closed and time.time() < deadline:
            time.sleep(0.01)
        assert closed, "source generator was never closed"

    def test_abandoning_a_slow_producer_stops_the_worker(self) -> None:
        """Test abandonment works while the worker is mid-read, not parked on `put`.

        The tests above abandon an instantaneous source, so the worker is always already
        blocked in `put`. A real band decode spends its time inside `next()`, and closing
        the generator from the consumer there raises "generator already executing",
        which skips the drain and strands the worker with a whole batch pinned.
        """
        import threading

        closed = []

        def slow_source() -> Iterator[tuple[list, list]]:
            try:
                for index in range(1000):
                    time.sleep(0.02)  # stand in for a band decode
                    yield ([], [[index, 0]])
            finally:
                closed.append(True)

        before = threading.active_count()
        for _ in _prefetch(slow_source()):
            break  # abandon while the worker is inside the source's sleep

        assert closed, "source generator was never closed"
        assert threading.active_count() == before, "prefetch worker outlived the iteration"


class TestChunkedBandReader:
    """Test band assembly from a forward-only source."""

    @pytest.mark.parametrize(
        "height,width",
        [(128, 128), (129, 127), (200, 300), (256, 256), (300, 151), (300, 400), (65, 400), (400, 65)],
    )
    def test_assembles_overlapping_bands_with_forward_only_reads(self, height: int, width: int) -> None:
        """Test overlapping bands are assembled without re-reading a row."""
        array = make_image(height, width, seed=2)
        bands = _group_bboxes_into_bands(get_slice_bboxes(height, width, SLICE, SLICE, False, OVERLAP, OVERLAP))
        band_height = max(y_max - y_min for (y_min, y_max), _ in bands)
        source = RecordingForwardSource(array)
        reader = _ChunkedBandReader(source.read_chunk, image_height=height, band_height=band_height)

        for (y_min, y_max), _ in bands:
            np.testing.assert_array_equal(
                reader.read_band(y_min, y_max),
                array[y_min:y_max],
                err_msg=f"band {y_min}:{y_max} differs",
            )

        assert source.reads, "expected the reader to pull chunks from the source"
        starts = [start for start, _ in source.reads]
        assert starts == sorted(starts), "chunk reads must advance monotonically"

    def test_skipping_rows_raises_instead_of_returning_the_wrong_ones(self) -> None:
        """Test a gap between bands is rejected rather than silently misread.

        A forward-only reader cannot serve a band starting past what it has read; the
        old code let the buffer label drift and returned plausible but wrong pixels.
        """
        array = make_image(600, 128, seed=3)
        source = RecordingForwardSource(array)
        reader = _ChunkedBandReader(source.read_chunk, image_height=600, band_height=128)
        reader.read_band(0, 128)

        with pytest.raises(ValueError, match="past buffered row"):
            reader.read_band(400, 528)


class TestCHWInput:
    """Test CHW arrays are transposed the way read_image_as_pil transposes them."""

    def test_chw_array_matches_slice_image(self) -> None:
        """Test a (3, H, W) array is sliced along the same axes as slice_image."""
        hwc = make_image(200, 300, seed=13)
        chw = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))

        stream = SliceImageStream(
            chw,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=chw,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        assert (stream.original_image_height, stream.original_image_width) == (200, 300)
        streamed = stream_all(stream)

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            assert got.shape == want.shape, f"slice {index} shape {got.shape} != {want.shape}"
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")


def pyvips_installed() -> bool:
    """Whether the optional streaming backend is usable.

    Importable is not enough: pyvips binds libvips through cffi at import time and
    raises OSError, not ImportError, when the shared library is missing.
    `_open_band_source` treats both the same way, so this must too.
    """
    try:
        import pyvips  # noqa: F401
    except Exception:
        return False
    return True


class TestPyvipsAbsentFallback:
    """Test the fallback taken when pyvips is unavailable."""

    def test_falls_back_to_eager_decode_and_matches_slice_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test an unimportable pyvips falls back to an eager decode."""
        import sys

        monkeypatch.setitem(sys.modules, "pyvips", None)  # makes `import pyvips` raise
        path = str(tmp_path / "scan.png")
        image = make_image(200, 300, seed=14)
        Image.fromarray(image).save(path)

        source = _open_band_source(path, band_height=SLICE)
        stream = SliceImageStream(
            path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        streamed = stream_all(stream)
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        assert not isinstance(source, _VipsBandSource)
        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    def test_pyvips_without_libvips_is_not_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test a pyvips that cannot bind libvips falls back quietly.

        pyvips binds libvips through cffi at import and raises OSError, not ImportError,
        when the shared library is missing. That is the same "extra not usable" case as
        pyvips being absent, which is the documented default, so it must not warn.
        """
        import builtins

        real_import = builtins.__import__

        def raise_oserror(name: str, *args: object, **kwargs: object) -> object:
            if name == "pyvips":
                raise OSError("cannot load library 'libvips.so.42'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", raise_oserror)
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(200, 300, seed=15)).save(path)

        # the sahi logger sits at INFO, so the debug record is dropped unless it is named
        with caplog.at_level("DEBUG", logger="sahi"):
            source = _open_band_source(path, band_height=SLICE)

        assert not isinstance(source, _VipsBandSource)
        assert not [record for record in caplog.records if record.levelname == "WARNING"]
        assert "pyvips is unusable" in caplog.text


URL = "http://example.invalid/scan.png"


@pytest.fixture
def served_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """Serve a local PNG at `URL`, recording every fetch of it.

    Returns the local path and the list requests are appended to.
    """
    import requests

    local = tmp_path / "scan.png"
    Image.fromarray(make_image(300, 400, seed=5)).save(local)
    payload = local.read_bytes()
    urls: list[str] = []

    class FakeResponse:
        content = payload

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    return local, urls


class TestUrlInput:
    """Test remote inputs, which have no local header to peek at."""

    def test_url_is_downloaded_once(self, served_image: tuple[Path, list[str]]) -> None:
        """Test sizing and slicing a URL share a single download.

        `read_image_size` cannot avoid fetching a remote image in full, so the stream
        keeps what it decoded: re-fetching in `iter_batches` doubles the transfer for
        every remote input.
        """
        local, urls = served_image

        stream = SliceImageStream(
            URL,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        assert len(urls) == 1, "sizing should account for the only download"

        streamed = stream_all(stream)
        assert len(urls) == 1, f"image was fetched {len(urls)} times"

        eager = slice_image(
            image=str(local),
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    @pytest.mark.skipif(not pyvips_installed(), reason="pyvips is an optional extra")
    def test_url_does_not_reach_libvips(
        self, served_image: tuple[Path, list[str]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test a URL skips the vips branch instead of failing into it.

        `pyvips.Image.new_from_file` cannot open a URL, so letting one through costs an
        exception and logs a warning blaming libvips for a file it was never given.
        """
        with caplog.at_level("WARNING"):
            source = _open_band_source(URL, band_height=SLICE)

        assert not isinstance(source, _VipsBandSource)
        assert "libvips cannot stream" not in caplog.text


@pytest.mark.skipif(not pyvips_installed(), reason="pyvips is an optional extra")
class TestVipsBandSource:
    """Test the libvips band source."""

    def test_path_uses_streaming_source_when_pyvips_available(self, tmp_path: Path) -> None:
        """Test a file path selects the streaming source rather than falling back."""
        path = str(tmp_path / "scan.jpg")
        Image.fromarray(make_image(300, 400, seed=3)).save(path, quality=95)

        source = _open_band_source(path, band_height=SLICE)

        assert isinstance(source, _VipsBandSource)

    @pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif"])
    def test_streamed_slices_match_slice_image(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, suffix: str
    ) -> None:
        """Test streamed slices match slice_image, in every container libvips handles.

        The image is tall enough to span several chunk reads, which is where a loader
        whose line counter has drifted starts serving the wrong rows or refuses outright.
        The caplog assertion keeps this honest: a silent fallback to an eager decode
        would otherwise make the comparison pass without streaming anything.
        """
        path = str(tmp_path / f"scan{suffix}")
        Image.fromarray(make_image(600, 400, seed=3)).save(path, quality=95)

        assert isinstance(_open_band_source(path, band_height=SLICE), _VipsBandSource)

        stream = SliceImageStream(
            path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        with caplog.at_level("WARNING"):
            streamed = stream_all(stream)

        assert "decoding the whole image instead" not in caplog.text
        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    @pytest.mark.parametrize("suffix", [".png", ".tif"])
    def test_deep_samples_fall_back_and_match_slice_image(self, tmp_path: Path, suffix: str) -> None:
        """Test 16-bit files take the eager path, which is the only one that matches PIL.

        libvips' colourspace("srgb") rescales 16-bit samples onto 0-255 while PIL's
        `I;16 -> RGB` clips at 255. `_read_local_image_as_arr` already bails to PIL for
        non-uint8 samples so both entry points convert identically; without a matching
        guard here the same file yields different pixels depending on whether the
        optional extra happens to be installed.
        """
        path = str(tmp_path / f"deep{suffix}")
        samples = np.random.default_rng(7).integers(0, 65536, (300, 400), dtype=np.uint16)
        Image.fromarray(samples).save(path)

        assert not isinstance(_open_band_source(path, band_height=SLICE), _VipsBandSource)

        streamed = stream_all(
            SliceImageStream(
                path,
                slice_height=SLICE,
                slice_width=SLICE,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                auto_slice_resolution=False,
            )
        )
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    @pytest.mark.parametrize("suffix", [".jpg", ".tif"])
    def test_cmyk_falls_back_and_matches_slice_image(self, tmp_path: Path, suffix: str) -> None:
        """Test CMYK files take the eager path, which is the only one that matches PIL.

        libvips converts CMYK to sRGB without the Adobe inversion PIL and OpenCV apply, so
        streaming a CMYK JPEG was off by up to 127 levels per channel, which is visibly
        wrong colour rather than a rounding difference. Only `srgb` and `b-w` convert
        identically in both, so anything else falls back.
        """
        path = str(tmp_path / f"cmyk{suffix}")
        Image.fromarray(make_image(300, 400, seed=6)).convert("CMYK").save(path, quality=95)

        assert not isinstance(_open_band_source(path, band_height=SLICE), _VipsBandSource)

        streamed = stream_all(
            SliceImageStream(
                path,
                slice_height=SLICE,
                slice_width=SLICE,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                auto_slice_resolution=False,
            )
        )
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

    def test_a_failing_stream_falls_back_mid_iteration(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test a read that fails part way in falls back to an eager decode.

        _open_band_source guards only the open, but a sequential loader rejects a read it
        dislikes when it reaches it, which can be several bands into the iteration.
        """
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(600, 400, seed=4)).save(path)
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        stream = SliceImageStream(
            path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        calls = {"n": 0}
        original = _VipsBandSource.read_band

        def fail_on_second_band(self: _VipsBandSource, y_min: int, y_max: int) -> np.ndarray:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("out of order read")
            return original(self, y_min, y_max)

        monkeypatch.setattr(_VipsBandSource, "read_band", fail_on_second_band)
        with caplog.at_level("WARNING"):
            streamed = stream_all(stream)

        assert "streaming read failed" in caplog.text
        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")


_PEAK_RSS_PROBE = """
import resource, sys
from sahi.slicing import SliceImageStream

for images, _ in SliceImageStream(
    sys.argv[1], slice_height=640, slice_width=640,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2, auto_slice_resolution=False,
).iter_batches(8):
    pass
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def peak_rss_mb(path: str) -> float:
    """Peak RSS of a fresh process that streams every slice of `path`.

    A child process because ru_maxrss is a high-water mark that never comes back down:
    measuring in the test process would report whatever the rest of the suite allocated.
    """
    probe = subprocess.run([sys.executable, "-c", _PEAK_RSS_PROBE, path], capture_output=True, text=True, check=True)
    return int(probe.stdout.split()[-1]) / 1024  # ru_maxrss is KiB on Linux


def write_uniform_jpeg(path: str, width: int, height: int) -> None:
    """Write a large flat JPEG without ever holding it in memory."""
    import pyvips

    pyvips.Image.black(width, height, bands=3).copy(interpretation="srgb").write_to_file(path)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss is KiB on Linux and bytes elsewhere")
@pytest.mark.skipif(not pyvips_installed(), reason="pyvips is an optional extra")
class TestPeakMemory:
    """Guard the property this path exists for: peak memory must not follow image height.

    Marked `slow` and deselected in CI: it writes a 144 MP JPEG and measures peak RSS in
    child processes, and RSS assertions are not dependable on shared runners. Run it
    locally with `pytest -m slow`; `scripts/benchmark_streaming_slices.py` is the fuller
    measurement.
    """

    def test_peak_memory_grows_far_slower_than_a_whole_image_decode(self, tmp_path: Path) -> None:
        """A 4x taller image at the same width costs far less than decoding it whole.

        A comparison rather than an absolute cap, because baseline RSS varies with
        platform and installed extras while the invariant does not. A reintroduced
        whole-image decode grows the peak by the extra decoded bytes and fails here.

        The bound is a fraction of `decode_growth_mb` rather than a flat cap because
        libvips retains roughly 0.45 bytes per pixel decoded, so the peak does grow with
        height, at about a tenth the rate of a full decode.
        """
        width, short_height, tall_height = 12_000, 3_000, 12_000
        short_path = str(tmp_path / "short.jpg")
        tall_path = str(tmp_path / "tall.jpg")
        write_uniform_jpeg(short_path, width, short_height)
        write_uniform_jpeg(tall_path, width, tall_height)
        # what a whole-image decode would add for the extra rows
        decode_growth_mb = width * (tall_height - short_height) * 3 / 1024**2

        growth_mb = peak_rss_mb(tall_path) - peak_rss_mb(short_path)

        assert growth_mb < decode_growth_mb / 3, (
            f"peak memory grew {growth_mb:.0f} MB for {tall_height - short_height} extra rows; "
            f"a whole-image decode would grow it {decode_growth_mb:.0f} MB"
        )


class TestAwkwardShapes:
    """Test images whose size is not a multiple of the slice stride."""

    @pytest.mark.parametrize(
        "height,width",
        [(128, 128), (129, 127), (200, 300), (256, 256), (300, 151), (65, 400), (400, 65)],
    )
    def test_streamed_slices_match_slice_image(self, tmp_path: Path, height: int, width: int) -> None:
        """Test slices match slice_image when the final band is clamped."""
        path = str(tmp_path / "scan.png")
        Image.fromarray(make_image(height, width, seed=height)).save(path)

        stream = SliceImageStream(
            path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        streamed = stream_all(stream)

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")


class TestChannelNormalisation:
    """Test grayscale and RGBA inputs are normalised to RGB."""

    @pytest.mark.parametrize("mode", ["L", "RGBA"])
    def test_non_rgb_input_matches_slice_image(self, tmp_path: Path, mode: str) -> None:
        """Test non-RGB inputs match slice_image."""
        path = str(tmp_path / f"scan_{mode}.png")
        channels = {"L": 1, "RGBA": 4}[mode]
        pixels = make_image(200, 300, seed=7)[:, :, :1] if channels == 1 else None
        if mode == "L":
            Image.fromarray(pixels[:, :, 0], mode="L").save(path)
        else:
            rgba = np.dstack([make_image(200, 300, seed=8), np.full((200, 300), 200, np.uint8)])
            Image.fromarray(rgba, mode="RGBA").save(path)

        stream = SliceImageStream(
            path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )
        eager = slice_image(
            image=path,
            slice_height=SLICE,
            slice_width=SLICE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            auto_slice_resolution=False,
        )

        streamed = stream_all(stream)

        assert len(streamed) == len(eager)
        for index, (got, want) in enumerate(zip(streamed, eager.images)):
            assert got.shape == want.shape, f"slice {index} shape {got.shape} != {want.shape}"
            np.testing.assert_array_equal(got, want, err_msg=f"slice {index} differs")

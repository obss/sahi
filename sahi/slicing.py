"""Image slicing utilities for splitting large images into tiles."""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from shapely.errors import TopologicalError
from tqdm import tqdm

from sahi.annotation import BoundingBox, Mask
from sahi.logger import logger
from sahi.utils.coco import Coco, CocoAnnotation, CocoImage, create_coco_dict
from sahi.utils.cv import IMAGE_EXTENSIONS_LOSSY, _to_hwc, read_image_as_pil, read_image_size
from sahi.utils.file import load_json, save_json

_CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(1, min(32, _CPU_COUNT * 2))

# Rows per forward read, trading calls into the decoder against the size of the buffer
# the reader holds. Reads go through a single libvips region, so the decoder stays in
# step regardless of where a chunk lands.
_CHUNK_ROWS = 128


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int | None = None,
    slice_width: int | None = None,
    auto_slice_resolution: bool | None = True,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
) -> list[list[int]]:
    """Generate bounding boxes for slicing an image into crops.

    The function calculates the coordinates for each slice based on the provided
    image dimensions, slice size, and overlap ratios. If slice size is not provided
    and auto_slice_resolution is True, the function will automatically determine
    appropriate slice parameters.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these parameters from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        # a negative ratio spaces slices apart instead of overlapping them, which leaves
        # rows no band covers and so cannot be streamed
        if overlap_height_ratio is not None and not 0.0 <= overlap_height_ratio < 1.0:
            raise ValueError("Overlap ratio must be in [0.0, 1.0)")
        if overlap_width_ratio is not None and not 0.0 <= overlap_width_ratio < 1.0:
            raise ValueError("Overlap ratio must be in [0.0, 1.0)")
        y_overlap = int((overlap_height_ratio if overlap_height_ratio is not None else 0.2) * slice_height)
        x_overlap = int((overlap_width_ratio if overlap_width_ratio is not None else 0.2) * slice_width)
    elif auto_slice_resolution:
        x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                x_max = min(image_width, x_max)
                y_max = min(image_height, y_max)
                x_min = max(0, x_max - slice_width)
                y_min = max(0, y_max - slice_height)
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def annotation_inside_slice(annotation: dict, slice_bbox: list[int]) -> bool:
    """Check whether annotation coordinates lie inside slice coordinates.

    Args:
        annotation (dict): Single annotation entry in COCO format.
        slice_bbox (List[int]): Generated from `get_slice_bboxes`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].

    Returns:
        (bool): True if any annotation coordinate lies inside slice.
    """
    left, top, width, height = annotation["bbox"]

    right = left + width
    bottom = top + height

    if left >= slice_bbox[2]:
        return False
    if top >= slice_bbox[3]:
        return False
    if right <= slice_bbox[0]:
        return False
    if bottom <= slice_bbox[1]:
        return False

    return True


def process_coco_annotations(
    coco_annotation_list: list[CocoAnnotation], slice_bbox: list[int], min_area_ratio: float
) -> list[CocoAnnotation]:
    """Slices and filters given list of CocoAnnotation objects with given 'slice_bbox' and 'min_area_ratio'.

    Args:
        coco_annotation_list: List[CocoAnnotation]
            Annotations to slice and filter.
        slice_bbox (List[int]): Generated from `get_slice_bboxes`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].
        min_area_ratio (float): If the cropped annotation area to original
            annotation ratio is smaller than this value, the annotation is
            filtered out. Default 0.1.

    Returns:
        (List[CocoAnnotation]): Sliced annotations.
    """
    sliced_coco_annotation_list: list[CocoAnnotation] = []
    for coco_annotation in coco_annotation_list:
        if annotation_inside_slice(coco_annotation.json, slice_bbox):
            sliced_coco_annotation = coco_annotation.get_sliced_coco_annotation(slice_bbox)
            if sliced_coco_annotation.area / coco_annotation.area >= min_area_ratio:
                sliced_coco_annotation_list.append(sliced_coco_annotation)
    return sliced_coco_annotation_list


class SlicedImage:
    """Container for a sliced image and its metadata."""

    def __init__(self, image: np.ndarray, coco_image: CocoImage, starting_pixel: list[int]) -> None:
        """Initialize SlicedImage.

        Args:
            image: np.array
                Sliced image.
            coco_image: CocoImage
                Coco styled image object that belong to sliced image.
            starting_pixel: list of list of int
                Starting pixel coordinates of the sliced image.
        """
        self.image = image
        self.coco_image = coco_image
        self.starting_pixel = starting_pixel


class SliceImageResult:
    """Container for sliced image results."""

    def __init__(self, original_image_size: list[int], image_dir: str | None = None) -> None:
        """Initialize SliceImageResult.

        Args:
            image_dir: str
                Directory of the sliced image exports.
            original_image_size: list of int
                Size of the unsliced original image in [height, width].
        """
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir

        self._sliced_image_list: list[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage) -> None:
        """Add a sliced image to the result."""
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self) -> list[SlicedImage]:
        """Return list of sliced images."""
        return self._sliced_image_list

    @property
    def images(self) -> list[np.ndarray]:
        """Returns sliced images.

        Returns:
            images: a list of np.array
        """
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def coco_images(self) -> list[CocoImage]:
        """Returns CocoImage representation of SliceImageResult.

        Returns:
            coco_images: a list of CocoImage
        """
        coco_images: list = []
        for sliced_image in self._sliced_image_list:
            coco_images.append(sliced_image.coco_image)
        return coco_images

    @property
    def starting_pixels(self) -> list[list[int]]:
        """Returns a list of starting pixels for each slice.

        Returns:
            starting_pixels: a list of starting pixel coords [x,y]
        """
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> list[str]:
        """Returns a list of filenames for each slice.

        Returns:
            filenames: a list of filenames as str
        """
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def __getitem__(self, i: int | slice | list | tuple) -> dict | list:
        """Get sliced image(s) by index or slice."""

        def _prepare_ith_dict(i: int) -> dict:
            return {
                "image": self.images[i],
                "coco_image": self.coco_images[i],
                "starting_pixel": self.starting_pixels[i],
                "filename": self.filenames[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self) -> int:
        """Return number of sliced images."""
        return len(self._sliced_image_list)


def _slice_file_suffix(image: str | Image.Image | np.ndarray, out_ext: str | None = None) -> str:
    """Resolve the file extension exported slices are written with.

    Taken from the source path when there is one: a path handed in directly, or the
    `filename` a PIL image opened from disk carries. Lossy sources are re-encoded as png
    so repeated slicing does not compound the compression. Arrays, URLs and in-memory
    images have no extension to inherit and also get png.
    """
    if out_ext:
        return out_ext
    if isinstance(image, str) and not image.startswith("http"):
        source_name = image
    elif isinstance(image, Image.Image):
        source_name = str(getattr(image, "filename", ""))
    else:
        return ".png"
    source_suffix = Path(source_name).suffix
    if not source_suffix or source_suffix in IMAGE_EXTENSIONS_LOSSY:
        return ".png"
    return source_suffix


def _export_slice(
    image: np.ndarray,
    starting_pixel: Sequence[int],
    output_dir: str,
    prefix: str,
    suffix: str,
) -> None:
    """Write one slice to `output_dir` under the name pattern `slice_image` uses.

    The slice bbox is recovered from the starting pixel plus the array shape, since
    `get_slice_bboxes` clamps the far edge to the image bounds.
    """
    x_min, y_min = starting_pixel
    height, width = image.shape[:2]
    file_name = f"{prefix}_{x_min}_{y_min}_{x_min + width}_{y_min + height}{suffix}"
    Image.fromarray(image).save(str(Path(output_dir) / file_name))


class SliceExporter:
    """Write slices to disk in the background, under the names `slice_image` uses.

    A context manager so the writer threads are shut down with the caller's loop. Exists
    so a streaming caller can flush each batch as it goes rather than holding every slice
    until the end, which is the memory `SliceImageStream` exists to bound.
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        image: str | os.PathLike | Image.Image | np.ndarray,
        out_ext: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Prepare `output_dir` and resolve the extension slices are written with.

        Args:
            output_dir: Directory to write slices into. Created if missing.
            prefix: File name prefix, as `slice_image`'s `output_file_name`.
            image: The image being sliced; its extension is inherited where it has one.
            out_ext: Extension to force instead of inheriting one.
            max_workers: Cap on writer threads. Defaults to `MAX_WORKERS`.
        """
        self._output_dir = output_dir
        self._prefix = prefix
        self._suffix = _slice_file_suffix(image, out_ext)
        self._max_workers = min(MAX_WORKERS, max_workers) if max_workers else MAX_WORKERS
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> SliceExporter:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def export(self, images: Sequence[np.ndarray], starting_pixels: Sequence[Sequence[int]]) -> None:
        """Write one batch of slices, returning once they are all on disk.

        Draining before returning is deliberate: queueing the writes would keep the batch
        alive alongside the next one, doubling the peak this class is used to avoid.
        """
        if self._executor is None:
            raise RuntimeError("SliceExporter must be used as a context manager")
        list(
            self._executor.map(
                _export_slice,
                images,
                starting_pixels,
                [self._output_dir] * len(images),
                [self._prefix] * len(images),
                [self._suffix] * len(images),
            )
        )


class SliceImageStream:
    """Compute slice geometry from the image header, reading pixels a band at a time.

    `slice_image` holds the whole decoded image while every slice is materialised. This
    exposes the geometry up front, which callers need for progress reporting and
    coordinate shifts, but defers the pixel reads so the full image is never resident.
    """

    def __init__(
        self,
        image: str | os.PathLike | Image.Image | np.ndarray,
        slice_height: int | None = None,
        slice_width: int | None = None,
        overlap_height_ratio: float | None = 0.2,
        overlap_width_ratio: float | None = 0.2,
        auto_slice_resolution: bool | None = True,
        exif_fix: bool = True,
    ) -> None:
        """Initialize the stream and compute slice geometry.

        Args:
            image: File path of image, Pillow Image, or numpy array to be sliced.
            slice_height: Height of each slice. Default None.
            slice_width: Width of each slice. Default None.
            overlap_height_ratio: Fractional overlap in height of each slice. Default 0.2.
            overlap_width_ratio: Fractional overlap in width of each slice. Default 0.2.
            auto_slice_resolution: Calculate slice params from resolution when not given.
            exif_fix: Whether to apply an EXIF fix to the image.
        """
        if isinstance(image, os.PathLike):
            # normalise early: every path check below (and _open_band_source's) tests for
            # str, so a Path would silently skip streaming and decode whole instead.
            image = os.fspath(image)
        if isinstance(image, str) and image.startswith("http"):
            # A remote image has no header to peek at, so read_image_size has to fetch the
            # whole thing anyway. Keep the pixels: re-fetching them in iter_batches would
            # double the transfer, and nothing is deferred by dropping them.
            image = read_image_as_pil(image, exif_fix=exif_fix, return_arr=True)  # type: ignore[assignment]
        self._image = image
        self._exif_fix = exif_fix
        width, height = read_image_size(image, exif_fix=exif_fix)
        if height <= 0 or width <= 0:
            # matches slice_image: an empty image yields no slices, and silently returning
            # zero predictions reads as "nothing detected" rather than "nothing was read".
            raise RuntimeError(f"invalid image size: {(width, height)}")
        self.original_image_height, self.original_image_width = height, width
        self.slice_bboxes = get_slice_bboxes(
            image_height=self.original_image_height,
            image_width=self.original_image_width,
            slice_height=slice_height,
            slice_width=slice_width,
            auto_slice_resolution=auto_slice_resolution,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

    def __len__(self) -> int:
        return len(self.slice_bboxes)

    @property
    def is_streaming(self) -> bool:
        """Whether pixels will be read a band at a time rather than decoded in one piece.

        False when the input is not a local path or the optional libvips extra is
        unusable, in which case the whole image is decoded and peak memory is unchanged.
        """
        return _can_stream(self._image)

    def iter_batches(
        self, batch_size: int = 1, prefetch: bool = True
    ) -> Iterator[tuple[list[np.ndarray], list[list[int]]]]:
        """Yield batches of slices together with their starting pixels.

        Slices are produced one row-band at a time, so at most one band is resident.
        Batches are allowed to span bands: slices are handed out as copies, so carrying
        the tail of one band into the next batch pins only those slices, and stopping at
        each band would give short batches on images a few slices wide.

        Args:
            batch_size: Maximum number of slices per yielded batch.
            prefetch: Decode the next band on a worker thread while the caller works on
                the current batch. Only applies to the libvips source; see `_prefetch`.

        Yields:
            Tuples of (slice images, starting pixels as [x_min, y_min]).
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        bands = _group_bboxes_into_bands(self.slice_bboxes)
        band_height = max((y_max - y_min for (y_min, y_max), _ in bands), default=0)

        def produce() -> Iterator[tuple[list[np.ndarray], list[list[int]]]]:
            # Opened in here, not in iter_batches, so that when this runs on a prefetch
            # worker the libvips region is created and used on the same thread. A
            # VipsRegion is thread-affine: fetching from a region built on another
            # thread aborts the process inside libvips.
            source = _open_band_source(
                self._image,
                band_height=band_height,
                exif_fix=self._exif_fix,
                expected_shape=(self.original_image_height, self.original_image_width),
            )
            yield from self._batches_from(source, bands, batch_size)

        batches = produce()
        # Only libvips releases the GIL, so only it can overlap with the caller. An
        # in-memory source is already decoded, and threading it would add handoffs to
        # hide work that no longer exists.
        if prefetch and self.is_streaming:
            batches = _prefetch(batches)
        # returned rather than delegated to, so this function is not itself a generator
        # and the batch_size check above raises on the call instead of the first `next`
        return batches

    def _batches_from(
        self,
        source: _InMemoryBandSource | _VipsBandSource,
        bands: list[tuple[tuple[int, int], list[list[int]]]],
        batch_size: int,
    ) -> Iterator[tuple[list[np.ndarray], list[list[int]]]]:
        """Cut `bands` out of `source`, batched. See `iter_batches` for the contract."""
        images: list[np.ndarray] = []
        shifts: list[list[int]] = []
        for (y_min, y_max), bboxes in bands:
            try:
                band = source.read_band(y_min, y_max)
            except Exception as error:
                if isinstance(source, _InMemoryBandSource):
                    raise
                # A streaming decoder can fail part way in, well after the open that
                # _open_band_source guards: libvips only rejects a read it dislikes once
                # it reaches it. Re-decode eagerly rather than failing the caller.
                logger.warning(f"streaming read failed ({error}), decoding the whole image instead")
                source = _decode_band_source(
                    self._image,
                    exif_fix=self._exif_fix,
                    expected_shape=(self.original_image_height, self.original_image_width),
                )
                band = source.read_band(y_min, y_max)
            for x_min, _, x_max, _ in bboxes:
                # copy: a streaming source reuses its band buffer, so a view would be
                # rewritten under the caller. np.ascontiguousarray is not enough here, as
                # it returns the input untouched when the slice spans the full width.
                images.append(band[:, x_min:x_max].copy())
                shifts.append([x_min, y_min])
                if len(images) == batch_size:
                    yield images, shifts
                    images, shifts = [], []
        if images:
            yield images, shifts


def _can_stream(image: str | Image.Image | np.ndarray) -> bool:
    """Whether `_open_band_source` will attempt libvips for this input.

    Answers without opening anything, so `iter_batches` can decide about prefetching
    before the source exists. A false positive (libvips is importable but rejects the
    file) only costs a worker thread that overlaps nothing.
    """
    if not isinstance(image, str) or image.startswith("http"):
        return False
    try:
        import pyvips  # noqa: F401
    except Exception:
        return False
    return True


def _prefetch(
    batches: Iterator[tuple[list[np.ndarray], list[list[int]]]], depth: int = 1
) -> Iterator[tuple[list[np.ndarray], list[list[int]]]]:
    """Run `batches` on a worker thread, keeping up to `depth` results ready.

    libvips releases the GIL while it decodes, so the next band can be read while the
    caller is busy with the current batch. Measured on a 256 MP JPEG: 1.5 s of
    GIL-holding work in the consumer cost 0.2 s of wall time instead of 1.5 s.

    `depth` is 1 because each queued item pins a whole batch of slices, which is the
    memory this class exists to bound. Exceptions are re-raised in the caller's thread.
    """
    slots: queue.Queue = queue.Queue(maxsize=depth)
    finished = object()
    stop = threading.Event()

    def pump() -> None:
        try:
            # closed here, on the thread that owns it: `batches` holds a thread-affine
            # VipsRegion, and closing it from the consumer would raise
            # "generator already executing" whenever the producer is mid-read.
            with contextlib.closing(batches):  # type: ignore[type-var]
                for batch in batches:
                    slots.put(batch)
                    if stop.is_set():
                        return
        except BaseException as error:
            # broad on purpose: anything the producer raises is re-raised in the
            # consumer's thread, where the caller can actually see it
            slots.put(error)
        else:
            slots.put(finished)

    worker = threading.Thread(target=pump, name="sahi-slice-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = slots.get()
            if item is finished:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # The caller may abandon us mid-iteration (`break`, or an error downstream).
        # Draining lets the worker finish its blocked `put`, see the stop flag and exit
        # instead of sitting on a full queue for the life of the process.
        stop.set()
        while worker.is_alive():
            with contextlib.suppress(queue.Empty):
                slots.get(timeout=0.1)
        worker.join()


def _group_bboxes_into_bands(
    slice_bboxes: list[list[int]],
) -> list[tuple[tuple[int, int], list[list[int]]]]:
    """Group slice bboxes by their shared vertical extent, preserving order.

    `get_slice_bboxes` emits slices in row-major order, so every slice sharing a
    (y_min, y_max) pair can be cut from a single horizontal band of the image.
    """
    bands: dict[tuple[int, int], list[list[int]]] = {}
    for bbox in slice_bboxes:
        bands.setdefault((bbox[1], bbox[3]), []).append(bbox)
    return list(bands.items())


class _ChunkedBandReader:
    """Assemble overlapping bands from a source that can only be read forwards.

    Consecutive bands overlap, but a sequential decoder cannot rewind to serve the
    overlap a second time. Rows therefore arrive in fixed forward-only chunks, and the
    overlap is carried forward inside a preallocated buffer.
    """

    def __init__(
        self,
        read_chunk: Callable[[int, int], np.ndarray],
        image_height: int,
        band_height: int,
    ) -> None:
        """Initialize the reader.

        Args:
            read_chunk: Callable taking (first row, number of rows) and returning them.
                It is only ever called with monotonically advancing rows.
            image_height: Total height of the source image.
            band_height: Tallest band that will be requested.
        """
        self._read_chunk = read_chunk
        self._image_height = image_height
        self._capacity = band_height + _CHUNK_ROWS
        self._buffer: np.ndarray | None = None
        self._buffer_start = 0  # image row held at buffer[0]
        self._valid = 0  # rows currently held in the buffer
        self._read_pos = 0  # rows already consumed from the source

    def read_band(self, y_min: int, y_max: int) -> np.ndarray:
        """Return rows [y_min, y_max) of the image.

        Bands must be requested with non-decreasing `y_min`. The returned array is a
        view into a reused buffer and is only valid until the next call; copy anything
        that must outlive it.
        """
        self._discard_rows_before(y_min)
        while self._buffer_start + self._valid < y_max:
            self._pull_chunk()
        start = y_min - self._buffer_start
        return self._buffer[start : start + (y_max - y_min)]  # type: ignore[index]

    def _discard_rows_before(self, y_min: int) -> None:
        """Drop rows that no later band can need, compacting in place."""
        drop = y_min - self._buffer_start
        if drop <= 0:
            return
        if drop > self._valid:
            # A gap means rows were never read, which would leave _read_pos behind
            # _buffer_start and make every later band return the wrong rows.
            raise ValueError(f"band starts at row {y_min}, past buffered row {self._buffer_start + self._valid}")
        keep = self._valid - drop
        if keep and self._buffer is not None:
            self._buffer[:keep] = self._buffer[drop : drop + keep]
        self._valid = keep
        self._buffer_start = y_min

    def _pull_chunk(self) -> None:
        """Read the next chunk of rows from the source."""
        rows = min(_CHUNK_ROWS, self._image_height - self._read_pos, self._capacity - self._valid)
        if rows <= 0:
            # read_band loops until the buffer reaches y_max, so a zero-row read would hang
            # instead of failing. Unreachable as constructed: the only source cross-checks
            # its dimensions at open.
            raise ValueError(f"no rows left to read at row {self._read_pos} of {self._image_height}")
        chunk = self._read_chunk(self._read_pos, rows)
        if self._buffer is None:
            self._buffer = np.empty((self._capacity, *chunk.shape[1:]), dtype=chunk.dtype)
        self._buffer[self._valid : self._valid + rows] = chunk
        self._valid += rows
        self._read_pos += rows


class _InMemoryBandSource:
    """Band source for images already decoded in memory; bands are views into the array."""

    def __init__(self, array: np.ndarray, expected_shape: tuple[int, int] | None = None) -> None:
        """Serve bands from `array`, refusing it if it is not the size slices were planned for.

        numpy clamps an out-of-range slice instead of raising, so a geometry/pixels
        disagreement would hand back short bands and mis-sized slices rather than an
        error. `_VipsBandSource` cross-checks its dimensions at open; this is that check
        here, and on both axes: the `band[:, x_min:x_max]` cut in `_batches_from` has the
        same hazard on x, where no quarter turn is needed to reach it.
        """
        actual = array.shape[:2]
        if expected_shape is not None and actual != expected_shape:
            raise ValueError(f"decoded image is {actual} where the header read saw {expected_shape}")
        self._array = array

    def read_band(self, y_min: int, y_max: int) -> np.ndarray:
        """Return rows [y_min, y_max) of the image."""
        return self._array[y_min:y_max]


class _VipsBandSource:
    """Stream bands from a file with libvips, decoding only as far as the current band.

    Requires the optional `pyvips` extra. This is the only source that does not hold the
    whole decoded image at once.

    Memory is not constant in image height: libvips itself retains roughly 0.45 bytes per
    pixel decoded. See docs/predict.md for the measurements.
    """

    def __init__(
        self,
        path: str,
        band_height: int,
        expected_shape: tuple[int, int] | None = None,
    ) -> None:
        """Open `path` for sequential access and prepare the band reader.

        Args:
            path: Local file to stream.
            band_height: Tallest band that will be requested.
            expected_shape: (height, width) the caller computed for the image; a
                mismatch means this source would mis-slice, so it refuses to open.
        """
        import pyvips

        image = pyvips.Image.new_from_file(path, access="sequential")
        # libvips omits the field entirely when the file carries no orientation tag
        orientation = image.get("orientation") if image.get_typeof("orientation") else 1
        if orientation != 1:
            # read_image_as_pil applies every EXIF transform (mirrors and 180s, not only
            # quarter turns) while sequential decoding applies none, so a streamed slice
            # would carry the right geometry and the wrong pixels.
            #
            # This refuses on the tag alone, ignoring exif_fix. Pillow's TIFF plugin
            # applies orientations 2-4 in load() whatever the caller asked for, so
            # honouring the flag here made a mirrored TIFF differ on every slice.
            raise ValueError(f"EXIF orientation {orientation} transforms the image")
        if expected_shape is not None and (image.height, image.width) != expected_shape:
            raise ValueError(f"libvips sees {(image.height, image.width)} where the header read saw {expected_shape}")
        if image.interpretation not in ("srgb", "b-w"):
            # libvips converts CMYK to sRGB without the Adobe inversion PIL and OpenCV
            # apply, which is up to 127 levels of error per channel. An allowlist, since
            # anything unlisted is safer decoded eagerly than converted differently.
            raise ValueError(f"{image.interpretation} converts differently in libvips than in PIL")
        if image.format != "uchar":
            # colourspace() rescales deep samples onto 0-255 where PIL's I;16 -> RGB
            # clips, so the two paths disagree by up to a full level.
            # _read_local_image_as_arr bails to PIL for the same reason.
            raise ValueError(f"{image.format} samples convert differently in libvips than in PIL")
        # Match read_image_as_pil's .convert("RGB"): promote grayscale/CMYK to 3-channel
        # sRGB and drop any alpha, so slices carry the same channels either way.
        image = image.colourspace("srgb")
        if image.bands > 3:
            image = image[0:3]
        self._image = image
        # One region for the whole read. crop(...).numpy() evaluates the pipeline afresh
        # per chunk, which desynchronises the sequential loader's line counter: the TIFF
        # loader rejects that outright ("out of order read"), while the JPEG and PNG
        # loaders happen to hold enough line cache to absorb it.
        self._region = pyvips.Region.new(image)
        self._reader = _ChunkedBandReader(
            self._read_chunk,
            image_height=self._image.height,
            band_height=band_height,
        )

    def _read_chunk(self, y_min: int, rows: int) -> np.ndarray:
        # colourspace("srgb") always yields an 8-bit image, so the buffer is plain uint8
        buffer = self._region.fetch(0, y_min, self._image.width, rows)
        return np.frombuffer(buffer, dtype=np.uint8).reshape(rows, self._image.width, self._image.bands)

    def read_band(self, y_min: int, y_max: int) -> np.ndarray:
        """Return rows [y_min, y_max) of the image."""
        return self._reader.read_band(y_min, y_max)


def _open_band_source(
    image: str | Image.Image | np.ndarray,
    band_height: int,
    exif_fix: bool = True,
    expected_shape: tuple[int, int] | None = None,
) -> _InMemoryBandSource | _VipsBandSource:
    """Select the cheapest band source available for this input.

    Falls back to decoding the whole image when nothing cheaper applies, so behaviour
    never regresses relative to `slice_image`.
    """
    # libvips reads files, so URLs skip it: handing it one costs an exception and a
    # warning blaming libvips for a file it was never given.
    if isinstance(image, str) and not image.startswith("http"):
        try:
            import pyvips  # noqa: F401
        except Exception as error:
            # pyvips binds libvips through cffi at import and raises OSError, not
            # ImportError, when the shared library is absent. Both mean the optional extra
            # is unusable here, which is the documented default.
            logger.debug(f"pyvips is unusable ({error}), decoding the whole image instead")
        else:
            try:
                return _VipsBandSource(image, band_height=band_height, expected_shape=expected_shape)
            except Exception as error:
                # warn: the caller may be relying on streaming to fit in RAM
                logger.warning(f"libvips cannot stream {image} ({error}), decoding the whole image instead")
    return _decode_band_source(image, exif_fix=exif_fix, expected_shape=expected_shape)


def _decode_band_source(
    image: str | Image.Image | np.ndarray,
    exif_fix: bool = True,
    expected_shape: tuple[int, int] | None = None,
) -> _InMemoryBandSource:
    """Decode the whole image up front and serve bands from memory."""
    if isinstance(image, np.ndarray):
        # match read_image_as_pil, which transposes CHW arrays; the transpose is a view
        return _InMemoryBandSource(_to_hwc(image), expected_shape=expected_shape)
    # a Pillow image is already decoded, so there is nothing left to defer
    image_arr: np.ndarray = read_image_as_pil(image, exif_fix=exif_fix, return_arr=True)  # type: ignore[assignment]
    return _InMemoryBandSource(image_arr, expected_shape=expected_shape)


def slice_image(
    image: str | Image.Image | np.ndarray,
    coco_annotation_list: list[CocoAnnotation] | None = None,
    output_file_name: str | None = None,
    output_dir: str | None = None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    auto_slice_resolution: bool | None = True,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
) -> SliceImageResult:
    """Slice a large image into smaller windows. If output_file_name and output_dir is given, export sliced images.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        coco_annotation_list (List[CocoAnnotation], optional): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.
        min_area_ratio (float, optional): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix for lossless image formats and png for lossy formats ('.jpg','.jpeg').
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.
        exif_fix (bool): Whether to apply an EXIF fix to the image.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
    """
    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    # create outdir if not present
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read image directly as an array, so no full-size PIL copy is held alongside it
    image_arr: np.ndarray = read_image_as_pil(image, exif_fix=exif_fix, return_arr=True)  # type: ignore[assignment]
    image_height, image_width = image_arr.shape[:2]
    verboselog("image.shape: " + str((image_width, image_height)))

    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {(image_width, image_height)} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width], image_dir=output_dir)

    suffix = _slice_file_suffix(image, out_ext)

    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_arr[tly:bry, tlx:brx]

        # set image file name and path
        slice_suffixes = "_".join(map(str, slice_bbox))
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"

        # create coco image
        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]
        coco_image = CocoImage(file_name=slice_file_name, height=slice_height, width=slice_width)

        # append coco annotations (if present) to coco image
        if coco_annotation_list is not None:
            min_area_ratio_val: float = min_area_ratio if min_area_ratio is not None else 0.1
            for sliced_coco_annotation in process_coco_annotations(
                coco_annotation_list, slice_bbox, min_area_ratio_val
            ):
                coco_image.add_annotation(sliced_coco_annotation)

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice, coco_image=coco_image, starting_pixel=[slice_bbox[0], slice_bbox[1]]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    # export slices if output directory is provided
    if output_file_name and output_dir:
        # Use a context-managed ThreadPoolExecutor for clean shutdown and
        # limit workers based on CPU count to avoid oversubscription.
        max_workers = min(MAX_WORKERS, len(sliced_image_result))
        max_workers = max(1, max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map will schedule tasks and wait for completion when the context exits
            num = len(sliced_image_result)
            list(
                executor.map(
                    _export_slice,
                    sliced_image_result.images,
                    sliced_image_result.starting_pixels,
                    [output_dir] * num,
                    [output_file_name] * num,
                    [suffix] * num,
                )
            )

    verboselog(
        "Num slices: " + str(n_ims) + " slice_height: " + str(slice_height) + " slice_width: " + str(slice_width)
    )

    return sliced_image_result


def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str,
    output_dir: str | None = None,
    ignore_negative_samples: bool | None = False,
    slice_height: int | None = 512,
    slice_width: int | None = 512,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
) -> tuple[dict, str]:
    """Slice large images given in a directory into smaller windows.

    If output_dir is given, export sliced images and coco file.

    Args:
        coco_annotation_file_path (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            dataset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool, optional): If True, images without annotations
            are ignored. Defaults to False.
        slice_height (int, optional): Height of each slice. Default 512.
        slice_width (int, optional): Width of each slice. Default 512.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image.

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """
    # read coco file
    coco_dict: dict = load_json(coco_annotation_file_path)  # type: ignore[assignment]
    # create image_id_to_annotation_list mapping
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images: list = []

    # iterate over images and slice
    for idx, coco_image in enumerate(tqdm(coco.images)):
        # get image path
        image_path: str = os.path.join(image_dir, coco_image.file_name)
        # get annotation json list corresponding to selected coco image
        # slice image
        try:
            slice_image_result = slice_image(
                image=image_path,
                coco_annotation_list=coco_image.annotations,
                output_file_name=f"{Path(coco_image.file_name).stem}_{idx}",
                output_dir=output_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_area_ratio=min_area_ratio,
                out_ext=out_ext,
                verbose=verbose,
                exif_fix=exif_fix,
            )
            # append slice outputs
            sliced_coco_images.extend(slice_image_result.coco_images)
        except TopologicalError:
            logger.warning(f"Invalid annotation found, skipping this image: {image_path}")

    # create and save coco dict
    ignore_negative_samples_val: bool = ignore_negative_samples if ignore_negative_samples is not None else False
    sliced_coco_dict = create_coco_dict(
        sliced_coco_images, coco_dict["categories"], ignore_negative_samples=ignore_negative_samples_val
    )
    save_path: str = ""
    if output_coco_annotation_file_name and output_dir:
        save_path = str(Path(output_dir) / (output_coco_annotation_file_name + "_coco.json"))
        save_json(sliced_coco_dict, save_path)

    return sliced_coco_dict, save_path


def calc_ratio_and_slice(
    orientation: Literal["vertical", "horizontal", "square"], slide: int = 1, ratio: float = 0.1
) -> tuple[int, int, float, float]:
    """Calculate overlap params according to image resolution.

    Args:
        orientation: image capture angle.
        slide: sliding window.
        ratio: buffer value.

    Returns:
        overlap params.
    """
    if orientation == "vertical":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide * 2, ratio, ratio
    elif orientation == "horizontal":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide * 2, slide, ratio, ratio
    elif orientation == "square":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide, ratio, ratio
    else:
        raise ValueError(f"Invalid orientation: {orientation}. Must be one of 'vertical', 'horizontal', or 'square'.")

    return slice_row, slice_col, overlap_height_ratio, overlap_width_ratio


def calc_resolution_factor(resolution: int) -> int:
    """Calculate power(2,n) and return the closest smaller `n` for resolution.

    Args:
        resolution: the width and height of the image multiplied. such as 1024x720 = 737280.

    Returns:
        Power value of 2 closest to the resolution.
    """
    expo = 0
    while np.power(2, expo) < resolution:
        expo += 1

    return expo - 1


def calc_aspect_ratio_orientation(width: int, height: int) -> Literal["vertical", "horizontal", "square"]:
    """Calculate image capture orientation from aspect ratio.

    Args:
        width: image width.
        height: image height.

    Returns:
        image capture orientation.
    """
    if width < height:
        return "vertical"
    elif width > height:
        return "horizontal"
    else:
        return "square"


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: Literal["vertical", "horizontal", "square"]
) -> tuple[int, int, int, int]:
    """Calculate slice and overlap params according to image resolution.

    Args:
        resolution: str
        height: int
        width: int
        orientation: str.

    Returns:
        x_overlap, y_overlap, slice_width, slice_height
    """
    if resolution == "medium":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=1, ratio=0.8
        )

    elif resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=2, ratio=0.4
        )

    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=4, ratio=0.4
        )
    else:  # low condition
        split_col = 1
        split_row = 1
        overlap_width_ratio = 1
        overlap_height_ratio = 1

    slice_height = height // split_col
    slice_width = width // split_row

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)

    return x_overlap, y_overlap, slice_width, slice_height


def get_resolution_selector(res: str, height: int, width: int) -> tuple[int, int, int, int]:
    """Get slicing parameters based on resolution.

    Args:
        res: resolution of image such as low, medium.
        height: image height.
        width: image width.

    Returns:
        overlap params from slicing params function.
    """
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    x_overlap, y_overlap, slice_width, slice_height = calc_slice_and_overlap_params(
        resolution=res, height=height, width=width, orientation=orientation
    )

    return x_overlap, y_overlap, slice_width, slice_height


def get_auto_slice_params(height: int, width: int) -> tuple[int, int, int, int]:
    """Calculate overlap sliding window and buffer params from image dimensions.

    Factor is the power value of 2 closest to the image resolution:
        - factor <= 18: low resolution image such as 300x300, 640x640
        - 18 < factor <= 21: medium resolution image such as 1024x1024, 1336x960
        - 21 < factor <= 24: high resolution image such as 2048x2048, 2048x4096, 4096x4096
        - factor > 24: ultra-high resolution image such as 6380x6380, 4096x8192.

    Args:
        height: image height.
        width: image width.

    Returns:
        slicing overlap params x_overlap, y_overlap, slice_width, slice_height.
    """
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if factor <= 18:
        return get_resolution_selector("low", height=height, width=width)
    elif 18 <= factor < 21:
        return get_resolution_selector("medium", height=height, width=width)
    elif 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    else:
        return get_resolution_selector("ultra-high", height=height, width=width)


def shift_bboxes(bboxes: Any, offset: Sequence[int]) -> Any:
    """Shift bboxes w.r.t offset.

    Supports Tensor, np.ndarray, and list inputs.

    Args:
        bboxes (Tensor, np.ndarray, list): The bboxes need to be translated. Its shape can
            be (n, 4), which means (x, y, x, y).
        offset (Sequence[int]): The translation offsets with shape of (2, ).

    Returns:
        Tensor, np.ndarray, list: Shifted bboxes.
    """
    shifted_bboxes = []

    if type(bboxes).__module__ == "torch":
        bboxes_is_torch_tensor = True
    else:
        bboxes_is_torch_tensor = False

    # Assert bboxes is iterable
    assert hasattr(bboxes, "__iter__"), "bboxes must be iterable"

    for bbox in bboxes:  # type: ignore[union-attr]
        if bboxes_is_torch_tensor or isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        bbox = BoundingBox(bbox, shift_amount=tuple(offset[:2]))  # type: ignore[arg-type]
        bbox = bbox.get_shifted_box()
        shifted_bboxes.append(bbox.to_xyxy())

    if isinstance(bboxes, np.ndarray):
        return np.stack(shifted_bboxes, axis=0)
    elif bboxes_is_torch_tensor:
        return bboxes.new_tensor(shifted_bboxes)  # type: ignore[attr-defined]
    else:
        return shifted_bboxes


def shift_masks(masks: np.ndarray, offset: Sequence[int], full_shape: Sequence[int]) -> np.ndarray:
    """Shift masks to the original image.

    Args:
        masks (np.ndarray): masks that need to be shifted.
        offset (Sequence[int]): The offset to translate with shape of (2, ).
        full_shape (Sequence[int]): A (height, width) tuple of the huge image's shape.

    Returns:
        np.ndarray: Shifted masks.
    """
    # empty masks
    if masks is None:
        return masks

    shifted_masks = []
    for mask_seg in masks:
        mask = Mask(segmentation=mask_seg, shift_amount=list(offset[:2]), full_shape=list(full_shape[:2]))  # type: ignore[arg-type]
        mask = mask.get_shifted_mask()
        shifted_masks.append(mask.bool_mask)

    return np.stack(shifted_masks, axis=0)

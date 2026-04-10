from __future__ import annotations

from typing import Any

import numpy as np

from sahi.slicers.base import BaseSlicer
from sahi.slicing import CocoImage, SliceImageResult, SlicedImage, get_slice_bboxes
from sahi.utils.cv import read_image_as_pil
from sahi.utils.import_utils import check_requirements


class DALISlicer(BaseSlicer):
    """NVIDIA DALI-based GPU image slicing backend.

    Uses DALI's ``fn.decoders.image_slice`` with hardware-accelerated
    NVDEC decoding to perform fused decode+crop on the GPU.  When a file
    path is provided the image bytes are sent through NVDEC directly,
    avoiding a CPU decode round-trip.

    Falls back to CPU-side NumPy slicing when the input is not a file
    path (e.g. an already-decoded numpy array or PIL image).
    """

    def __init__(
        self,
        slice_height: int | None = 512,
        slice_width: int | None = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        auto_slice_resolution: bool = True,
        device_id: int = 0,
        num_threads: int = 4,
        prefetch_queue_depth: int = 2,
        hw_decoder_load: float = 0.65,
    ) -> None:
        super().__init__(
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            auto_slice_resolution=auto_slice_resolution,
        )
        check_requirements(["nvidia.dali"])
        self.device_id = device_id
        self.num_threads = num_threads
        self.prefetch_queue_depth = prefetch_queue_depth
        self.hw_decoder_load = hw_decoder_load

    # -- public API (overrides BaseSlicer) ------------------------------------

    def slice_image(
        self,
        image: str | Any,
        verbose: bool = False,
        exif_fix: bool = True,
    ) -> SliceImageResult:
        """Slice an image using DALI GPU pipeline.

        When *image* is a file path, DALI decodes and crops in a single
        fused GPU operation.  Otherwise, falls back to reading the image
        on the CPU and slicing with NumPy (DALI still benefits from GPU
        transpose/cast if needed downstream).

        Returns:
            SliceImageResult – same type as all other backends.
        """
        # We need image dimensions to compute slice bboxes.  For file
        # paths we do a cheap PIL open (header only) to get the size.
        image_pil = read_image_as_pil(image, exif_fix=exif_fix)
        image_width, image_height = image_pil.size

        slice_bboxes = get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            auto_slice_resolution=self.auto_slice_resolution,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )

        if isinstance(image, str):
            crops = self._dali_slice_from_path(image, slice_bboxes)
        else:
            # Fallback: image already decoded – use plain NumPy slicing.
            image_arr = np.asarray(image_pil)
            crops = [image_arr[tly:bry, tlx:brx] for tlx, tly, brx, bry in slice_bboxes]

        result = SliceImageResult(original_image_size=[image_height, image_width])
        for bbox, crop in zip(slice_bboxes, crops):
            tlx, tly, brx, bry = bbox
            coco_image = CocoImage(file_name="", height=bry - tly, width=brx - tlx)
            result.add_sliced_image(
                SlicedImage(image=crop, coco_image=coco_image, starting_pixel=[tlx, tly])
            )

        return result

    # -- DALI pipeline --------------------------------------------------------

    def _dali_slice_from_path(
        self, image_path: str, slice_bboxes: list[list[int]]
    ) -> list[np.ndarray]:
        """Run a DALI pipeline that decodes + crops each slice on the GPU.

        One pipeline iteration is executed per slice.  The image bytes
        are read once and reused for every ROI crop via
        ``fn.decoders.image_slice``.
        """
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali import pipeline_def

        raw_bytes = np.fromfile(image_path, dtype=np.uint8)

        crops: list[np.ndarray] = []
        for bbox in slice_bboxes:
            x_min, y_min, x_max, y_max = bbox
            begin = np.array([y_min, x_min, 0], dtype=np.float32)
            size = np.array([y_max - y_min, x_max - x_min, -1], dtype=np.float32)

            @pipeline_def(
                batch_size=1,
                num_threads=self.num_threads,
                device_id=self.device_id,
                prefetch_queue_depth=self.prefetch_queue_depth,
            )
            def roi_pipe():
                encoded = fn.external_source(
                    source=[[raw_bytes]], dtype=types.UINT8, batch=True
                )
                decoded = fn.decoders.image_slice(
                    encoded,
                    start=fn.external_source(
                        source=[[begin]], dtype=types.FLOAT, batch=True
                    ),
                    shape=fn.external_source(
                        source=[[size]], dtype=types.FLOAT, batch=True
                    ),
                    device="mixed",
                    hw_decoder_load=self.hw_decoder_load,
                    output_type=types.RGB,
                )
                return decoded

            pipe = roi_pipe()
            pipe.build()
            (output,) = pipe.run()
            # output is a TensorListGPU – move to CPU as HWC uint8 ndarray
            crop_arr = np.array(output.as_cpu()[0])
            crops.append(crop_arr)

        return crops

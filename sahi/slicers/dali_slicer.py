from __future__ import annotations

from typing import Any

import numpy as np

from sahi.slicers.base import BaseSlicer
from sahi.slicing import CocoImage, SliceImageResult, SlicedImage, get_slice_bboxes
from sahi.utils.cv import read_image_as_pil
from sahi.utils.import_utils import check_requirements


class DALISlicer(BaseSlicer):
    """NVIDIA DALI-based GPU image slicing backend.

    For **file-path** input, uses ``fn.decoders.image_slice`` with
    hardware-accelerated NVDEC to perform fused decode+crop on the GPU —
    all slices are decoded in a single batched pipeline run.

    For **numpy/PIL** input (e.g. video frames already in memory), uploads
    the full image to the GPU once and uses ``fn.crop`` to extract all
    slices in one batched run.
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
        self.hw_decoder_load = hw_decoder_load

    # -- public API (overrides BaseSlicer) ------------------------------------

    def slice_image(
        self,
        image: str | Any,
        verbose: bool = False,
        exif_fix: bool = True,
    ) -> SliceImageResult:
        """Slice an image using a batched DALI GPU pipeline.

        Returns:
            SliceImageResult -- same type as all other backends.
        """
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
            crops = self._dali_decode_and_slice(image, slice_bboxes)
        else:
            crops = self._dali_crop_array(np.asarray(image_pil), slice_bboxes)

        result = SliceImageResult(original_image_size=[image_height, image_width])
        for bbox, crop in zip(slice_bboxes, crops):
            tlx, tly, brx, bry = bbox
            coco_image = CocoImage(file_name="", height=bry - tly, width=brx - tlx)
            result.add_sliced_image(
                SlicedImage(image=crop, coco_image=coco_image, starting_pixel=[tlx, tly])
            )

        return result

    # -- DALI pipelines -------------------------------------------------------

    def _dali_decode_and_slice(
        self, image_path: str, slice_bboxes: list[list[int]]
    ) -> list[np.ndarray]:
        """Fused NVDEC decode + crop: one batched pipeline run for all slices.

        Each sample in the batch receives the same encoded bytes but a
        different ROI, so NVDEC only decodes the requested region.
        """
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali import pipeline_def

        raw_bytes = np.fromfile(image_path, dtype=np.uint8)
        num_slices = len(slice_bboxes)

        # Replicate encoded bytes for each sample in the batch
        encoded_batch = [raw_bytes] * num_slices

        # Build ROI coordinate arrays (int32, [y, x])
        start_batch = []
        shape_batch = []
        for x_min, y_min, x_max, y_max in slice_bboxes:
            start_batch.append(np.array([y_min, x_min], dtype=np.int32))
            shape_batch.append(np.array([y_max - y_min, x_max - x_min], dtype=np.int32))

        @pipeline_def(
            batch_size=num_slices,
            num_threads=self.num_threads,
            device_id=self.device_id,
            prefetch_queue_depth=1,
        )
        def decode_slice_pipe():
            encoded = fn.external_source(
                source=[encoded_batch], dtype=types.UINT8, batch=True,
            )
            roi_start = fn.external_source(
                source=[start_batch], dtype=types.INT32, batch=True,
            )
            roi_shape = fn.external_source(
                source=[shape_batch], dtype=types.INT32, batch=True,
            )
            decoded = fn.decoders.image_slice(
                encoded,
                start=roi_start,
                shape=roi_shape,
                device="mixed",
                hw_decoder_load=self.hw_decoder_load,
                output_type=types.RGB,
            )
            return decoded

        pipe = decode_slice_pipe()
        pipe.build()
        (output,) = pipe.run()

        # TensorListGPU -> list of CPU ndarrays (HWC uint8)
        output_cpu = output.as_cpu()
        return [np.array(output_cpu[i]) for i in range(num_slices)]

    def _dali_crop_array(
        self, image_arr: np.ndarray, slice_bboxes: list[list[int]]
    ) -> list[np.ndarray]:
        """GPU crop for already-decoded images (video frames, numpy arrays).

        Uploads the full image to GPU once, then extracts all crops in a
        single batched pipeline run using ``fn.crop``.
        """
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali import pipeline_def

        num_slices = len(slice_bboxes)

        # Replicate the full image for each sample in the batch
        image_batch = [image_arr] * num_slices

        # fn.crop uses (y, x) anchor and (h, w) shape -- float32 normalized
        # or absolute pixel coords.  We use absolute int coords.
        anchor_batch = []
        shape_batch = []
        for x_min, y_min, x_max, y_max in slice_bboxes:
            anchor_batch.append(np.array([y_min, x_min], dtype=np.int32))
            shape_batch.append(np.array([y_max - y_min, x_max - x_min], dtype=np.int32))

        @pipeline_def(
            batch_size=num_slices,
            num_threads=self.num_threads,
            device_id=self.device_id,
            prefetch_queue_depth=1,
        )
        def crop_pipe():
            images = fn.external_source(
                source=[image_batch], dtype=types.UINT8, batch=True,
            )
            images_gpu = images.gpu()
            crop_anchor = fn.external_source(
                source=[anchor_batch], dtype=types.INT32, batch=True,
            )
            crop_shape = fn.external_source(
                source=[shape_batch], dtype=types.INT32, batch=True,
            )
            cropped = fn.crop(
                images_gpu,
                crop_pos_x=fn.element_extract(crop_anchor, element_map=[1]),
                crop_pos_y=fn.element_extract(crop_anchor, element_map=[0]),
                crop_w=fn.element_extract(crop_shape, element_map=[1]),
                crop_h=fn.element_extract(crop_shape, element_map=[0]),
                out_of_bounds_policy="error",
            )
            return cropped

        pipe = crop_pipe()
        pipe.build()
        (output,) = pipe.run()

        output_cpu = output.as_cpu()
        return [np.array(output_cpu[i]) for i in range(num_slices)]

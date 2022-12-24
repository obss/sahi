# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

import os
import time
from typing import List, Union

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

from sahi.prediction import PredictionResult
from sahi.slicing import SliceImageResult, get_slice_bboxes, slice_image
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil
from sahi.utils.file import list_files


def collate_fn(input_batch):
    output_batch = {}
    output_batch["filepaths"] = [item["filepath"] for item in input_batch]
    output_batch["image_ids"] = [item["image_id"] for item in input_batch]
    output_batch["offset_amounts"] = [item["offset_amount"] for item in input_batch]
    output_batch["images"] = [item["image"] for item in input_batch]
    output_batch["full_shapes"] = [item["full_shape"] for item in input_batch]
    return output_batch


def process_predictions(postprocess_queue, result_queue, max_counter):
    image_id = 0
    prediction_results = []
    object_predictions = []
    counter = 0
    while counter < (max_counter - 1):
        print(counter, max_counter - 1)
        try:
            # Try to get next element from queue
            args = postprocess_queue.get()
            new_image_id = args["image_id"]
            sliced_object_predictions = args["object_predictions_per_image"]
            image_path = args["image_path"]
            confidence_threshold = args["confidence_threshold"]
            postprocess = args["postprocess"]

            if new_image_id != image_id:
                # postprocess matching predictions
                if postprocess is not None:
                    object_predictions = postprocess(object_predictions)

                prediction_result = PredictionResult(
                    image=image_path,
                    object_predictions=object_predictions,
                    durations_in_seconds=None,
                )
                prediction_results.append(prediction_result)
                object_predictions = []
            image_id = new_image_id

            # filter out predictions with lower score
            sliced_object_predictions = [
                object_prediction
                for object_prediction in sliced_object_predictions
                if object_prediction.score.value > confidence_threshold
            ]

            # append slice predictions
            object_predictions.extend(sliced_object_predictions)
            print("len_object_predictions: ", len(object_predictions))
            print("len_prediction_results: ", len(prediction_results))

            counter += 1
        except:
            # Wait if queue is empty
            time.sleep(0.01)  # queue is either empty or no update

    result_queue.put(prediction_results)


class ImageDataset(IterableDataset):
    def __init__(
        self,
        image: Union[np.ndarray, Image.Image, str] = None,
        image_folder_dir: str = None,
        image_ids: List[int] = None,
        include_slices: bool = False,
        include_original: bool = True,
        slice_height: int = 512,
        slice_width: int = 512,
        slice_overlap_height_ratio: float = 0.2,
        slice_overlap_width_ratio: float = 0.2,
        target_image_format: str = "rgb",
        export_folder_dir: str = None,
        verbose: int = 1,
        load_images_as_numpy: bool = True,
    ):
        self.image = image
        self.image_folder_dir = image_folder_dir
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.slice_overlap_height_ratio = slice_overlap_height_ratio
        self.slice_overlap_width_ratio = slice_overlap_width_ratio
        self.target_image_format = target_image_format
        self.export_folder_dir = export_folder_dir
        self.include_original = include_original
        self.include_slices = include_slices
        self.load_images_as_numpy = load_images_as_numpy

        self._image_paths = []
        self._samples = []
        self._current_image_id = 0

        if image_ids is None:
            self.image_ids = []
        else:
            self.image_ids = image_ids

        if image is None and image_folder_dir is None:
            raise ValueError("Either image or image_folder_dir must be provided.")

        if image is not None and image_folder_dir is not None:
            raise ValueError("Only one of image or image_folder_dir must be provided.")

        if image is None and image_folder_dir is not None:
            if not os.path.isdir(image_folder_dir):
                raise ValueError(f"image_folder_dir {image_folder_dir} does not exist.")
            self._image_paths = list_files(
                directory=image_folder_dir,
                contains=IMAGE_EXTENSIONS,
                verbose=verbose,
            )
            self._prepare_samples_from_image_folder()
        elif image is not None and image_folder_dir is None:
            self._prepare_samples_from_single_image()
        else:
            raise NotImplementedError

        self._sample_iterator = iter(self._samples)

    def _slice_image(self, image) -> SliceImageResult:
        sliced_image_result = slice_image(
            image,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.slice_overlap_height_ratio,
            overlap_width_ratio=self.slice_overlap_width_ratio,
        )
        return sliced_image_result

    def _get_slice_bboxes(self, image_height: int, image_width: int) -> List[List[int]]:
        slice_bboxes = get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            auto_slice_resolution=False,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.slice_overlap_height_ratio,
            overlap_width_ratio=self.slice_overlap_width_ratio,
        )
        return slice_bboxes

    def _prepare_samples_from_single_image(self):
        pil_image = read_image_as_pil(self.image)
        image = np.array(pil_image) if self.load_images_as_numpy else self.image
        filepath = self.image if isinstance(self.image, str) else "image.png"

        # calculate number of samples for this image
        num_image_samples = 0
        if self.include_original:
            num_image_samples += 1
        if self.include_slices:
            sliced_image_result = self._slice_image(image)
            num_image_samples += len(sliced_image_result)

        if self.include_original:
            image_width, image_height = pil_image.size
            self._samples.append(
                {
                    "image": image,
                    "image_id": self._current_image_id,
                    "num_image_samples": num_image_samples,
                    "offset_amount": [0, 0],
                    "slice_bbox": None,
                    "full_shape": [image_height, image_width],
                    "filepath": filepath,
                }
            )

        if self.include_slices:
            for sliced_image in sliced_image_result.sliced_images:
                self._samples.append(
                    {
                        "image": sliced_image.image,
                        "image_id": self._current_image_id,
                        "num_image_samples": num_image_samples,
                        "offset_amount": sliced_image.starting_pixel,
                        "slice_bbox": None,
                        "full_shape": [
                            sliced_image_result.original_image_height,
                            sliced_image_result.original_image_width,
                        ],
                        "filepath": filepath,
                    }
                )

        self._current_image_id += 1

    def _prepare_samples_from_image_folder(self):
        for image_path in self._image_paths:
            pil_image = read_image_as_pil(image_path)
            image_width, image_height = pil_image.size

            # calculate number of samples for this image
            num_image_samples = 0
            if self.include_original:
                num_image_samples += 1
            if self.include_slices:
                slice_bboxes = self._get_slice_bboxes(image_height=image_height, image_width=image_width)
                num_image_samples += len(slice_bboxes)

            if self.include_original:
                self._samples.append(
                    {
                        "image": None,
                        "image_id": self._current_image_id,
                        "num_image_samples": num_image_samples,
                        "offset_amount": [0, 0],
                        "slice_bbox": None,
                        "full_shape": [image_height, image_width],
                        "filepath": image_path,
                    }
                )

            if self.include_slices:
                for slice_bbox in slice_bboxes:
                    self._samples.append(
                        {
                            "image": None,
                            "image_id": self._current_image_id,
                            "num_image_samples": num_image_samples,
                            "offset_amount": [slice_bbox[0], slice_bbox[1]],
                            "slice_bbox": slice_bbox,
                            "full_shape": [image_height, image_width],
                            "filepath": image_path,
                        }
                    )

            self._current_image_id += 1

    def __iter__(self):
        try:
            while True:
                sample = next(self._sample_iterator)
                if sample["image"] is None:
                    image = read_image_as_pil(sample["filepath"])
                    image = np.array(image) if self.load_images_as_numpy else image
                    if sample["slice_bbox"] is not None:
                        image = image[
                            sample["slice_bbox"][1] : sample["slice_bbox"][3],
                            sample["slice_bbox"][0] : sample["slice_bbox"][2],
                        ]
                    sample["image"] = image
                yield sample
        except StopIteration:
            return None

    def __len__(self):
        return len(self._samples)

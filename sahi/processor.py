import os
import time
from typing import List, Union

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from sahi.prediction import PredictionResult
from sahi.slicing import SliceImageResult, get_slice_bboxes, slice_image
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil
from sahi.utils.file import list_files


def collate_fn(input_batch):
    output_batch = {}
    output_batch["image_paths"] = [item["image_paths"] for item in input_batch]
    output_batch["relative_image_paths"] = [item["relative_image_paths"] for item in input_batch]
    output_batch["image_ids"] = [item["image_ids"] for item in input_batch]
    output_batch["offset_amounts"] = [item["offset_amounts"] for item in input_batch]
    output_batch["images"] = [item["images"] for item in input_batch]
    output_batch["full_shapes"] = [item["full_shapes"] for item in input_batch]
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


class ImageProcessor(IterableDataset):
    def __init__(
        self,
        image: Union[np.ndarray, Image.Image, str] = None,
        image_folder_dir: str = None,
        image_ids: List[int] = None,
        include_slices: bool = False,
        include_original: bool = True,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        target_image_format: str = "rgb",
        batch_size: int = 1,
        export_folder_dir: str = None,
        verbose: int = 1,
        load_images_as_numpy: bool = True,
    ):
        self.image = image
        self.image_folder_dir = image_folder_dir
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.target_image_format = target_image_format
        self.batch_size = batch_size
        self.export_folder_dir = export_folder_dir
        self.include_original = include_original
        self.include_slices = include_slices
        self.load_images_as_numpy = load_images_as_numpy

        self._image_paths = []
        self._image_queue = []
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

    def _slice_new_image(self, image) -> SliceImageResult:
        sliced_image_result = slice_image(
            image,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )
        return sliced_image_result

    def _populate_image_queue(self):
        if self.include_original:
            if self.image:
                pil_image = read_image_as_pil(self.image)
                image = np.array(pil_image) if self.load_images_as_numpy else self.image
                filepath = self.image if isinstance(self.image, str) else "image.png"
            elif self.image_folder_dir:
                filepath = next(self._image_paths)
                pil_image = read_image_as_pil(filepath)
                image = np.array(pil_image) if self.load_images_as_numpy else filepath

            image_width, image_height = pil_image.size
            if self.target_image_format == "bgr" and self.load_images_as_numpy:
                image = image[:, :, ::-1]

            sample = {
                "image": image,
                "image_id": self._current_image_id,
                "offset_amount": [0, 0],
                "full_shape": [image_height, image_width],
                "filepath": filepath,
            }
            self._image_queue.append(sample)

        if self.include_slices:
            if self.image:
                sliced_image_result = self._slice_new_image(self.image)
                filepath = self.image if isinstance(self.image, str) else "image.png"
            elif self.image_folder_dir:
                filepath = next(self._image_paths)
                pil_image = read_image_as_pil(filepath)
                sliced_image_result = self._slice_new_image(pil_image)

            for sliced_image in sliced_image_result.sliced_images:
                if self.target_image_format == "bgr":
                    image = image[:, :, ::-1]
                sample = {
                    "image": sliced_image.image,
                    "image_id": self._current_image_id,
                    "offset_amount": sliced_image.starting_pixel,
                    "full_shape": [sliced_image_result.original_image_height, sliced_image_result.original_image_width],
                    "filepath": filepath,
                }
                self._image_queue.append(sample)

        self._current_image_id += 1

    def _get_next_image(self):
        if len(self._image_queue) == 0:
            self._populate_image_queue()
        return self._image_queue.pop(0)

    def __iter__(self):
        return self._get_next_image()

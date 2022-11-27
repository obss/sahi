import os
import time

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sahi.prediction import PredictionResult
from sahi.slicing import get_slice_bboxes
from sahi.utils.cv import read_image_as_pil


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


class SAHIImageDataset(Dataset):
    def __init__(
        self,
        image_path=None,
        image_dir=None,
        image_ids=None,
        sliced_prediction=False,
        standard_prediction=True,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        target_image_format="rgb",
    ):
        self.image_dir = image_dir
        self.sliced_prediction = sliced_prediction
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.target_image_format = target_image_format

        relative_image_path_list = []
        new_image_path_list = []
        slice_bbox_list = []
        new_image_id_list = []
        for ind, image_path in enumerate(tqdm(image_paths, "preparing dataloader")):
            if image_ids is not None:
                image_id = ind
            else:
                image_id = image_ids[ind]
            # get filename
            if self.image_dir is not None:  # preserve source folder structure in export
                relative_filepath = image_path.split(self.image_dir)[-1]
                relative_filepath = relative_filepath[1:] if relative_filepath[0] == os.sep else relative_filepath
            else:  # no process if source is single file
                relative_filepath = image_path
            relative_image_path_list.append(relative_filepath)

            # prepare image slices and paths
            image_width, image_height = read_image_as_pil(image_path).size
            if sliced_prediction:
                slice_bboxes = get_slice_bboxes(
                    image_height=image_height,
                    image_width=image_width,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                )
                for slice_bbox in slice_bboxes:
                    slice_bbox_list.append(slice_bbox)
                    relative_image_path_list.append(relative_filepath)
                    new_image_path_list.append(image_path)
                    new_image_id_list.append(image_id)
            if standard_prediction:
                slice_bboxes = get_slice_bboxes(
                    image_height=image_height,
                    image_width=image_width,
                    slice_height=image_height,
                    slice_width=image_width,
                    overlap_height_ratio=0,
                    overlap_width_ratio=0,
                )
                slice_bbox_list.append(slice_bboxes[0])
                relative_image_path_list.append(relative_filepath)
                new_image_path_list.append(image_path)
                new_image_id_list.append(image_id)

        self.slice_bbox_list = slice_bbox_list
        self.image_path_list = new_image_path_list
        self.image_id_list = new_image_id_list
        self.relative_image_path_list = relative_image_path_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = read_image_as_pil(self.image_path_list[idx])
        image_width, image_height = image.size

        sample = {}
        sample["image_paths"] = self.image_path_list[idx]
        sample["relative_image_paths"] = self.relative_image_path_list[idx]
        sample["image_ids"] = self.image_id_list[idx]
        sample["offset_amounts"] = (self.slice_bbox_list[idx][0], self.slice_bbox_list[idx][1])
        image = image.crop(self.slice_bbox_list[idx])
        if self.target_image_format == "bgr":
            image = image[:, :, ::-1]
        sample["images"] = image
        sample["full_shapes"] = (image_height, image_width)

        return sample

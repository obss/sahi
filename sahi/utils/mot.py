import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from sahi.utils.file import increment_path


try:
    import norfair
    from norfair import Tracker, Detection
    from norfair.metrics import PredictionsTextFile, InformationFile
except ImportError:
    raise ImportError('Please run "pip install -U norfair" to install norfair first for MOT format handling.')


class GroundTruthTextFile(PredictionsTextFile):
    def __init__(self, save_path="."):

        predictions_folder = os.path.join(save_path, "gt")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)

        self.out_file_name = os.path.join(predictions_folder, "gt" + ".txt")

        self.frame_number = 1

    def update(self, predictions, frame_number=None):
        if frame_number is None:
            frame_number = self.frame_number
        """
        Write tracked object information in the output file (for this frame), in the format
        frame_number, id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1
        """
        text_file = open(self.out_file_name, "a+")

        for obj in predictions:
            frame_str = str(int(frame_number))
            id_str = str(int(obj.id))
            bb_left_str = str((obj.estimate[0, 0]))
            bb_top_str = str((obj.estimate[0, 1]))  # [0,1]
            bb_width_str = str((obj.estimate[1, 0] - obj.estimate[0, 0]))
            bb_height_str = str((obj.estimate[1, 1] - obj.estimate[0, 1]))
            row_text_out = (
                frame_str
                + ","
                + id_str
                + ","
                + bb_left_str
                + ","
                + bb_top_str
                + ","
                + bb_width_str
                + ","
                + bb_height_str
                + ",1,-1,-1,-1"
            )
            text_file.write(row_text_out)
            text_file.write("\n")

        self.frame_number += 1

        text_file.close()


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


class MotAnnotation:
    def __init__(self, bbox: List[int], score: Optional[float] = 1):
        """
        Args:
            bbox (List[int]): [x_min, y_min, width, height]
            score (Optional[float])
        """
        self.bbox = bbox
        self.score = score


class MotFrame:
    def __init__(self):
        self.annotation_list: List[MotAnnotation] = []

    def add_annotation(self, detection: MotAnnotation):
        assert type(detection) == MotAnnotation, "'detection' should be a MotAnnotation object."
        self.annotation_list.append(detection)

    def to_norfair_detections(self, track_points: str = "bbox"):
        """
        Args:
            track_points (str): 'centroid' or 'bbox'. Defaults to 'bbox'.
        """
        norfair_detections: List[Detection] = []
        # convert all detections to norfair detections
        for annotation in self.annotation_list:
            # calculate bbox points
            xmin = annotation.bbox[0]
            ymin = annotation.bbox[1]
            xmax = annotation.bbox[0] + annotation.bbox[2]
            ymax = annotation.bbox[1] + annotation.bbox[3]
            scores = None
            # calculate points as bbox or centroid
            if track_points == "bbox":
                points = np.array([[xmin, ymin], [xmax, ymax]])  # bbox
                if annotation.score is not None:
                    scores = np.array([annotation.score, annotation.score])

            elif track_points == "centroid":
                points = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])  # centroid
                if annotation.score is not None:
                    scores = np.array([annotation.score])
            else:
                ValueError("'track_points' should be one of ['centroid', 'bbox'].")
            # create norfair formatted detection
            norfair_detections.append(Detection(points=points, scores=scores))
        return norfair_detections


class MotVideo:
    def __init__(
        self, export_dir: str = "runs/mot", track_points: str = "bbox", tracker_kwargs: Optional[Dict] = dict()
    ):
        """
        Args
            export_dir (str): Folder directory that will contain gt/gt.txt and seqinfo.ini
                For details: https://github.com/tryolabs/norfair/issues/42#issuecomment-819211873
            track_points (str): Track detections based on 'centroid' or 'bbox'. Defaults to 'bbox'.
            tracker_kwargs (dict): a dict contains the tracker keys as below:
                - max_distance_between_points (int)
                - min_detection_threshold (float)
                - hit_inertia_min (int)
                - hit_inertia_max (int)
                - point_transience (int)
                For details: https://github.com/tryolabs/norfair/tree/master/docs#arguments
        """

        self.export_dir: str = str(increment_path(Path(export_dir), exist_ok=False))
        self.track_points: str = track_points

        self.groundtruth_text_file: Optional[GroundTruthTextFile] = None
        self.tracker: Optional[Tracker] = None

        self._create_gt_file()
        self._init_tracker(
            tracker_kwargs.get("max_distance_between_points", 30),
            tracker_kwargs.get("min_detection_threshold", 0),
            tracker_kwargs.get("hit_inertia_min", 10),
            tracker_kwargs.get("hit_inertia_max", 12),
            tracker_kwargs.get("point_transience", 4),
        )

    def _create_info_file(self, seq_length: int):
        """
        Args:
            seq_length (int): Number of frames present in video (seqLength parameter in seqinfo.ini)
                For details: https://github.com/tryolabs/norfair/issues/42#issuecomment-819211873
        """
        # set file path
        filepath = Path(self.export_dir) / "seqinfo.ini"
        # create folder directory if not exists
        filepath.parent.mkdir(exist_ok=True)
        # create seqinfo.ini file with seqLength
        with open(str(filepath), "w") as file:
            file.write(f"seqLength={seq_length}")

    def _create_gt_file(self):
        self.groundtruth_text_file = GroundTruthTextFile(save_path=self.export_dir)

    def _init_tracker(
        self,
        max_distance_between_points: int = 30,
        min_detection_threshold: float = 0,
        hit_inertia_min: int = 10,
        hit_inertia_max: int = 12,
        point_transience: int = 4,
    ):
        """
        Args
            max_distance_between_points (int)
            min_detection_threshold (float)
            hit_inertia_min (int)
            hit_inertia_max (int)
            point_transience (int)
        For details: https://github.com/tryolabs/norfair/tree/master/docs#arguments
        """
        self.tracker = Tracker(
            distance_function=euclidean_distance,
            initialization_delay=0,
            distance_threshold=max_distance_between_points,
            detection_threshold=min_detection_threshold,
            hit_inertia_min=hit_inertia_min,
            hit_inertia_max=hit_inertia_max,
            point_transience=point_transience,
        )

    def add_frame(self, frame: MotFrame):
        assert type(frame) == MotFrame, "'frame' should be a MotFrame object."
        norfair_detections: List[Detection] = frame.to_norfair_detections(track_points=self.track_points)
        tracked_objects = self.tracker.update(detections=norfair_detections)
        self.groundtruth_text_file.update(predictions=tracked_objects)
        self._create_info_file(seq_length=self.groundtruth_text_file.frame_number)

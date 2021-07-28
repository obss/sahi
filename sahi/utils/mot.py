import copy
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sahi.utils.file import increment_path

try:
    import norfair
    from norfair import Detection, Tracker
    from norfair.metrics import InformationFile, PredictionsTextFile
    from norfair.tracker import FilterSetup, TrackedObject
except ImportError:
    raise ImportError('Please run "pip install -U norfair" to install norfair first for MOT format handling.')


class MotTextFile(PredictionsTextFile):
    def __init__(self, save_dir: str = ".", save_name: str = "gt"):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.out_file_name = os.path.join(save_dir, save_name + ".txt")

        self.frame_number = 0

    def update(self, predictions: List[TrackedObject], frame_number: int = None):
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
    def __init__(self, bbox: List[int], track_id: Optional[int] = -1, score: Optional[float] = 1):
        """
        Args:
            bbox (List[int]): [x_min, y_min, width, height]
            track_id: (Optional[int]): track id of the annotation
            score (Optional[float])
        """
        self.bbox = bbox
        self.track_id = track_id
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

    def to_norfair_trackedobjects(self, track_points: str = "bbox"):
        """
        Args:
            track_points (str): 'centroid' or 'bbox'. Defaults to 'bbox'.
        """
        tracker = Tracker(
            distance_function=euclidean_distance,
            distance_threshold=30,
            detection_threshold=0,
            hit_inertia_min=0,
            hit_inertia_max=12,
            point_transience=4,
        )

        tracked_object_list: List[TrackedObject] = []
        # convert all detections to norfair detections
        for annotation in self.annotation_list:
            # ensure annotation.track_id is not None
            assert annotation.track_id is not None, TypeError(
                "to_norfair_trackedobjects() requires annotation.track_id to be set."
            )
            # calculate bbox points
            xmin = annotation.bbox[0]
            ymin = annotation.bbox[1]
            xmax = annotation.bbox[0] + annotation.bbox[2]
            ymax = annotation.bbox[1] + annotation.bbox[3]
            track_id = annotation.track_id
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
            detection = Detection(points=points, scores=scores)
            # create trackedobject from norfair detection
            tracked_object = TrackedObject(
                detection,
                tracker.hit_inertia_min,
                tracker.hit_inertia_max,
                tracker.initialization_delay,
                tracker.detection_threshold,
                period=1,
                point_transience=tracker.point_transience,
                filter_setup=tracker.filter_setup,
            )
            tracked_object.id = track_id
            tracked_object.point_hit_counter = np.ones(tracked_object.num_points) * 1
            # append to tracked_object_list
            tracked_object_list.append(tracked_object)
        return tracked_object_list


class MotVideo:
    def __init__(
        self,
        name: str = "sequence_name",
        frame_rate: Optional[int] = 30,
        image_height: int = 720,
        image_width: int = 1280,
        tracker_kwargs: Optional[Dict] = dict(),
    ):
        """
        Args
            name (str): Name of the video file.
            frame_rate (int): FPS of the video.
            image_height (int): Frame height of the video.
            image_width (int): Frame width of the video.
            tracker_kwargs (dict): a dict contains the tracker keys as below:
                - max_distance_between_points (int)
                - min_detection_threshold (float)
                - hit_inertia_min (int)
                - hit_inertia_max (int)
                - point_transience (int)
                For details: https://github.com/tryolabs/norfair/tree/master/docs#arguments
        """

        self.name = name
        self.frame_rate = frame_rate
        self.image_height = image_height
        self.image_width = image_width
        self.tracker_kwargs = tracker_kwargs

        self.frame_list: List[MotFrame] = []

    def _create_info_file(self, seq_length: int, export_dir: str):
        """
        Args:
            seq_length (int): Number of frames present in video (seqLength parameter in seqinfo.ini)
                For details: https://github.com/tryolabs/norfair/issues/42#issuecomment-819211873
            export_dir (str): Folder directory that will contain exported file.
        """
        # set file path
        filepath = Path(export_dir) / "seqinfo.ini"
        # create folder directory if not exists
        filepath.parent.mkdir(exist_ok=True)
        # create seqinfo.ini file with seqLength
        with open(str(filepath), "w") as file:
            file.write(f"seqLength={seq_length}\n")
            file.write(f"frameRate={self.frame_rate}\n")
            file.write(f"imWidth={self.image_width}\n")
            file.write(f"imHeight={self.image_height}")

    def add_frame(self, frame: MotFrame):
        assert type(frame) == MotFrame, "'frame' should be a MotFrame object."
        self.frame_list.append(frame)

    def export(self, export_dir: str = "runs/mot", type: str = "gt", use_tracker: bool = None, exist_ok=False):
        """
        Args
            export_dir (str): Folder directory that will contain exported mot challenge formatted data.
            type (str): Type of the MOT challenge export. 'gt' for groundturth data export, 'det' for detection data export.
            use_tracker (bool): Determines whether to apply kalman based tracker over frame detections or not.
                Default is True for type='gt'.
                It is always False for type='det'.
            exist_ok (bool): If True overwrites given directory.
        """
        assert type in ["gt", "det"], TypeError(f"'type' can be one of ['gt', 'det'], you provided: {type}")

        export_dir: str = str(increment_path(Path(export_dir), exist_ok=exist_ok))

        if type == "gt":
            gt_dir = os.path.join(export_dir, self.name if self.name else "", "gt")
            mot_text_file: MotTextFile = MotTextFile(save_dir=gt_dir, save_name="gt")
            if not use_tracker:
                use_tracker = True
        elif type == "det":
            det_dir = os.path.join(export_dir, self.name if self.name else "", "det")
            mot_text_file: MotTextFile = MotTextFile(save_dir=det_dir, save_name="det")
            use_tracker = False

        tracker = Tracker(
            distance_function=self.tracker_kwargs.get("distance_function", euclidean_distance),
            distance_threshold=self.tracker_kwargs.get("distance_threshold", 50),
            hit_inertia_min=self.tracker_kwargs.get("hit_inertia_min", 1),
            hit_inertia_max=self.tracker_kwargs.get("hit_inertia_max", 1),
            initialization_delay=self.tracker_kwargs.get("initialization_delay", 0),
            detection_threshold=self.tracker_kwargs.get("detection_threshold", 0),
            point_transience=self.tracker_kwargs.get("point_transience", 4),
            filter_setup=self.tracker_kwargs.get("filter_setup", FilterSetup(R=0.2)),
        )

        for mot_frame in self.frame_list:
            if use_tracker:
                norfair_detections: List[Detection] = mot_frame.to_norfair_detections(track_points="bbox")
                tracked_objects = tracker.update(detections=norfair_detections)
            else:
                tracked_objects = mot_frame.to_norfair_trackedobjects(track_points="bbox")
            mot_text_file.update(predictions=tracked_objects)

        if type == "gt":
            info_dir = os.path.join(export_dir, self.name if self.name else "")
            self._create_info_file(seq_length=mot_text_file.frame_number, export_dir=info_dir)

# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import shutil
import unittest


class TestMotUtils(unittest.TestCase):
    def test_mot_vid(self):
        from sahi.utils.mot import MotAnnotation, MotFrame, MotVideo

        export_dir = "tests/data/mot/"
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)

        mot_video = MotVideo(name="video.mp4")
        # frame 0
        mot_frame = MotFrame()
        mot_detection = MotAnnotation(bbox=[10, 10, 100, 100])
        mot_frame.add_annotation(mot_detection)
        mot_video.add_frame(mot_frame)
        # frame 1
        mot_frame = MotFrame()
        mot_detection = MotAnnotation(bbox=[12, 12, 98, 98])
        mot_frame.add_annotation(mot_detection)
        mot_detection = MotAnnotation(bbox=[95, 95, 98, 98])
        mot_frame.add_annotation(mot_detection)
        mot_video.add_frame(mot_frame)
        # export
        mot_video.export(export_dir=export_dir, type="gt", exist_ok=True)

        mot_video = MotVideo(name="video.mp4")
        # frame 0
        mot_frame = MotFrame()
        mot_detection = MotAnnotation(bbox=[10, 10, 100, 100])
        mot_frame.add_annotation(mot_detection)
        mot_video.add_frame(mot_frame)
        # frame 1
        mot_frame = MotFrame()
        mot_detection = MotAnnotation(bbox=[12, 12, 98, 98])
        mot_frame.add_annotation(mot_detection)
        mot_detection = MotAnnotation(bbox=[95, 95, 98, 98])
        mot_frame.add_annotation(mot_detection)
        mot_video.add_frame(mot_frame)
        # export
        mot_video.export(export_dir=export_dir, type="det", exist_ok=True)


if __name__ == "__main__":
    unittest.main()

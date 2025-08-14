# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2025.

from unittest.mock import patch


class TestFileUtils:
    def test_list_files(self):
        from sahi.utils.file import list_files

        directory = "tests/data/coco_utils/"
        filepath_list = list_files(directory, contains=["json"], verbose=False)
        assert len(filepath_list) == 11

    def test_list_files_recursively(self):
        from sahi.utils.file import list_files_recursively

        directory = "tests/data/coco_utils/"
        relative_filepath_list, abs_filepath_list = list_files_recursively(
            directory, contains=["coco.json"], verbose=False
        )
        assert len(relative_filepath_list) == 7
        assert len(abs_filepath_list) == 7

    def test_increment_path(self):
        from sahi.utils.file import increment_path

        with patch("sahi.utils.file.Path.exists", return_value=False):
            path = increment_path("test.txt")
            assert path == "test.txt"
        with patch("sahi.utils.file.Path.exists", return_value=True):
            path = increment_path("test.txt", exist_ok=False)
            assert path == "test.txt2"

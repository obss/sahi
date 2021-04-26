# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest


class TestFileUtils(unittest.TestCase):
    def test_list_files(self):
        from sahi.utils.file import list_files

        directory = "tests/data/coco_utils/"
        filepath_list = list_files(directory, contains=["json"], verbose=False)
        self.assertEqual(len(filepath_list), 11)

    def test_list_files_recursively(self):
        from sahi.utils.file import list_files_recursively

        directory = "tests/data/coco_utils/"
        relative_filepath_list, abs_filepath_list = list_files_recursively(
            directory, contains=["coco.json"], verbose=False
        )
        self.assertEqual(len(relative_filepath_list), 7)
        self.assertEqual(len(abs_filepath_list), 7)


if __name__ == "__main__":
    unittest.main()

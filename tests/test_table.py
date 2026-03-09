from __future__ import annotations

import unittest

from sahi.utils.table import create_ascii_table


class TestTable(unittest.TestCase):
    def test_create_ascii_table_basic(self):
        data = [
            ["Model", "Params(M)", "Dataset"],
            ["ResNet50", 25.6, "ImageNet"],
            ["YOLOv8n", 4.5, "COCO"],
            ["Swin-Transformer", 88.2, "ADE20K"],
        ]
        table = create_ascii_table(data)

        # Verify basic structure
        self.assertIn("ResNet50", table)
        self.assertIn("Dataset", table)
        self.assertTrue(table.startswith("+"))
        self.assertTrue(table.endswith("+"))

        # Verify alignment (Swin-Transformer is the longest model name)
        # Content row should contain the full model name padded correctly.
        self.assertIn("| Swin-Transformer |", table)

    def test_empty_data(self):
        self.assertEqual(create_ascii_table([]), "")
        self.assertEqual(create_ascii_table([[]]), "")

    def test_none_values(self):
        data = [["ID", "Value"], [1, None]]
        table = create_ascii_table(data)
        self.assertIn("| 1  |       |", table)

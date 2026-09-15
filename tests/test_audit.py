import unittest
from pathlib import Path

from src.audit import label_for_path, parse_extensions


class AuditHelpersTest(unittest.TestCase):
    def test_parse_extensions_normalizes_dots_and_case(self) -> None:
        self.assertEqual(parse_extensions("jpg,.PNG"), {".jpg", ".png"})

    def test_label_level_one_uses_parent_folder(self) -> None:
        root = Path("data")
        self.assertEqual(label_for_path(root / "thai_01" / "image.jpg", root, 1), "thai_01")

    def test_label_level_two_uses_grandparent_folder(self) -> None:
        root = Path("data")
        self.assertEqual(label_for_path(root / "thai_01" / "source_a" / "image.jpg", root, 2), "thai_01")


if __name__ == "__main__":
    unittest.main()

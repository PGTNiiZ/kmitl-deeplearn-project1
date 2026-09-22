import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.download_data import drive_file_id, gdown_command
from src.paths import (DEFAULT_CLEAN_DATA_DIR, DRIVE_URL_KEY, ENVIRONMENT_KEY,
                       configured_data_dir, get_data_dir, get_drive_url,
                       prepare_image_directory)


class DataPathTest(unittest.TestCase):
    def test_defaults_to_clean_dataset_directory(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("src.paths._read_dotenv_value", return_value=None):
            self.assertEqual(configured_data_dir(), DEFAULT_CLEAN_DATA_DIR)

    def test_environment_variable_has_priority(self) -> None:
        with patch.dict(os.environ, {ENVIRONMENT_KEY: "/tmp/thai-data"}, clear=True):
            self.assertEqual(configured_data_dir(), Path("/tmp/thai-data"))

    def test_existing_environment_path_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.dict(os.environ, {ENVIRONMENT_KEY: temporary_dir}, clear=True):
                self.assertEqual(get_data_dir(), Path(temporary_dir).resolve())

    def test_zip_path_is_extracted_and_returns_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_path = Path(temporary_dir) / "clean_32x32.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("clean_32x32/consonants/0_test/image.png", b"not inspected here")
            data_dir = prepare_image_directory(archive_path, {".png"})
            self.assertEqual(data_dir, Path(temporary_dir, "clean_32x32").resolve())
            self.assertTrue((data_dir / "consonants/0_test/image.png").is_file())

    def test_google_drive_url_is_read_from_environment(self) -> None:
        url = "https://drive.google.com/drive/folders/example"
        with patch.dict(os.environ, {DRIVE_URL_KEY: url}, clear=True):
            self.assertEqual(get_drive_url(), url)

    def test_google_drive_file_url_id_is_detected(self) -> None:
        url = "https://drive.google.com/file/d/1BnPkvJJlE7QDZ0sgEDujS23Pr8Cj3IKq/view?usp=sharing"
        self.assertEqual(drive_file_id(url), "1BnPkvJJlE7QDZ0sgEDujS23Pr8Cj3IKq")
        self.assertIsNone(drive_file_id("https://drive.google.com/drive/folders/example"))
        command = gdown_command(url, Path("cache"))
        self.assertEqual(command[3], "1BnPkvJJlE7QDZ0sgEDujS23Pr8Cj3IKq")
        self.assertNotIn("--id", command)

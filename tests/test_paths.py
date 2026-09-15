import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.paths import DEFAULT_DATA_DIR, DRIVE_URL_KEY, ENVIRONMENT_KEY, configured_data_dir, get_data_dir, get_drive_url


class DataPathTest(unittest.TestCase):
    def test_defaults_to_repository_data_raw(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("src.paths._read_dotenv_value", return_value=None):
            self.assertEqual(configured_data_dir(), DEFAULT_DATA_DIR)

    def test_environment_variable_has_priority(self) -> None:
        with patch.dict(os.environ, {ENVIRONMENT_KEY: "/tmp/thai-data"}, clear=True):
            self.assertEqual(configured_data_dir(), Path("/tmp/thai-data"))

    def test_existing_environment_path_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.dict(os.environ, {ENVIRONMENT_KEY: temporary_dir}, clear=True):
                self.assertEqual(get_data_dir(), Path(temporary_dir).resolve())

    def test_google_drive_url_is_read_from_environment(self) -> None:
        url = "https://drive.google.com/drive/folders/example"
        with patch.dict(os.environ, {DRIVE_URL_KEY: url}, clear=True):
            self.assertEqual(get_drive_url(), url)

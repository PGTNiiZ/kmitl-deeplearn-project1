"""Shared dataset-path resolution for scripts and Jupyter notebooks.

Resolution order:
1. THAI_CHAR_DATA_DIR environment variable
2. THAI_CHAR_DATA_DIR in the repository's untracked .env file
3. <repository>/data/raw
"""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "raw"
ENVIRONMENT_KEY = "THAI_CHAR_DATA_DIR"
DRIVE_URL_KEY = "THAI_CHAR_DRIVE_URL"


def _read_dotenv_value(dotenv_path: Path, key: str) -> str | None:
    """Read one simple KEY=VALUE value without adding a notebook dependency."""
    if not dotenv_path.is_file():
        return None
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate_key, candidate_value = line.split("=", 1)
        if candidate_key.strip() == key:
            return candidate_value.strip().strip('"').strip("'") or None
    return None


def configured_data_dir() -> Path:
    """Return configured data directory without checking whether it exists."""
    raw_value = os.environ.get(ENVIRONMENT_KEY) or _read_dotenv_value(REPOSITORY_ROOT / ".env", ENVIRONMENT_KEY)
    return Path(raw_value).expanduser() if raw_value else DEFAULT_DATA_DIR


def get_data_dir() -> Path:
    """Return an existing dataset directory or raise an actionable error."""
    data_dir = configured_data_dir()
    if data_dir.is_dir():
        return data_dir.resolve()
    raise FileNotFoundError(
        f"Dataset directory was not found: {data_dir}\n"
        f"Either place the dataset in {DEFAULT_DATA_DIR}, or copy .env.example to .env "
        f"and set {ENVIRONMENT_KEY} to your own local/mounted dataset path."
    )


def get_drive_url() -> str:
    """Return a Google Drive folder URL from environment or the private .env file."""
    value = os.environ.get(DRIVE_URL_KEY) or _read_dotenv_value(REPOSITORY_ROOT / ".env", DRIVE_URL_KEY)
    if not value:
        raise ValueError(
            f"{DRIVE_URL_KEY} is not configured. Copy .env.example to .env and set the shared Google Drive folder URL."
        )
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or parsed.netloc not in {"drive.google.com", "www.drive.google.com"}:
        raise ValueError(f"{DRIVE_URL_KEY} must be an https://drive.google.com/... URL, not: {value}")
    return value

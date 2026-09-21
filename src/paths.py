"""Shared dataset-path resolution for scripts and Jupyter notebooks.

Resolution order:
1. THAI_CHAR_DATA_DIR environment variable
2. THAI_CHAR_DATA_DIR in the repository's untracked .env file
3. <repository>/data/raw/clean_32x32
"""

from __future__ import annotations

import os
import urllib.parse
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_CLEAN_DATA_DIR = DEFAULT_DATA_DIR / "clean_32x32"
DEFAULT_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
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
    """Return the configured dataset directory or ZIP path without checking it."""
    raw_value = os.environ.get(ENVIRONMENT_KEY) or _read_dotenv_value(REPOSITORY_ROOT / ".env", ENVIRONMENT_KEY)
    return Path(raw_value).expanduser() if raw_value else DEFAULT_CLEAN_DATA_DIR


def get_data_dir() -> Path:
    """Return an existing dataset directory or raise an actionable error."""
    data_dir = prepare_image_directory(configured_data_dir(), DEFAULT_IMAGE_EXTENSIONS)
    if data_dir.is_dir():
        return data_dir.resolve()
    raise FileNotFoundError(
        f"Dataset was not found: {configured_data_dir()}\n"
        f"Either place the extracted dataset in {DEFAULT_CLEAN_DATA_DIR}, or copy .env.example to .env "
        f"and set {ENVIRONMENT_KEY} to your own local/mounted dataset path."
    )


def prepare_image_directory(data_dir: Path, extensions: set[str]) -> Path:
    """Extract a local ZIP cache when needed and return its image root.

    Some shared Drive exports contain one archive whose top-level directory is
    the dataset (for example ``raw/ThaiCharacter Dataset.zip`` containing
    ``round2/<class>/<image>.jpg``).  Training code needs the extracted
    directory, with class folders immediately below it.  The helper leaves an
    already usable image directory untouched.
    """
    def has_images(directory: Path) -> bool:
        if not directory.is_dir():
            return False
        return any(
            path.is_file() and path.suffix.lower() in extensions
            for path in directory.rglob("*")
            if not any(part.startswith(".") for part in path.relative_to(directory).parts)
        )

    data_dir = data_dir.expanduser().resolve()
    if data_dir.is_file():
        if data_dir.suffix.lower() != ".zip":
            return data_dir
        archive_path = data_dir
        expected_root = archive_path.with_suffix("")
        if not has_images(expected_root):
            with zipfile.ZipFile(archive_path) as archive:
                target = archive_path.parent.resolve()
                for member in archive.infolist():
                    destination = (target / member.filename).resolve()
                    if destination != target and target not in destination.parents:
                        raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
                archive.extractall(target)
        if expected_root.is_dir():
            return expected_root
        data_dir = archive_path.parent
    elif not data_dir.is_dir():
        matching_archive = data_dir.with_suffix(".zip")
        return prepare_image_directory(matching_archive, extensions) if matching_archive.is_file() else data_dir

    if not has_images(data_dir):
        archives = sorted(path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() == ".zip")
        if len(archives) == 1:
            with zipfile.ZipFile(archives[0]) as archive:
                target = data_dir.resolve()
                for member in archive.infolist():
                    destination = (target / member.filename).resolve()
                    if destination != target and target not in destination.parents:
                        raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
                archive.extractall(target)
        elif len(archives) > 1:
            raise ValueError(f"Found multiple ZIP archives in {data_dir}; set DATA_PATH to the extracted dataset folder.")

    # A single wrapper directory (e.g. raw/round2) is not a label; use it as
    # the image root so LABEL_LEVEL=1 still selects each class folder.
    children = sorted(path for path in data_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    image_children = [child for child in children if has_images(child)]
    direct_images = any(path.parent == data_dir for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    if not direct_images and len(image_children) == 1:
        return image_children[0]
    return data_dir


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

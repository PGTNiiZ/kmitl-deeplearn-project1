"""Download the shared public Google Drive dataset to a local training cache.

The Google Drive URL remains the team source of truth. Images are cached locally
because notebooks and PyTorch DataLoaders need filesystem files for fast,
reproducible batch loading.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from src.paths import DEFAULT_DATA_DIR, get_drive_url


def has_visible_files(directory: Path) -> bool:
    return directory.is_dir() and any(not item.name.startswith(".") for item in directory.iterdir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the configured public Google Drive dataset.")
    parser.add_argument("--drive-url", help="Optional Google Drive folder URL; overrides THAI_CHAR_DRIVE_URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR, help="Local cache directory (default: data/raw).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow download into a non-empty output directory. Existing files may be affected by gdown.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        drive_url = args.drive_url or get_drive_url()
    except ValueError as error:
        print(f"Download setup failed: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if has_visible_files(output_dir) and not args.force:
        print(
            f"Refusing to download into non-empty directory: {output_dir}\n"
            "Use a new --output-dir, or inspect it and pass --force deliberately.",
            file=sys.stderr,
        )
        return 2
    if importlib.util.find_spec("gdown") is None:
        print("gdown is not installed. Run: python3 -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gdown", "--folder", drive_url, "-O", str(output_dir)]
    print(f"Downloading shared Google Drive folder to local cache: {output_dir}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            "Google Drive download failed. Confirm that the folder is shared as 'Anyone with the link: Viewer', "
            "then retry. Do not train from a partial download.",
            file=sys.stderr,
        )
        return completed.returncode or 1
    print("Download complete. Run: python3 -m src.audit --output-dir results/audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Group 01: modern CNN, mobile CNN, high-resolution CNN and transformer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.team_benchmark import run_group


CANDIDATES = [
    {"family": "ConvNeXt", "model": "convnext_tiny"},
    {"family": "MobileNetV3", "model": "mobilenetv3_large_100"},
    {"family": "HRNet", "model": "hrnet_w18_small_v2"},
    {"family": "PVTv2", "model": "pvt_v2_b1"},
]


if __name__ == "__main__":
    raise SystemExit(run_group(1, CANDIDATES))

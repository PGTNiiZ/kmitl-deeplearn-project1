"""Group 03: compact modern, efficient and hybrid architectures."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.team_benchmark import run_group


CANDIDATES = [
    {"family": "ConvNeXtV2", "model": "convnextv2_atto"},
    {"family": "EdgeNeXt", "model": "edgenext_xx_small"},
    {"family": "RegNet", "model": "regnety_032"},
    {"family": "MobileViT", "model": "mobilevit_xxs"},
]


if __name__ == "__main__":
    raise SystemExit(run_group(3, CANDIDATES))

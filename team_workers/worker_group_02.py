"""Group 02: efficient hybrids and conventional CNN controls."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.team_benchmark import run_group


CANDIDATES = [
    {"family": "FastViT", "model": "fastvit_t8"},
    {"family": "RepViT", "model": "repvit_m1"},
    {"family": "ResNet", "model": "resnet18"},
    {"family": "DenseNet", "model": "densenet121"},
    {"family": "CustomCNN", "model": "custom_cnn"},
]


if __name__ == "__main__":
    raise SystemExit(run_group(2, CANDIDATES))

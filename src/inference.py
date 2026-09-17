"""Load a trusted team checkpoint and apply exactly the training preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.train import TrainConfig, build_model, choose_device, make_transform


class Predictor:
    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.device = choose_device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = TrainConfig(**state["config"])
        self.labels = [label for label, index in sorted(state["label_to_index"].items(), key=lambda pair: pair[1])]
        self.transform = make_transform(self.config)
        self.model = build_model(self.config, len(self.labels), load_pretrained=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, paths, top_k: int = 3, batch_size: int = 32):
        if top_k < 1 or batch_size < 1:
            raise ValueError("top_k and batch_size must be positive")
        results = []
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset:offset + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            with torch.inference_mode():
                probs = self.model(torch.stack(tensors).to(self.device)).softmax(1)
                scores, indices = probs.topk(min(top_k, len(self.labels)), dim=1)
            for path, values, indexes in zip(batch_paths, scores.cpu().tolist(), indices.cpu().tolist()):
                results.append({"path": str(path), "predicted_label": self.labels[indexes[0]],
                                "confidence": values[0], "top_k": [
                                    {"label": self.labels[i], "confidence": value} for i, value in zip(indexes, values)]})
        return results


def main():
    from src.audit import DEFAULT_EXTENSIONS, image_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path, help="One image or a folder")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    paths = list(image_paths(args.image, DEFAULT_EXTENSIONS)) if args.image.is_dir() else [args.image]
    print(json.dumps(Predictor(args.weights, args.device).predict(paths), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

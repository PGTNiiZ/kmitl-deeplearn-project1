"""Offline inference for the 72-class Thai character classifier.

The classifier is a closed-set isolated-character recognizer.  Optional detector
integration localizes separated characters but does not claim connected-text OCR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, ImageOps

from src.train import TrainConfig, build_model, make_transform, choose_device

SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = REPOSITORY_ROOT / "results" / "model_search" / "full" / "e90d3777438a" / "export"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float | None
    top_k: list[dict[str, Any]]
    status: str


@dataclass(frozen=True)
class DetectorBox:
    bbox: tuple[float, float, float, float]
    confidence: float


def resolve_device(name: str) -> torch.device:
    """Resolve and validate a requested backend without silently falling back."""
    return choose_device(name)


def discover_defaults() -> tuple[Path, Path, Path]:
    """Return repository-relative export artifacts, if present."""
    return DEFAULT_EXPORT / "best_model.pt", DEFAULT_EXPORT / "best_config.json", DEFAULT_EXPORT / "label_to_index.json"


def _read_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}: {exc}") from exc


def load_config(path: Path | None, checkpoint_state: dict[str, Any]) -> TrainConfig:
    embedded = checkpoint_state.get("config")
    external = _read_json(path, "Config") if path else None
    if external is not None and embedded is not None:
        for key in set(external) | set(embedded):
            external_value, embedded_value = external.get(key), embedded.get(key)
            # JSON serializes the checkpoint's mean/std tuples as lists.
            if key in {"mean", "std"} and external_value is not None and embedded_value is not None:
                external_value, embedded_value = tuple(external_value), tuple(embedded_value)
            if external_value != embedded_value:
                raise ValueError(f"Config/checkpoint mismatch for '{key}': {external.get(key)!r} != {embedded.get(key)!r}")
    raw = external or embedded
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint has no usable model config; pass --config")
    try:
        return TrainConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Unsupported or incomplete TrainConfig: {exc}") from exc


def load_label_mapping(path: Path | None, checkpoint_state: dict[str, Any]) -> dict[str, int]:
    embedded = checkpoint_state.get("label_to_index")
    external = _read_json(path, "Label mapping") if path else None
    if external is not None and embedded is not None and external != embedded:
        raise ValueError("External label mapping disagrees with the checkpoint label mapping")
    raw = external or embedded
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Label mapping is missing or empty")
    try:
        mapping = {str(label): int(index) for label, index in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Label mapping must map labels to integer indices") from exc
    indices = list(mapping.values())
    if len(set(indices)) != len(indices) or sorted(indices) != list(range(len(indices))):
        raise ValueError("Label mapping indices must be unique and exactly 0..N-1")
    return mapping


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to load checkpoint safely: {path}: {exc}") from exc
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        return state
    if isinstance(state, dict) and isinstance(state.get("model_state_dict"), dict):
        return {**state, "state_dict": state["model_state_dict"]}
    if isinstance(state, dict) and all(isinstance(k, str) for k in state) and state:
        return {"state_dict": state}
    raise ValueError("Unsupported checkpoint structure; expected state_dict/model_state_dict wrapper")


def validate_output_dimension(model: torch.nn.Module, classes: int, image_size: int) -> None:
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, image_size, image_size))
    if output.ndim != 2 or output.shape[1] != classes:
        raise ValueError(f"Model output dimension {tuple(output.shape)} disagrees with {classes} labels")


def _composite_rgb(image: Image.Image, background: int) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", image.size, (background,) * 3)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return image.convert("RGB")


def load_image_safe(path: Path, background: int = 255) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as source:
            image = _composite_rgb(source, background)
            image.load()
            return image.copy()
    except Exception as exc:
        raise ValueError(f"Image unreadable or unsupported ({path}): {exc}") from exc


def _tta_views(image: Image.Image, mode: str, fill: int = 255) -> list[Image.Image]:
    if mode == "none":
        return [image]
    if mode != "safe":
        raise ValueError("tta must be none or safe")
    # Fixed, non-random views.  No flips, crops, or large rotations are used.
    return [image, image.rotate(-3, resample=Image.Resampling.BILINEAR, fillcolor=(fill,) * 3),
            image.rotate(3, resample=Image.Resampling.BILINEAR, fillcolor=(fill,) * 3),
            ImageOps.expand(image, border=(1, 0, 0, 0), fill=(fill,) * 3).crop((0, 0, image.width, image.height)),
            ImageOps.expand(image, border=(0, 0, 1, 0), fill=(fill,) * 3).crop((1, 0, image.width + 1, image.height))]


def expand_bbox(box: Sequence[float], image_size: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    if margin < 0 or margin > 1:
        raise ValueError("bbox margin must be between 0 and 1")
    width, height = image_size
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    coords = (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
              min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy)))
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError(f"Invalid zero-area bounding box: {box}")
    return coords


def sort_reading_order(boxes: Sequence[DetectorBox]) -> list[DetectorBox]:
    if not boxes:
        return []
    heights = [max(1.0, b.bbox[3] - b.bbox[1]) for b in boxes]
    tolerance = sum(heights) / len(heights) * 0.6
    lines: list[list[DetectorBox]] = []
    for box in sorted(boxes, key=lambda b: ((b.bbox[1] + b.bbox[3]) / 2, b.bbox[0])):
        center = (box.bbox[1] + box.bbox[3]) / 2
        target = next((line for line in lines
                       if abs(center - sum((item.bbox[1] + item.bbox[3]) / 2 for item in line) / len(line)) <= tolerance), None)
        if target is None:
            lines.append([box])
        else:
            target.append(box)
    lines.sort(key=lambda line: sum((item.bbox[1] + item.bbox[3]) / 2 for item in line) / len(line))
    return [box for line in lines for box in sorted(line, key=lambda item: item.bbox[0])]


def _detector_predictions(image: Image.Image, weights: Path, confidence: float) -> list[DetectorBox]:
    try:
        from ultralytics import YOLO  # optional dependency; never imported for single mode
    except ImportError as exc:
        raise RuntimeError("Detector unavailable: install/configure ultralytics for multi mode") from exc
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights missing: {weights}")
    results = YOLO(str(weights)).predict(source=image, conf=confidence, verbose=False)
    boxes: list[DetectorBox] = []
    for result in results:
        if result.boxes is None:
            continue
        for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            boxes.append(DetectorBox(tuple(float(x) for x in xyxy), float(score)))
    return boxes


class Predictor:
    def __init__(self, checkpoint: Path, device: str = "auto", config_path: Path | None = None,
                 labels_path: Path | None = None, confidence_threshold: float = 0.0):
        self.checkpoint = checkpoint.resolve()
        self.device = resolve_device(device)
        self.state = _load_checkpoint(self.checkpoint)
        self.config = load_config(config_path, self.state)
        self.mapping = load_label_mapping(labels_path, self.state)
        self.labels = [label for label, _ in sorted(self.mapping.items(), key=lambda pair: pair[1])]
        self.confidence_threshold = confidence_threshold
        self.transform = make_transform(self.config)
        self.model = build_model(self.config, len(self.labels), load_pretrained=False)
        try:
            incompatible = self.model.load_state_dict(self.state["state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint state_dict is incompatible with architecture {self.config.architecture}: {exc}") from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"Checkpoint keys mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
        self.model.to(self.device).eval()
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.validate()

    def validate(self) -> None:
        validate_output_dimension(self.model.cpu(), len(self.labels), self.config.image_size)
        self.model.to(self.device).eval()

    def predict_tensor(self, tensor: torch.Tensor, top_k: int = 3) -> Prediction:
        if not 1 <= top_k <= len(self.labels):
            raise ValueError(f"top-k must be between 1 and {len(self.labels)}")
        with torch.inference_mode():
            probabilities = self.model(tensor.unsqueeze(0).to(self.device)).softmax(1)[0]
        values, indices = probabilities.topk(top_k)
        values, indices = values.cpu().tolist(), indices.cpu().tolist()
        top = [{"label": self.labels[index], "confidence": float(value)} for index, value in zip(indices, values)]
        margin = float(values[0] - values[1]) if len(values) > 1 else None
        status = "LOW_CONFIDENCE" if values[0] < self.confidence_threshold else "OK"
        return Prediction(top[0]["label"], float(values[0]), margin, top, status)

    def predict_image(self, image: Image.Image, top_k: int = 3, tta: str = "none") -> Prediction:
        tensors = [self.transform(view) for view in _tta_views(image, tta, self.config.pad_value)]
        with torch.inference_mode():
            probabilities = self.model(torch.stack(tensors).to(self.device)).softmax(1).mean(0)
        values, indices = probabilities.topk(min(top_k, len(self.labels)))
        values, indices = values.cpu().tolist(), indices.cpu().tolist()
        top = [{"label": self.labels[i], "confidence": float(v)} for i, v in zip(indices, values)]
        margin = float(values[0] - values[1]) if len(values) > 1 else None
        status = "LOW_CONFIDENCE" if values[0] < self.confidence_threshold else "OK"
        return Prediction(top[0]["label"], float(values[0]), margin, top, status)

    def predict(self, paths: Sequence[Path], top_k: int = 3, batch_size: int = 32, tta: str = "none") -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch-size must be positive")
        results: list[dict[str, Any] | None] = [None] * len(paths)
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset:offset + batch_size]
            started = time.perf_counter()
            valid = []
            for local_index, path in enumerate(batch_paths):
                try:
                    views = [self.transform(view) for view in _tta_views(
                        load_image_safe(path, self.config.pad_value), tta, self.config.pad_value)]
                    valid.append((local_index, path, torch.stack(views)))
                except Exception as exc:
                    results[offset + local_index] = {
                        "path": str(path), "filename": path.name, "status": "ERROR", "error": str(exc),
                        "prediction": "", "confidence": "", "margin": "", "top_k": [],
                        "device": str(self.device), "mode": "single", "inference_ms": 0.0,
                    }
            if not valid:
                continue
            view_count = valid[0][2].shape[0]
            tensors = torch.stack([item[2] for item in valid])
            with torch.inference_mode():
                probabilities = self.model(tensors.flatten(0, 1).to(self.device)).softmax(1)
                probabilities = probabilities.reshape(len(valid), view_count, -1).mean(1)
            values, indices = probabilities.topk(min(top_k, len(self.labels)), dim=1)
            elapsed_ms = (time.perf_counter() - started) * 1000 / len(valid)
            for (local_index, path, _), scores, indexes in zip(valid, values.cpu().tolist(), indices.cpu().tolist()):
                top = [{"label": self.labels[index], "confidence": float(score)}
                       for index, score in zip(indexes, scores)]
                margin = float(scores[0] - scores[1]) if len(scores) > 1 else None
                status = "LOW_CONFIDENCE" if scores[0] < self.confidence_threshold else "OK"
                results[offset + local_index] = {
                    "path": str(path), "filename": path.name, "predicted_label": top[0]["label"],
                    "prediction": top[0]["label"], "label": top[0]["label"],
                    "confidence": float(scores[0]), "margin": margin, "top_k": top, "status": status,
                    "device": str(self.device), "mode": "single", "inference_ms": elapsed_ms, "error": "",
                }
        return [result for result in results if result is not None]

    def predict_multi(self, path: Path, detector_weights: Path, top_k: int = 3, tta: str = "none",
                      bbox_margin: float = 0.08, detector_confidence: float = 0.25) -> dict[str, Any]:
        image = load_image_safe(path, self.config.pad_value)
        boxes = sort_reading_order(_detector_predictions(image, detector_weights, detector_confidence))
        characters = []
        for box in boxes:
            crop_box = expand_bbox(box.bbox, image.size, bbox_margin)
            prediction = self.predict_image(image.crop(crop_box), top_k, tta)
            characters.append({"bbox": list(crop_box), "detector_confidence": box.confidence,
                               "label": prediction.label, "confidence": prediction.confidence,
                               "margin": prediction.margin, "status": prediction.status,
                               "top_k": prediction.top_k})
        return {"input": str(path), "mode": "multi", "characters": characters,
                "sequence": [item["label"] for item in characters], "device": str(self.device)}


def discover_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {path}")
    return sorted((p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS), key=lambda p: str(p).lower())


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    top = result.get("top_k", [])
    return {"filename": result.get("filename", ""), "path": result.get("path", ""), "status": result.get("status", ""),
            "prediction": result.get("label", result.get("prediction", "")), "confidence": result.get("confidence", ""),
            "margin": result.get("margin", ""), "top2_label": top[1]["label"] if len(top) > 1 else "",
            "top2_confidence": top[1]["confidence"] if len(top) > 1 else "", "top3_label": top[2]["label"] if len(top) > 2 else "",
            "top3_confidence": top[2]["confidence"] if len(top) > 2 else "", "device": result.get("device", ""),
            "mode": result.get("mode", ""), "inference_ms": result.get("inference_ms", ""), "error": result.get("error", "")}


def save_csv(path: Path, results: Iterable[dict[str, Any]]) -> None:
    rows = [_csv_row(result) for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "path", "status", "prediction", "confidence", "margin", "top2_label", "top2_confidence",
              "top3_label", "top3_confidence", "device", "mode", "inference_ms", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def print_single(result: dict[str, Any], predictor: Predictor, input_path: Path, tta: str) -> None:
    if result.get("status") == "ERROR":
        raise RuntimeError(result["error"])
    print("Thai Character Recognition\n")
    print(f"Input       : {input_path}\nMode        : single\nDevice      : {predictor.device}\nArchitecture: {predictor.config.architecture}\nImage size  : {predictor.config.image_size}\nTTA         : {tta}\n")
    print("Prediction\n────────────────────────────────")
    for index, item in enumerate(result["top_k"], 1):
        print(f"{index}. {item['label']}   {item['confidence']:.2%}")
    print(f"\nTop-1 confidence : {result['confidence']:.2%}")
    if result["margin"] is not None:
        print(f"Top1-Top2 margin : {result['margin']:.2%}")
    print(f"Status           : {result['status']}\nInference time   : {result['inference_ms']:.1f} ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_weights, _, _ = discover_defaults()
    parser.add_argument("--input", "--image", dest="input", type=Path, help="Image file or directory")
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--mode", choices=("single", "multi", "auto"), default="single")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--tta", choices=("none", "safe"), default="none")
    parser.add_argument("--output", type=Path, help="CSV/JSON output path; omitted for human-readable output")
    parser.add_argument("--detector-weights", type=Path)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    parser.add_argument("--self-check", action="store_true")
    return parser


def self_check(args: argparse.Namespace) -> int:
    predictor = Predictor(args.weights, args.device, args.config, args.labels)
    tensor = torch.zeros(3, predictor.config.image_size, predictor.config.image_size)
    prediction = predictor.predict_tensor(tensor, min(3, len(predictor.labels)))
    print(json.dumps({"checkpoint": str(predictor.checkpoint), "sha256": predictor.sha256,
                      "architecture": predictor.config.architecture, "image_size": predictor.config.image_size,
                      "classes": len(predictor.labels), "dummy_prediction": prediction.label,
                      "offline": True}, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        return self_check(args)
    if args.input is None:
        raise SystemExit("--input is required unless --self-check is used")
    if not 1 <= args.top_k:
        raise SystemExit("--top-k must be positive")
    if not 0 <= args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if args.mode == "multi" and args.input.is_dir():
        raise SystemExit("Explicit multi mode accepts one image; use folder mode with single/auto")
    predictor = Predictor(args.weights, args.device, args.config, args.labels, args.confidence_threshold)
    if args.mode == "multi":
        if not args.detector_weights:
            raise SystemExit("Detector weights are required for explicit multi mode")
        result = predictor.predict_multi(args.input, args.detector_weights, args.top_k, args.tta, args.bbox_margin, args.detector_confidence)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "auto" and args.input.is_file() and args.detector_weights:
        try:
            result = predictor.predict_multi(args.input, args.detector_weights, args.top_k, args.tta,
                                             args.bbox_margin, args.detector_confidence)
            if result["characters"]:
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            print(f"Auto mode: detector unavailable ({exc}); falling back to single mode.", file=sys.stderr)
    paths = discover_images(args.input)
    if not paths:
        raise SystemExit(f"No supported images found in {args.input}")
    results = predictor.predict(paths, args.top_k, tta=args.tta)
    if args.output:
        output = args.output
        if output.is_dir() or output.suffix == "":
            output.mkdir(parents=True, exist_ok=True)
            save_csv(output / "predictions.csv", results)
        elif output.suffix.lower() == ".csv":
            save_csv(output, results)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2), encoding="utf-8")
    elif len(results) == 1:
        print_single(results[0], predictor, paths[0], args.tta)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Inference error: {exc}", file=sys.stderr)
        raise SystemExit(2)

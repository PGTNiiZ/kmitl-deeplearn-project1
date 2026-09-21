"""Shared notebook/CLI trainer. All model selection uses fixed validation macro F1."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.split import fingerprint, read_rows


@dataclass
class TrainConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 12
    freeze_epochs: int = 2
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    label_smoothing: float = 0.0
    augmentation: bool = False
    class_weights: bool = False
    patience: int = 5
    num_workers: int = 0  # Portable in Jupyter on macOS, Windows and Colab.
    amp: bool = True
    device: str = "auto"
    # Fixed ImageNet statistics for a controlled cross-backbone benchmark.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    pad_value: int = 255


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class ResizePad:
    def __init__(self, size: int, fill: int):
        self.size, self.fill = size, fill

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR,
                            color=(self.fill,) * 3)


def make_transform(config: TrainConfig, training: bool = False):
    operations = [ResizePad(config.image_size, config.pad_value)]
    if training and config.augmentation:
        operations.append(transforms.RandomAffine(
            degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=(config.pad_value,) * 3))
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])


class ManifestDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], mapping: dict, transform):
        self.root, self.rows, self.mapping, self.transform = root, rows, mapping, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.mapping[row["label"]]


class CustomCNN(nn.Module):
    def __init__(self, classes: int, dropout: float):
        super().__init__()
        layers, channels = [], 3
        for width in (32, 64, 128, 256):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width),
                           nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, classes))

    def forward(self, x):
        return self.head(self.features(x))

    def get_classifier(self):
        return self.head


def build_model(config: TrainConfig, classes: int, load_pretrained: bool = True):
    if config.architecture == "custom_cnn":
        return CustomCNN(classes, config.dropout)
    import timm
    return timm.create_model(config.architecture, pretrained=config.pretrained and load_pretrained,
                             num_classes=classes, drop_rate=config.dropout)


def choose_device(name: str):
    name = name.lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA GPU is available to PyTorch")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device.index} does not exist")
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available to PyTorch")
    elif device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, cuda:<index>, or mps")
    return device


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "Apple MPS" if device.type == "mps" else "CPU"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_split(data_dir: Path, split_dir: Path):
    train, val = read_rows(split_dir / "train.csv"), read_rows(split_dir / "val.csv")
    mapping = json.loads((split_dir / "label_to_index.json").read_text())
    meta = json.loads((split_dir / "split_meta.json").read_text())
    if fingerprint([train, val, mapping]) != meta["split_hash"]:
        raise ValueError("Split/mapping hash mismatch. Restore the frozen manifests.")
    if {r["sha256"] for r in train} & {r["sha256"] for r in val}:
        raise ValueError("Duplicate leakage across train and validation")
    for row in train + val:
        path = (data_dir / row["path"]).resolve()
        if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
            raise ValueError(f"Missing/invalid image in manifest: {row['path']}")
    return train, val, mapping, meta


def evaluate(model, loader, device, criterion, classes):
    model.eval()
    actual, predicted, confidence = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).sum().item()
            probs = logits.softmax(1)
            scores, indices = probs.max(1)
            actual.extend(targets.cpu().tolist())
            predicted.extend(indices.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
    report = classification_report(actual, predicted, labels=list(range(classes)),
                                   output_dict=True, zero_division=0)
    metrics = {"val_loss": total_loss / len(actual), "val_accuracy": accuracy_score(actual, predicted),
               "val_macro_f1": report["macro avg"]["f1-score"],
               "val_macro_precision": report["macro avg"]["precision"],
               "val_macro_recall": report["macro avg"]["recall"],
               "minimum_per_class_recall": min(report[str(i)]["recall"] for i in range(classes))}
    return metrics, report, actual, predicted, confidence


def fit(config: TrainConfig, data_dir: Path, split_dir: Path, output_dir: Path,
        phase: str = "benchmark", trial=None, reuse: bool = True) -> dict:
    """Reuse completed identical runs; interrupted runs restart in a new directory.

    Trial pruning never produces an eligible completed run. Each checkpoint carries
    its preprocessing, label mapping and split hash for standalone inference.
    """
    if config.image_size < 32 or config.batch_size < 2 or config.epochs < 1:
        raise ValueError("Use image_size >= 32, batch_size >= 2, epochs >= 1")
    if config.patience < 1 or not 0 <= config.freeze_epochs < config.epochs:
        raise ValueError("patience >= 1 and 0 <= freeze_epochs < epochs are required")
    data_dir, split_dir, output_dir = map(lambda p: Path(p).resolve(), (data_dir, split_dir, output_dir))
    train_rows, val_rows, mapping, meta = load_split(data_dir, split_dir)
    code_hash = fingerprint({name: (Path(__file__).parent / name).read_text()
                             for name in ("train.py", "split.py")})
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "timm", "numpy", "scikit-learn")}
    run_key = fingerprint([asdict(config), meta["split_hash"], phase, code_hash, versions])
    output_dir.mkdir(parents=True, exist_ok=True)
    if reuse and trial is None:
        for receipt_path in sorted(output_dir.glob("*/run_receipt.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("run_key") == run_key and receipt.get("status") == "complete" and all(
                    Path(receipt[k]).is_file() for k in ("checkpoint_path", "metrics_path", "config_path")):
                print("Reusing completed run:", receipt["run_id"])
                return receipt
    seed_everything(config.seed)
    device = choose_device(config.device)
    print(f"Training device: {device} ({device_name(device)})", flush=True)
    run_id = f"{phase}-{config.architecture}-{uuid.uuid4().hex[:10]}"
    run_dir = output_dir / run_id
    run_dir.mkdir()
    save_json(run_dir / "config.json", asdict(config))
    save_json(run_dir / "environment.json", {**versions, "device": str(device),
                                               "device_name": device_name(device), "code_hash": code_hash})
    receipt = {"run_id": run_id, "run_key": run_key, "status": "running", "phase": phase,
               "split_hash": meta["split_hash"], "seed": config.seed, "backbone": config.architecture,
               "max_epochs": config.epochs, "config_path": str(run_dir / "config.json"),
               "checkpoint_path": str(run_dir / "best.pt"), "metrics_path": str(run_dir / "metrics.json")}
    save_json(run_dir / "run_receipt.json", receipt)
    model = optimizer = None
    started = time.perf_counter()
    try:
        train_set = ManifestDataset(data_dir, train_rows, mapping, make_transform(config, True))
        val_set = ManifestDataset(data_dir, val_rows, mapping, make_transform(config))
        # Drop a singleton tail: BatchNorm in small backbones can reject it.
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=device.type == "cuda",
                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config.seed),
                                  drop_last=len(train_set) % config.batch_size == 1)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=config.num_workers)
        model = build_model(config, len(mapping)).to(device)
        weights = None
        if config.class_weights:
            counts = np.bincount([mapping[r["label"]] for r in train_rows], minlength=len(mapping))
            values = np.minimum(np.sqrt(counts.max() / counts), 3.0)
            weights = torch.tensor(values / values.mean(), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing, reduction="none")
        # Validation loss is always plain CE so loss remains comparable across ablations.
        val_criterion = nn.CrossEntropyLoss(reduction="none")
        use_amp = config.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        freeze_epochs = config.freeze_epochs if config.pretrained and config.architecture != "custom_cnn" else 0
        history, best, stale = [], (-1.0, -1.0), 0
        for epoch in range(config.epochs):
            frozen = epoch < freeze_epochs
            if epoch in {0, freeze_epochs}:
                for parameter in model.parameters():
                    parameter.requires_grad_(not frozen)
                for parameter in model.get_classifier().parameters():
                    parameter.requires_grad_(True)
                lr = config.head_lr if frozen or config.architecture == "custom_cnn" else config.finetune_lr
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                              lr=lr, weight_decay=config.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=2)
                stale = 0
            model.train()
            if frozen:
                model.eval()  # Freeze running BatchNorm statistics as well as gradients.
                model.get_classifier().train()
            total_loss = correct = seen = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(targets)
                correct += (logits.argmax(1) == targets).sum().item()
                seen += len(targets)
            metrics, report, actual, predicted, confidence = evaluate(model, val_loader, device, val_criterion, len(mapping))
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError("Non-finite validation metrics")
            record = {"epoch": epoch + 1, "stage": "head" if frozen else "finetune",
                      "train_loss": total_loss / seen, "train_accuracy": correct / seen,
                      "lr": optimizer.param_groups[0]["lr"], **metrics}
            history.append(record)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            score = metrics["val_macro_f1"], metrics["val_accuracy"]
            if score > best:
                best, stale = score, 0
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "config": asdict(config), "label_to_index": mapping,
                            "split_hash": meta["split_hash"], "best_epoch": epoch + 1}, run_dir / "best.pt")
                save_json(run_dir / "metrics.json", {**metrics, "classification_report": report})
                names = list(mapping)
                pd.DataFrame(confusion_matrix(actual, predicted, labels=list(range(len(mapping)))),
                             index=names, columns=names).to_csv(run_dir / "confusion_matrix.csv")
                pd.DataFrame({"path": [r["path"] for r in val_rows], "true_label": [names[i] for i in actual],
                              "predicted_label": [names[i] for i in predicted], "confidence": confidence}).to_csv(
                                  run_dir / "predictions.csv", index=False)
                receipt.update(metrics, best_epoch=epoch + 1, train_accuracy=record["train_accuracy"])
            else:
                stale += 1
            scheduler.step(metrics["val_macro_f1"])
            print(f"{run_id} | {epoch + 1}/{config.epochs} | loss={record['train_loss']:.4f} "
                  f"val_acc={metrics['val_accuracy']:.4f} macro_f1={metrics['val_macro_f1']:.4f}", flush=True)
            if trial is not None:
                trial.report(best[0], epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            if not frozen and stale >= config.patience:
                break
        receipt.update(status="complete", epochs_completed=len(history),
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       training_seconds=time.perf_counter() - started)
        save_json(run_dir / "run_receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="pruned" if type(error).__name__ == "TrialPruned" else "failed",
                       error=f"{type(error).__name__}: {error}")
        save_json(run_dir / "run_receipt.json", receipt)
        raise
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Flat TrainConfig JSON exported by the search notebook")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/clean_32x32"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    args = parser.parse_args()
    from src.paths import DEFAULT_IMAGE_EXTENSIONS, get_data_dir, prepare_image_directory
    config = TrainConfig(**json.loads(args.config.read_text()))
    data_dir = prepare_image_directory(args.data_dir, DEFAULT_IMAGE_EXTENSIONS) if args.data_dir else get_data_dir()
    receipt = fit(config, data_dir, args.split_dir, args.output_dir)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

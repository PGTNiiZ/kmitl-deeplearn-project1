"""Shared runner for the three compact team model-screening jobs."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.audit import DEFAULT_EXTENSIONS, audit_dataset
from src.paths import REPOSITORY_ROOT, prepare_image_directory
from src.split import fingerprint, prepare_split
from src.train import TrainConfig, build_model, choose_device, fit, load_split, save_json


DRIVE_FILE_ID = "1BnPkvJJlE7QDZ0sgEDujS23Pr8Cj3IKq"
DRIVE_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing"


def parser(group_id: int) -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=f"Run model-screening group {group_id}.")
    command.add_argument("--data-dir", type=Path, help="Extracted dataset or ZIP; downloads the shared file when omitted.")
    command.add_argument("--split-dir", type=Path,
                         default=REPOSITORY_ROOT / f"data/splits/{DRIVE_FILE_ID}_source_v3")
    command.add_argument("--output-dir", type=Path,
                         default=REPOSITORY_ROOT / f"results/team_model_benchmark/group_{group_id:02d}")
    command.add_argument("--device", default="auto")
    command.add_argument("--epochs", type=int, default=6)
    command.add_argument("--image-size", type=int, default=128)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--num-workers", type=int, default=4)
    command.add_argument("--expected-split-hash", default="")
    command.add_argument("--smoke", action="store_true", help="Run one model on generated data without downloading.")
    return command


def smoke_data(root: Path) -> Path:
    data_dir = root / "images"
    for label in range(3):
        folder = data_dir / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(10):
            path = folder / f"{index:03d}.png"
            if not path.exists():
                Image.new("RGB", (32, 32), (label * 100, index * 20, 40)).save(path)
    return data_dir


def real_data(requested: Path | None) -> Path:
    if requested:
        return prepare_image_directory(requested, DEFAULT_EXTENSIONS)
    cache = REPOSITORY_ROOT / "data/raw" / DRIVE_FILE_ID
    target = cache / "clean_32x32"
    try:
        data_dir = prepare_image_directory(target, DEFAULT_EXTENSIONS)
        if data_dir.is_dir() and any(data_dir.rglob("*.png")):
            return data_dir
    except (FileNotFoundError, ValueError):
        pass
    subprocess.run(
        [sys.executable, "-m", "src.download_data", "--drive-url", DRIVE_URL,
         "--output-dir", str(cache), "--force"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    return prepare_image_directory(target, DEFAULT_EXTENSIONS)


def audit_and_split(data_dir: Path, split_dir: Path, output_root: Path,
                    expected_classes: int) -> tuple[list[dict], list[dict], dict, dict]:
    audit_dir = output_root / "audit"
    report_path = audit_dir / "audit_report.json"
    reuse_audit = False
    if report_path.is_file() and (audit_dir / "image_manifest.csv").is_file():
        report = json.loads(report_path.read_text())
        reuse_audit = Path(report.get("data_dir", "")).resolve() == data_dir.resolve()
    if not reuse_audit:
        report = audit_dataset(data_dir, audit_dir, 1, DEFAULT_EXTENSIONS)
    if report["corrupt_images"] or report["label_errors"]:
        raise ValueError("Dataset contains corrupt images or label errors; inspect the audit output.")
    prepare_split(data_dir, audit_dir, split_dir, seed=42, val_fraction=0.2,
                  expected_classes=expected_classes)
    return load_split(data_dir, split_dir)


def resolve_architecture(name: str) -> str:
    import timm
    if name == "custom_cnn":
        return name
    if not timm.is_model(name):
        raise ValueError(f"{name} is unavailable in the installed timm version")
    pretrained = timm.models.get_pretrained_cfg(name)
    if pretrained is None or not pretrained.has_weights:
        raise ValueError(f"{name} has no registered pretrained weights")
    return name if "." in name or not pretrained.tag else f"{name}.{pretrained.tag}"


def preflight(config: TrainConfig, classes: int) -> int:
    device = choose_device(config.device)
    model = build_model(config, classes, load_pretrained=False).to(device)
    try:
        model.train()
        sample = torch.randn(2, 3, config.image_size, config.image_size, device=device)
        target = torch.tensor([0, classes - 1], device=device)
        logits = model(sample)
        if logits.shape != (2, classes):
            raise ValueError(f"Expected [2,{classes}] logits, got {tuple(logits.shape)}")
        torch.nn.functional.cross_entropy(logits, target).backward()
        return sum(parameter.numel() for parameter in model.parameters())
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


def run_group(group_id: int, candidates: list[dict]) -> int:
    args = parser(group_id).parse_args()
    if min(args.epochs, args.image_size, args.batch_size) < 1 or args.num_workers < 0:
        raise ValueError("epochs/image-size/batch-size must be positive and num-workers cannot be negative")

    mode = "smoke" if args.smoke else "quick"
    output_root = args.output_dir.resolve() / mode
    if args.smoke:
        data_dir = smoke_data(output_root / "smoke_fixture")
        split_dir = output_root / "smoke_fixture/split"
        expected_classes = 3
        selected = candidates[:1]
        base = TrainConfig(seed=42, image_size=32, batch_size=8, epochs=2, freeze_epochs=0,
                           patience=1, pretrained=False, device=args.device, num_workers=0)
    else:
        data_dir = real_data(args.data_dir)
        split_dir = args.split_dir.resolve()
        expected_classes = 72
        selected = candidates
        base = TrainConfig(seed=42, image_size=args.image_size, batch_size=args.batch_size,
                           epochs=args.epochs, freeze_epochs=2, patience=3, pretrained=True,
                           device=args.device, num_workers=args.num_workers)

    train_rows, val_rows, mapping, split_meta = audit_and_split(
        data_dir, split_dir, output_root.parent / "shared", expected_classes)
    if args.expected_split_hash and split_meta["split_hash"] != args.expected_split_hash:
        raise ValueError(f"Split hash mismatch: {split_meta['split_hash']}")

    protocol = {
        "group_id": group_id,
        "mode": mode,
        "config": asdict(base),
        "split_hash": split_meta["split_hash"],
        "models": [candidate["model"] for candidate in selected],
    }
    protocol_id = fingerprint(protocol)[:16]
    experiment_dir = output_root / split_meta["split_hash"][:12] / protocol_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(experiment_dir / "protocol.json", protocol)

    statuses, receipts = [], []
    for candidate in selected:
        status = {"group_id": group_id, **candidate, "split_hash": split_meta["split_hash"],
                  "protocol_id": protocol_id, "status": "running"}
        try:
            architecture = resolve_architecture(candidate["model"])
            config = replace(base, architecture=architecture,
                             pretrained=base.pretrained and architecture != "custom_cnn")
            status["architecture"] = architecture
            status["parameter_count_preflight"] = preflight(config, len(mapping))
            receipt = fit(config, data_dir, split_dir, experiment_dir / "runs",
                          phase=f"group_{group_id:02d}_screening")
            receipt.update(candidate, group_id=group_id, protocol_id=protocol_id)
            receipts.append(receipt)
            status.update(status="complete", run_id=receipt["run_id"])
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            print("FAILED", candidate["family"], status["error"], flush=True)
        statuses.append(status)
        pd.DataFrame(statuses).to_csv(experiment_dir / f"group_{group_id:02d}_run_status.csv", index=False)

    summary = pd.DataFrame(receipts)
    if not summary.empty:
        summary = summary.sort_values(["val_macro_f1", "val_accuracy"], ascending=False)
    summary.to_csv(experiment_dir / f"group_{group_id:02d}_model_summary.csv", index=False)

    histories = []
    for receipt in receipts:
        path = Path(receipt["checkpoint_path"]).parent / "history.csv"
        if path.is_file():
            history = pd.read_csv(path)
            history.insert(0, "model", receipt["model"])
            history.insert(0, "family", receipt["family"])
            histories.append(history)
    pd.concat(histories, ignore_index=True).to_csv(
        experiment_dir / f"group_{group_id:02d}_epoch_history.csv", index=False) if histories else None

    manifest = {
        **protocol,
        "protocol_id": protocol_id,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "complete_runs": sum(row["status"] == "complete" for row in statuses),
        "failed_runs": sum(row["status"] != "complete" for row in statuses),
        "experiment_dir": str(experiment_dir),
    }
    save_json(experiment_dir / f"group_{group_id:02d}_manifest.json", manifest)
    print(summary[[column for column in
                   ("family", "model", "val_macro_f1", "val_accuracy", "parameter_count", "training_seconds")
                   if column in summary]].to_string(index=False) if not summary.empty else "No completed runs")
    print("Results:", experiment_dir)
    return 0 if receipts else 1

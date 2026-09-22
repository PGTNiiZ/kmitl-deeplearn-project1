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
                         default=REPOSITORY_ROOT / f"results/team_model_benchmark/worker_{group_id:02d}")
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
        data_dir = prepare_image_directory(requested, DEFAULT_EXTENSIONS)
        print(f"Dataset ready: {data_dir}", flush=True)
        return data_dir
    cache = REPOSITORY_ROOT / "data/raw" / DRIVE_FILE_ID
    target = cache / "clean_32x32"
    try:
        data_dir = prepare_image_directory(target, DEFAULT_EXTENSIONS)
        if data_dir.is_dir() and any(data_dir.rglob("*.png")):
            print(f"Dataset ready (cached): {data_dir}", flush=True)
            return data_dir
    except (FileNotFoundError, ValueError):
        pass
    subprocess.run(
        [sys.executable, "-m", "src.download_data", "--drive-url", DRIVE_URL,
         "--output-dir", str(cache), "--force"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    print(f"Extracting dataset into {cache}...", flush=True)
    data_dir = prepare_image_directory(target, DEFAULT_EXTENSIONS)
    print(f"Dataset ready: {data_dir}", flush=True)
    return data_dir


def audit_and_split(data_dir: Path, split_dir: Path, output_root: Path,
                    expected_classes: int) -> tuple[list[dict], list[dict], dict, dict]:
    audit_dir = output_root / "audit"
    report_path = audit_dir / "audit_report.json"
    reuse_audit = False
    if report_path.is_file() and (audit_dir / "image_manifest.csv").is_file():
        report = json.loads(report_path.read_text())
        reuse_audit = Path(report.get("data_dir", "")).resolve() == data_dir.resolve()
    if not reuse_audit:
        print("Stage 2/4: auditing dataset and hashing files", flush=True)
        report = audit_dataset(data_dir, audit_dir, 1, DEFAULT_EXTENSIONS)
    else:
        print(f"Stage 2/4: reusing audit from {audit_dir}", flush=True)
    if report["corrupt_images"] or report["label_errors"]:
        raise ValueError("Dataset contains corrupt images or label errors; inspect the audit output.")
    print("Stage 3/4: preparing leakage-safe split", flush=True)
    prepare_split(data_dir, audit_dir, split_dir, seed=42, val_fraction=0.2,
                  expected_classes=expected_classes)
    loaded = load_split(data_dir, split_dir)
    print(f"Split ready: {len(loaded[0]):,} train / {len(loaded[1]):,} validation", flush=True)
    return loaded


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

    worker = f"worker_{group_id:02d}"
    mode = "smoke" if args.smoke else "quick"
    output_root = args.output_dir.resolve() / mode
    print(f"\n=== {worker}: complete model benchmark ===", flush=True)
    print("Stage 1/4: preparing dataset", flush=True)
    print(f"Settings: epochs={args.epochs}, image_size={args.image_size}, "
          f"batch_size={args.batch_size}, num_workers={args.num_workers}, device={args.device}",
          flush=True)
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

    # Real workers share one expensive audit; smoke runs stay isolated.
    shared_root = (output_root / "smoke_fixture/shared" if args.smoke else
                   REPOSITORY_ROOT / "results/team_model_benchmark/worker_01/shared")
    train_rows, val_rows, mapping, split_meta = audit_and_split(
        data_dir, split_dir, shared_root, expected_classes)
    if args.expected_split_hash and split_meta["split_hash"] != args.expected_split_hash:
        raise ValueError(f"Split hash mismatch: {split_meta['split_hash']}")

    protocol = {
        "worker_id": group_id,
        "mode": mode,
        "config": asdict(base),
        "split_hash": split_meta["split_hash"],
        "assigned_models": selected,
    }
    protocol_id = fingerprint(protocol)[:16]
    experiment_dir = output_root / split_meta["split_hash"][:12] / protocol_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(experiment_dir / "protocol.json", protocol)

    statuses, receipts = [], []
    print(f"Stage 4/4: training {len(selected)} model(s) and exporting results", flush=True)
    for model_number, candidate in enumerate(selected, start=1):
        print(f"\nModel {model_number}/{len(selected)}: "
              f"{candidate['family']} ({candidate['model']})", flush=True)
        status = {"worker_id": group_id, **candidate, "seed": base.seed,
                  "epochs_requested": args.epochs, "split_hash": split_meta["split_hash"],
                  "protocol_id": protocol_id, "status": "running"}
        statuses.append(status)
        status_path = experiment_dir / f"{worker}_run_status.csv"
        pd.DataFrame(statuses).to_csv(status_path, index=False)
        try:
            architecture = resolve_architecture(candidate["model"])
            config = replace(base, architecture=architecture,
                             pretrained=base.pretrained and architecture != "custom_cnn")
            status["architecture"] = architecture
            status["parameter_count_preflight"] = preflight(config, len(mapping))
            receipt = fit(config, data_dir, split_dir, experiment_dir / "runs",
                          phase=f"{worker}_family_screening")
            receipt.update(candidate, worker_id=group_id, protocol_id=protocol_id,
                           epochs_requested=args.epochs)
            receipts.append(receipt)
            status.update(status="complete", run_id=receipt["run_id"],
                          checkpoint_path=receipt["checkpoint_path"])
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            print("FAILED", candidate["family"], status["error"], flush=True)
        pd.DataFrame(statuses).to_csv(status_path, index=False)

    summary = pd.DataFrame(receipts)
    if not summary.empty:
        summary = summary.sort_values(["val_macro_f1", "val_accuracy"], ascending=False)
    summary_path = experiment_dir / f"{worker}_model_summary.csv"
    status_path = experiment_dir / f"{worker}_run_status.csv"
    history_path = experiment_dir / f"{worker}_epoch_history.csv"
    excel_path = experiment_dir / f"{worker}_training_results.xlsx"
    summary.to_csv(summary_path, index=False)

    histories = []
    for receipt in receipts:
        path = Path(receipt["checkpoint_path"]).parent / "history.csv"
        if path.is_file():
            history = pd.read_csv(path)
            history.insert(0, "run_id", receipt["run_id"])
            history.insert(0, "backbone", receipt["backbone"])
            history.insert(0, "model", receipt["model"])
            history.insert(0, "family", receipt["family"])
            history.insert(0, "worker_id", group_id)
            history["seed"] = receipt["seed"]
            history["split_hash"] = receipt["split_hash"]
            history["protocol_id"] = protocol_id
            histories.append(history)
    history_table = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    history_table.to_csv(history_path, index=False)

    protocol_table = pd.DataFrame(
        {"key": protocol.keys(),
         "value": [json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                   for value in protocol.values()]})
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([{"worker_id": group_id, **candidate} for candidate in selected]).to_excel(
            writer, sheet_name="Assignment", index=False)
        pd.DataFrame(statuses).to_excel(writer, sheet_name="RunStatus", index=False)
        summary.to_excel(writer, sheet_name="ModelSummary", index=False)
        history_table.to_excel(writer, sheet_name="EpochHistory", index=False)
        protocol_table.to_excel(writer, sheet_name="Protocol", index=False)

    manifest = {
        **protocol,
        "protocol_id": protocol_id,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "complete_runs": sum(row["status"] == "complete" for row in statuses),
        "failed_or_unavailable": sum(row["status"] != "complete" for row in statuses),
        "experiment_dir": str(experiment_dir),
        "files": {
            "summary_csv": str(summary_path),
            "epoch_history_csv": str(history_path),
            "status_csv": str(status_path),
            "excel": str(excel_path),
        },
    }
    manifest_path = experiment_dir / f"{worker}_manifest.json"
    save_json(manifest_path, manifest)
    print(summary[[column for column in
                   ("family", "model", "val_macro_f1", "val_accuracy", "parameter_count", "training_seconds")
                   if column in summary]].to_string(index=False) if not summary.empty else "No completed runs")
    print(f"\n=== {worker} finished ===", flush=True)
    print(f"Results folder: {experiment_dir}", flush=True)
    for path in (summary_path, history_path, status_path, manifest_path, excel_path):
        print(f"  - {path.name}", flush=True)
    return 0 if receipts else 1

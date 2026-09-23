"""Validate and run the frozen final-training protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from src.split import fingerprint
from src.train import TrainConfig, choose_device, device_name, fit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "configs/experiments/densenet121_final_protocol.json"
TRAINING_CODE_FILES = ("train.py", "split.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_files(protocol_path: Path) -> tuple[dict, TrainConfig, Path, Path, Path]:
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked":
        raise ValueError("Protocol status must be 'locked'")

    paths = protocol["paths"]
    data_dir = (REPOSITORY_ROOT / paths["data_dir"]).resolve()
    split_dir = (REPOSITORY_ROOT / paths["split_dir"]).resolve()
    output_dir = (REPOSITORY_ROOT / paths["output_dir"]).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Canonical dataset missing: {data_dir}")

    split = protocol["split"]
    for name, expected in split["files"].items():
        path = split_dir / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen split file changed: {name}: {actual} != {expected}")
    meta = json.loads((split_dir / "split_meta.json").read_text(encoding="utf-8"))
    if meta.get("split_hash") != split["split_hash"]:
        raise ValueError(f"Canonical split hash mismatch: {meta.get('split_hash')} != {split['split_hash']}")

    code_hash = fingerprint({name: (REPOSITORY_ROOT / "src" / name).read_text()
                             for name in TRAINING_CODE_FILES})
    expected_code_hash = protocol["environment"]["training_code_hash"]
    if code_hash != expected_code_hash:
        raise ValueError(f"Training code changed: {code_hash} != {expected_code_hash}")
    return protocol, TrainConfig(**protocol["config"]), data_dir, split_dir, output_dir


def validate_environment(protocol: dict, config: TrainConfig) -> None:
    expected = protocol["environment"]
    actual_packages = {name: importlib.metadata.version(name) for name in expected["packages"]}
    if actual_packages != expected["packages"]:
        raise ValueError(f"Package environment mismatch: {actual_packages} != {expected['packages']}")
    device = choose_device(config.device)
    if device.type != expected["device_type"] or device_name(device) != expected["device_name"]:
        raise ValueError(
            f"Training device mismatch: {device.type}/{device_name(device)} != "
            f"{expected['device_type']}/{expected['device_name']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--check-files-only", action="store_true",
                        help="Verify immutable files/code without starting training or requiring the target GPU")
    args = parser.parse_args()

    protocol, config, data_dir, split_dir, output_dir = validate_files(args.protocol)
    protocol_hash = _sha256(args.protocol.resolve())
    print(f"Protocol OK: {protocol['protocol_id']} ({protocol_hash})")
    if args.check_files_only:
        return 0

    validate_environment(protocol, config)
    receipt = fit(config, data_dir, split_dir, output_dir, phase=protocol["protocol_id"])
    run_dir = Path(receipt["checkpoint_path"]).parent
    shutil.copy2(args.protocol.resolve(), run_dir / "canonical_protocol.json")
    (run_dir / "canonical_protocol.sha256").write_text(protocol_hash + "\n", encoding="utf-8")
    checkpoint_hash = _sha256(Path(receipt["checkpoint_path"]))
    (run_dir / "checkpoint.sha256").write_text(checkpoint_hash + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"Inference: python -m src.inference --weights {receipt['checkpoint_path']} --input <image-or-folder>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

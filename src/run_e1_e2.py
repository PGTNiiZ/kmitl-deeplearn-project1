"""Run a fresh controlled DenseNet121 pair, then stop for E5 error review.

Usage: python -u -m src.run_e1_e2
Uses the locked canonical dataset/environment; never trains on external data.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from src.locked_train import DEFAULT_PROTOCOL, REPOSITORY_ROOT, _sha256, validate_environment, validate_files

DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/final/densenet121_e1_e2"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(protocol_path=DEFAULT_PROTOCOL, output_dir=DEFAULT_OUTPUT, *, check_only=False,
        verify_content_hashes=False):
    from src.train import TrainConfig, fit
    from src.split import read_rows
    from src.visualize import validation_gallery
    import pandas as pd

    protocol_path, output_dir = Path(protocol_path).resolve(), Path(output_dir).resolve()
    print("Checking canonical dataset, split and training code...", flush=True)
    protocol, config_data, data_dir, split_dir, _ = validate_files(
        protocol_path, verify_content_hashes=verify_content_hashes)
    base = TrainConfig(**config_data)
    if base.architecture != "densenet121.ra_in1k" or base.augmentation:
        raise ValueError("E1 requires DenseNet121 with augmentation=false")
    validate_environment(protocol, base)
    print("Dataset/code/environment PASS", flush=True)
    if check_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / "RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError:
        raise RuntimeError(f"Another run may be active. Check {lock}; remove it only after confirming the process stopped.") from None

    state = {"status": "running", "completed": []}
    # Keep a separate namespace from historical E1 and refuse changed protocols.
    contract = {"protocol": protocol, "runner_sha256": _sha256(Path(__file__)),
                "experiments": {"E1_NEW": {"augmentation": False}, "E2": {"augmentation": True}},
                "selection": ["val_macro_f1", "val_accuracy"],
                "next_step": "human_validation_error_review_before_E5"}
    snapshot = output_dir / "pair_protocol.json"
    try:
        if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != contract:
            raise ValueError("Pair protocol/runner changed; use a new --output-dir for a new comparison")
        write_json(snapshot, contract)
        write_json(output_dir / "status.json", state)
        receipts = []
        for name, augmentation in (("E1_NEW", False), ("E2", True)):
            state["stage"] = name
            write_json(output_dir / "status.json", state)
            print(f"\n=== {name}: augmentation={augmentation} ===", flush=True)
            # Recheck environment/code before E2, without scanning 431k files twice.
            from src.split import fingerprint
            from src.locked_train import TRAINING_CODE_FILES
            current_code = fingerprint({file: (REPOSITORY_ROOT / "src" / file).read_text()
                                        for file in TRAINING_CODE_FILES})
            if current_code != protocol["environment"]["training_code_hash"]:
                raise ValueError("Training code changed during the pair")
            validate_environment(protocol, base)
            receipt = fit(replace(base, augmentation=augmentation), data_dir, split_dir,
                          output_dir / "runs" / name, phase="controlled_e1_e2")
            if receipt["status"] != "complete":
                raise ValueError(f"{name} did not complete")
            run_dir = Path(receipt["checkpoint_path"]).parent
            for filename in ("predictions.csv", "confusion_matrix.csv", "history.csv"):
                if not (run_dir / filename).is_file():
                    raise FileNotFoundError(f"Missing completed-run artifact: {run_dir / filename}")
            receipt = {**receipt, "experiment": name, "family": name,
                       "checkpoint_sha256": _sha256(Path(receipt["checkpoint_path"]))}
            receipts.append(receipt)
            write_json(output_dir / f"{name}.json", receipt)
            state["completed"].append(name)
            write_json(output_dir / "status.json", state)

        state["stage"] = "validation_error_analysis"
        write_json(output_dir / "status.json", state)
        ranked = sorted(receipts, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]), reverse=True)
        pd.DataFrame(ranked).to_csv(output_dir / "comparison.csv", index=False)
        validation_gallery(receipts, data_dir, read_rows(split_dir / "val.csv"),
                           output_dir / "error_analysis", seed=base.seed, sample_count=24)
        winner = ranked[0]
        write_json(output_dir / "best_e1_e2_candidate.json", winner)
        (output_dir / "REVIEW_NEXT.md").write_text(
            "# E1/E2 complete — review before E5\n\n"
            f"Best validation candidate: **{winner['experiment']}**\n\n"
            f"Macro F1: {winner['val_macro_f1']:.6f}; accuracy: {winner['val_accuracy']:.6f}.\n\n"
            "Ranking: macro F1, then accuracy; exact ties retain E1 NEW.\n\n"
            "Open [comparison.csv](comparison.csv) and [actual error images](error_analysis/index.html).\n"
            "Each training run also contains history.csv, metrics.json, predictions.csv and confusion_matrix.csv.\n\n"
            "Review confusion pairs and actual wrong images. If E5 is warranted, define its policy once before training. "
            "Otherwise select the best E1/E2. This candidate is NOT a frozen final model.\n\n"
            "Next: select final, freeze artifacts/inference settings, smoke test, then evaluate a label-compatible "
            "external dataset. No external data was used to select this candidate.\n\n"
            "Completed runs are reused on rerun. Interrupted training restarts from epoch 1; "
            "optimizer/scheduler resume is not implemented. Both runs share the epoch ceiling and early-stopping "
            "policy, so actual epoch counts can differ.\n", encoding="utf-8")
        state.update(status="awaiting_e5_review", stage="done", best_candidate=winner["experiment"])
        write_json(output_dir / "status.json", state)
        print(f"\nE1/E2 finished. Review: {output_dir / 'REVIEW_NEXT.md'}", flush=True)
        return ranked
    except BaseException as error:
        # A contract rejection must not overwrite a prior comparison's status.
        if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) == contract:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(output_dir / "status.json", state)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true", help="Check dataset/code/environment without training")
    parser.add_argument("--verify-content-hashes", action="store_true", help="Also hash every canonical image")
    args = parser.parse_args()
    run(args.protocol, args.output_dir, check_only=args.check_only,
        verify_content_hashes=args.verify_content_hashes)


if __name__ == "__main__":
    main()

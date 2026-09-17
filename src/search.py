"""Controlled architecture benchmark, ablations, persistent HPO and confirmation."""
from __future__ import annotations

import json
import importlib.metadata
from dataclasses import asdict, replace
from pathlib import Path

import optuna
import pandas as pd

from src.split import fingerprint
from src.train import TrainConfig, fit, save_json


SEARCH_SPACE = {"head_lr": [1e-4, 3e-3], "finetune_lr": [1e-5, 1e-4],
                "weight_decay": [1e-6, 1e-3], "dropout": [0.1, 0.5]}


def rank_runs(receipts: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(receipts).sort_values(["val_macro_f1", "val_accuracy"], ascending=False).reset_index(drop=True)


def config_from_receipt(receipt: dict) -> TrainConfig:
    return TrainConfig(**json.loads(Path(receipt["config_path"]).read_text()))


def run_benchmark(base: TrainConfig, architectures: list[str], data_dir: Path,
                  split_dir: Path, output_dir: Path) -> pd.DataFrame:
    receipts = []
    for name in dict.fromkeys(["custom_cnn", *architectures]):
        config = replace(base, architecture=name, pretrained=False if name == "custom_cnn" else base.pretrained)
        receipts.append(fit(config, data_dir, split_dir, output_dir / "runs", phase="benchmark"))
    table = rank_runs(receipts)
    table.to_csv(output_dir / "backbone_screening.csv", index=False)
    return table


def run_ablations(base: TrainConfig, data_dir: Path, split_dir: Path, output_dir: Path):
    # Each row changes just one factor from the selected policy at that step.
    control = fit(base, data_dir, split_dir, output_dir / "runs", phase="benchmark")
    augmented = fit(replace(base, augmentation=True), data_dir, split_dir, output_dir / "runs", phase="ablation")
    selected = rank_runs([control, augmented]).iloc[0].to_dict()
    weighted_config = replace(config_from_receipt(selected), class_weights=True)
    weighted = fit(weighted_config, data_dir, split_dir, output_dir / "runs", phase="ablation")
    table = rank_runs([control, augmented, weighted])
    table.to_csv(output_dir / "ablation_screening.csv", index=False)
    return table


def tune(base: TrainConfig, data_dir: Path, split_dir: Path, output_dir: Path,
         n_trials: int = 12):
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    meta = json.loads((split_dir / "split_meta.json").read_text())
    # Changing the split, code, search space or training recipe creates a new study.
    space = dict(SEARCH_SPACE)
    if base.architecture == "custom_cnn":
        space.pop("finetune_lr")
    elif not base.pretrained or base.freeze_epochs == 0:
        space.pop("head_lr")
    contract = {"config": asdict(base), "split_hash": meta["split_hash"], "space": space,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "torchvision", "timm", "optuna", "numpy", "scikit-learn")},
                "code": fingerprint([(Path(__file__).parent / name).read_text()
                                     for name in ("search.py", "train.py", "split.py")])}
    study_name = "thai-search-" + fingerprint(contract)[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=study_name, direction="maximize",
                                storage="sqlite:///" + str((output_dir / "optuna.sqlite3").resolve()),
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=base.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2))
    study.set_user_attr("contract", contract)

    def objective(trial):
        params = {name: trial.suggest_float(name, *bounds, log=name != "dropout")
                  for name, bounds in space.items()}
        receipt = fit(replace(base, **params), data_dir, split_dir, output_dir / "runs",
                      phase="hpo", trial=trial, reuse=False)
        trial.set_user_attr("receipt", receipt)
        return receipt["val_macro_f1"]

    # n_trials is a total finished-trial target, not an extra budget on every Run All.
    finished = sum(t.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
                   for t in study.trials)
    if finished < n_trials:
        try:
            study.optimize(objective, n_trials=n_trials - finished, n_jobs=1, gc_after_trial=True)
        finally:
            study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    else:
        study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed trials; inspect failures before selecting a model")
    return study


def confirm(base: TrainConfig, study, data_dir: Path, split_dir: Path, output_dir: Path,
            epochs: int = 30, seeds: tuple = (42, 123), top_k: int = 2):
    if not seeds or top_k < 1:
        raise ValueError("Confirmation needs at least one seed and one HPO candidate")
    completed = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                       key=lambda t: (t.value, t.user_attrs["receipt"]["val_accuracy"]), reverse=True)
    candidates = [("untuned_control", base)] + [(f"trial_{t.number}", replace(base, **t.params))
                                               for t in completed[:top_k]]
    receipts = []
    for candidate, config in candidates:
        for seed in dict.fromkeys(seeds):
            receipt = fit(replace(config, epochs=epochs, seed=seed), data_dir, split_dir,
                          output_dir / "runs", phase="confirmation")
            receipts.append({**receipt, "candidate": candidate})
    per_run = rank_runs(receipts)
    per_run.to_csv(output_dir / "confirmation_runs.csv", index=False)
    summary = per_run.groupby("candidate", sort=False).agg(
        mean_macro_f1=("val_macro_f1", "mean"), std_macro_f1=("val_macro_f1", "std"),
        mean_accuracy=("val_accuracy", "mean"), seeds=("seed", "nunique"),
        parameter_count=("parameter_count", "first"), training_seconds=("training_seconds", "sum"))
    summary = summary.sort_values(["mean_macro_f1", "mean_accuracy"], ascending=False).reset_index()
    summary.to_csv(output_dir / "leaderboard.csv", index=False)
    winner = summary.iloc[0]["candidate"]
    # Export the first requested seed of the best mean-scoring configuration;
    # avoid choosing a lucky seed after examining validation scores.
    selected = per_run[(per_run.candidate == winner) & (per_run.seed == seeds[0])].iloc[0].to_dict()
    save_json(output_dir / "best_config.json", asdict(config_from_receipt(selected)))
    save_json(output_dir / "best_run.json", selected)
    return summary, selected

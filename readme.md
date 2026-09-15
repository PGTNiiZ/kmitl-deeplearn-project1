# Thai Character Recognition — Project Handoff

CNN-based recognition of 72 Thai character and digit classes using transfer
learning, semantically safe augmentation, and controlled experiments.

## Current phase

This repository is intentionally in the **planning and scaffolding phase**.
The dataset has not yet been audited locally, so no image counts, class
frequencies, resolutions, or performance results are claimed.

Read these files in order:

1. [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — decisions and rationale.
2. [`docs/EXECUTION_CHECKLIST.md`](docs/EXECUTION_CHECKLIST.md) — ordered work.
3. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — experiment and evidence log.
4. [`docs/FILE_TREE.md`](docs/FILE_TREE.md) — existing and planned files.
5. [`docs/NEXT_MODEL_HANDOFF.md`](docs/NEXT_MODEL_HANDOFF.md) — direct handoff.
6. [`configs/first_experiment.yaml`](configs/first_experiment.yaml) — first run.
7. [`src/README.md`](src/README.md) — implementation contracts.
8. [`docs/TEAM_EXPERIMENT_PLAN.md`](docs/TEAM_EXPERIMENT_PLAN.md) — team model search and leaderboard.

The original assignment prompt is preserved in [`plan.md`](plan.md). Treat it
as requirements/context; do not fabricate missing dataset facts or results.

## Intended commands

These commands describe the interface the implementation should provide:

```bash
# Download the shared Google Drive folder to data/raw/ once per machine.
python3 -m src.download_data

# Uses data/raw/ by default, or THAI_CHAR_DATA_DIR from a private .env file.
python3 -m src.audit --output-dir results/audit

# A one-off alternative for any direct local/mounted path.
python3 -m src.audit --data-dir "/path/provided-by-instructor" --output-dir results/audit
python -m src.split --data-dir data/clean --output-dir data/splits --seed 42
python -m src.train --config configs/first_experiment.yaml
python -m src.evaluate --config configs/first_experiment.yaml --weights weights/best.pth
python -m src.inference --image sample.jpg --weights weights/best.pth
python -m src.inference --input-dir sample_folder --weights weights/best.pth
```

Only `src.audit` is implemented at this stage. The remaining commands are the
planned interface and should be implemented in the priority order defined by the
execution checklist.

Google Drive is the shared data source. `src.download_data` downloads the public
folder configured by `THAI_CHAR_DRIVE_URL` into a local cache before audit or
training. A notebook should not read individual images from a browser URL.

## Non-negotiable rules

- Use a reproducible stratified 80/20 split and save the split manifest.
- Detect duplicates before splitting; duplicate groups must not cross splits.
- Fit preprocessing decisions using training data only.
- Select checkpoints using validation macro F1, with accuracy as a co-metric.
- Report per-class recall and confusion pairs, not accuracy alone.
- Change one experimental factor at a time.
- Keep label mapping, resize, and normalization identical in training/inference.
- Never report a metric unless its artifact exists under `results/`.

## Submission targets

- Training and inference code
- Best model weights and label mapping
- Dataset audit and 80/20 split evidence
- Ablation table, curves, confusion matrix, and error examples
- 8–10 presentation slides for a rehearsed presentation under 9:30
- Offline-tested demo command and backup model

Deadline: **September 25, 2026**.

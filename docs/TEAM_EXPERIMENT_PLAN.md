# Team Experiment Plan

Status date: 2026-09-15. This is a 72-class **image classification** project,
not object detection. The team source of truth is the shared Google Drive URL
configured in `.env`; each machine caches it locally with `src.download_data`.

## Decision: tools and division of responsibility

Use **PyTorch + timm + Optuna** after the core pipeline is ready.

- `timm` provides comparable pretrained CNN backbones such as EfficientNet-B0,
  ResNet-18, and MobileNetV3-Large.
- Optuna is used only for a narrow hyperparameter search after a backbone wins.
- Use notebooks for orchestration, inspection, and charts—not duplicate training
  implementations. Reusable code belongs in `src/` and is run from configs.

AutoGluon MultiModal can be a separate quick benchmark if one teammate has
time. Its API should be verified against the installed version; do not assume
`MultiModalPredictor.leaderboard()` works like `TabularPredictor.leaderboard()`.
The project's own leaderboard remains the common comparison. Ludwig offers a
declarative pipeline and HPO, but would duplicate the shared PyTorch training
code. Ray Tune becomes useful if the team has several GPUs; otherwise local
Optuna trials are simpler. AutoKeras would add another model/search framework.
Ultralytics/YOLO addresses detection, while this assignment classifies one
character image into one of 72 labels.

Why this order: the rubric requires an explainable CNN baseline, visible transfer
learning and augmentation decisions, and ablations. A transparent training
pipeline also makes live inference easier to package and debug. The team can
still cite an AutoML result as an external reference, but all shortlisted models
must be scored through the same validation evaluator.

## Rules that make experiments comparable

Every notebook and configuration must use:

1. The same Google Drive dataset version and audit report.
2. The same saved grouped-stratified 80/20 manifests.
3. The same label mapping, image resize/padding, and validation preprocessing.
4. Seed 42 for the first comparison.
5. Validation macro F1 as checkpoint metric, with accuracy reported beside it.
6. One changed factor per ablation after the backbone benchmark.

The manifests should contain image paths **relative to the dataset root**, plus
labels and duplicate-group IDs. Each teammate resolves those paths against
their own `get_data_dir()`. Sharing absolute paths from one laptop would make
the same split unusable on another machine. Record a dataset inventory hash
and a split hash so a changed Drive folder or missing cache file is detected.

Never train directly from the Google Drive URL. Run `python3 -m src.download_data`
once, then read the local `DATA_DIR` cache through `src.paths.get_data_dir()`.

## Notebook responsibilities

| Notebook | Owner | Purpose | Required output |
|---|---|---|---|
| `00_data_setup_audit.ipynb` | A | download, inspect folder layout, audit data | audit report, class distribution, sample grid |
| `01_split_validation.ipynb` | A + B review | create/check duplicate-safe 80/20 split | manifests, label map, split integrity report |
| `02_baseline_cnn.ipynb` | B | run E0 custom CNN | run folder, curves, E0 row |
| `TEAM_MODEL_TEMPLATE.ipynb` | every model owner | common setup/checks/run/receipt template | config and comparable run artifacts |
| `03_backbone_benchmark.ipynb` | C + D | run E1 candidates under fixed recipe | ResNet-18/EfficientNet-B0/MobileNet leaderboard |
| `04_augmentation_imbalance.ipynb` | E | E2/E3 safe augmentation and imbalance tests | reviewed augmentation grid, ablation rows |
| `05_optuna_hpo.ipynb` | F | narrow HPO for selected backbone only | Optuna trials and ranked configs |
| `06_confusion_error_analysis.ipynb` | G | confused pairs and E5 proposal | error contact sheets, targeted plan |
| `07_final_training_inference.ipynb` | B + C + H | final confirmation and demo package | final checkpoint, inference proof |
| `08_presentation_figures.ipynb` | H | create slide-quality figures only | plots/tables used in slides |

Copy `TEAM_MODEL_TEMPLATE.ipynb` to a personal notebook, e.g.
`03_resnet18_<name>.ipynb`, then edit only the model config and written
hypothesis. Each owner may work independently once the data/split gate is met.
Notebooks should call the shared command, for example:

```python
!python3 -m src.train --config configs/experiments/efficientnet_b0_base.yaml
```

`src.train` and the later modules are not implemented yet. Build them once;
each notebook calls that same implementation. For fair comparison, the
transfer-model configs must differ only in architecture during the first
backbone search. A runner should reject a missing split or config before
starting a GPU job.

## Workstream assignments

| Workstream | Primary files | Owner role | Completion condition |
|---|---|---|---|
| Data foundation | `download_data.py`, `audit.py`, `split.py` | A | audited data + locked manifests |
| Shared training platform | `dataset.py`, `models.py`, `metrics.py`, `train.py` | B | E0 runs from YAML and saves artifacts |
| Backbone search | `configs/experiments/backbone_*.yaml` | C/D | fair 3-backbone leaderboard |
| Robustness | `augmentation.py`, class-weight config | E | E2/E3 ablation evidence |
| HPO + registry | `hpo.py`, `05_optuna_hpo.ipynb` | F | ranked reproducible trials |
| Error technique | `error_analysis.py` | G | E5 confusion-aware comparison |
| Demo + slides | `inference.py`, `presentation/` | H | offline demo and 9:25 rehearsal |

For a 5-person team, combine C/D, F/G, and H with E. Every pull request must
keep the shared command/config interface working.

## Timeline and gates

| Date | Gate | What must be true before advancing |
|---|---|---|
| Sep 15 | Data gate | Drive download succeeds; audit knows actual data facts |
| Sep 16 | Split/platform gate | manifests are locked; E0 can train/evaluate |
| Sep 17–18 | Backbone gate | E1 leaderboard has 3 fair candidates |
| Sep 18 | Robustness gate | E2/E3 results identify safe augmentation/imbalance choice |
| Sep 19–20 | HPO gate | one selected backbone, narrow search completed/pruned |
| Sep 20 | Interesting-technique gate | confusion-aware E5 accepted or rejected by evidence |
| Sep 21 | Selection gate | top two recipes repeated with multiple seeds |
| Sep 22 | Package gate | final and backup checkpoint; inference works offline |
| Sep 23–24 | Presentation gate | figures, Q&A, two rehearsals under 9:30 |
| Sep 25 | Delivery | no unvalidated last-minute model change |

## Experiment order

### Phase 0 — Build the shared platform

1. Install dependencies and run the Drive download.
2. Audit actual image layout and create the frozen split.
3. Implement one config-driven trainer, evaluator, metrics, and checkpoint format.
4. Make a tiny-subset smoke test pass before a full run.

### Phase 1 — Find the backbone (do not tune everything)

Keep input size, epochs, augmentation, optimizer family, split, and seed fixed.
Compare only the architecture:

| Run | Backbone | Why it is included |
|---|---|---|
| E0 | custom CNN | class-learning baseline |
| E1a | ResNet-18 | stable, interpretable transfer baseline |
| E1b | EfficientNet-B0 | likely accuracy/compute balance |
| E1c | MobileNetV3-Large | fast backup/demo model |

Rank by validation macro F1 first, validation accuracy second, then latency and
parameter count. Select one winner and one backup.

### Phase 2 — Improve the selected backbone

Run controlled ablations, always retaining the same E1 winning config:

| Run | One change | Keep it only if |
|---|---|---|
| E2 | safe Thai-character augmentation | macro F1 or accuracy improves without new semantic errors |
| E3 | capped class-weighted loss | minority recall and macro F1 improve |
| E4 | targeted synthetic data, conditional | real-only validation improves |
| E5 | confusion-aware augmentation | top confused pair recall improves and global metrics do not regress |

### Phase 3 — Narrow HPO and confirmation

Run Optuna only after selecting backbone + augmentation policy. Start with
8–12 trials on a single GPU, with a 6–10 epoch screening budget and pruning;
expand only if time and compute remain. Search a few parameters at once:

- head learning rate: `1e-4` to `3e-3` (log scale)
- fine-tuning learning rate: `1e-5` to `1e-4` (log scale)
- weight decay: `1e-6` to `1e-3` (log scale)
- dropout: `0.1` to `0.5`
- label smoothing: `0.0` to `0.15`
- batch size: feasible values such as 16 or 32
- unfreeze depth: last block versus last two blocks, in a separate small search

Keep the search to about 3–4 dimensions per study, such as head LR, fine-tune
LR, weight decay, and dropout. Do not search architecture, synthetic ratio, all
augmentation values, and loss at the same time. Retrain the top two complete
trials at the full epoch budget. If compute permits, repeat with two or three
seeds. Choose the final model using mean validation macro F1 and accuracy,
then inspect class recall, latency, and score variation. A short pruned trial
never outranks a completed full-budget run.

## Leaderboard format

Write every accepted run to `results/leaderboard.csv` and mirror its summary in
`docs/EXPERIMENTS.md`.

| rank | run_id | config | backbone | seed | val_accuracy | val_macro_f1 | min_recall | params | latency_ms | status |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| TBD | E1b | `efficientnet_b0_base.yaml` | EfficientNet-B0 | 42 | TBD | TBD | TBD | TBD | TBD | planned |

Only completed, full-budget runs whose config, split hash, metrics JSON, and
checkpoint exist may enter the main leaderboard. Screening/HPO trials have a
separate table. The trainer should write one `run_receipt.json` per run with
`run_id`, `config_path`, `split_hash`, `seed`, `backbone`, `max_epochs`,
`best_epoch`, `val_accuracy`, `val_macro_f1`, `minimum_per_class_recall`,
`parameter_count`, `latency_ms`, `metrics_path`, and `checkpoint_path`.
Every teammate submits this receipt, rather than copying scores from notebook
output. If results were obtained with different image size, validation split,
or epoch budget, label them as a separate track instead of mixing ranks.

## Final deliverable workflow

The final notebook does not invent a new model. It loads `configs/final.yaml`,
runs the selected recipe, saves `weights/best_model.pth` and a backup, then
proves `src.inference` produces the same label mapping/preprocessing on a sample
folder. The presentation uses only saved figures/results from this workflow.

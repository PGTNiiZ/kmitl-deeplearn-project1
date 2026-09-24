# Source Implementation Contracts

Implemented: `audit.py`, `paths.py`, `download_data.py`, `split.py`, `train.py`,
`search.py` and `inference.py`. Run them through `notebooks/01-model-search-lab.ipynb`.
The manifest dataset, transforms, model factory and evaluator currently live in
`train.py` so notebook experiments use one implementation. `search.py` handles
benchmarking, controlled ablations, SQLite-backed HPO and confirmation.
`train.py` consumes flat `TrainConfig` JSON; the original YAML is a planning
reference. The table below describes responsibilities, including possible
future module extractions and optional techniques.

| Module | Responsibility |
|---|---|
| `audit.py` | inventory, image metadata, corruption, exact hashes, CSV/JSON report; accepts any `--data-dir` path |
| `split.py` | grouped-stratified 80/20 manifests and stable label mapping |
| `dataset.py` | manifest-backed dataset and shared deterministic preprocessing |
| `augmentation.py` | safe/moderate policies and augmented preview grids |
| `synthesis.py` | optional targeted font rendering for audited minority classes |
| `models.py` | custom CNN plus transfer-backbone factory |
| `metrics.py` | aggregate/per-class metrics and confusion-pair extraction |
| `train.py` | seeded training, AMP, scheduler, early stop, checkpoint, logging |
| `evaluate.py` | real-validation evaluation and artifact export |
| `error_analysis.py` | misclassification CSV/contact sheets/reason taxonomy |
| `inference.py` | CPU/GPU single/folder batched top-3 predictions |
| `utils.py` | config, seeding, hashes, environment and artifact helpers |

Keep inference imports minimal. Save architecture name, input size,
normalization, and label mapping with or beside every checkpoint. Reject a
checkpoint when its class count or label-map hash does not match.

## Offline inference

For a fresh clone, install the project dependencies and run the bundled default
export (the model, config, and label mapping are versioned with the repository):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python -m src.inference --self-check --device cpu
.venv/bin/python -m src.inference --input path/to/character.png --device cpu --top-k 3
```

The included model recognizes one isolated Thai character per image. It does not
perform connected-text OCR.

The current default is the **DenseNet121 E1 candidate** at
`results/inference/densenet121_e1_candidate/`. Its checkpoint, config, and
Thai label mapping are included in the repository. This is a runnable E1
checkpoint, not a claim that E1/E2 comparison or final model selection is complete.
To select this model explicitly:

```bash
python -m src.inference \
  --weights results/inference/densenet121_e1_candidate/best_model.pt \
  --config results/inference/densenet121_e1_candidate/best_config.json \
  --labels results/inference/densenet121_e1_candidate/label_to_index.json \
  --input path/to/character.png \
  --device cpu --top-k 3
```

The checkpoint embeds its config and label mapping; explicit JSON paths are
validated against them, not allowed to override disagreements. Loading applies
EXIF transpose and safe RGB conversion; `src.train.make_transform(config,
training=False)` then performs aspect-preserving bilinear resize with white
square padding to 128×128 for this DenseNet checkpoint, tensor conversion,
and ImageNet mean/std.
Normal inference is deterministic with no training augmentation; `--tta none`
is the default. All canonical validation images are grayscale; transparency
compositing and EXIF correction for unusual inputs are conservative loading
behavior, **not** demonstrated equivalent to training on those inputs.

- IMPLEMENTED + TESTED ON THIS MACHINE: isolated-character single/folder and
  batched inference; CPU, MPS, auto (MPS here); Top-K, softmax confidence,
  Top1–Top2 margin for K≥2, timing, status, deterministic folder ordering,
  per-file ERROR rows, CSV/JSON, checkpoint SHA256 and self-check. At K=1
  margin is `null` in JSON, blank in CSV, and omitted from human-readable text.
- IMPLEMENTED, NOT TESTED HERE: explicit CUDA backend (unavailable on this Mac;
  an explicit request fails rather than falling back).
- AVAILABLE BUT NOT VALIDATION-ABLATED: deterministic `--tta safe` (small fixed
  rotations/translations), execution-tested only; no accuracy gain claimed.
- NOT AVAILABLE AS AN END-TO-END PATH: multi-character detection without local
  detector weights and optional Ultralytics. Production scope here is 72-class
  isolated-character classification, not connected-text OCR.

`--mode single` is the default. `--mode auto` falls back to single-image
classification unless detector weights are explicitly supplied; `--mode multi`
requires an installed detector package and local weights. Use
`--output predictions.csv` or `--output predictions.json` for a folder. A malformed
image produces an ERROR result without discarding valid results. The actual
commands and outcomes of the inference verification are recorded in
`results/final/inference_smoke.json`.

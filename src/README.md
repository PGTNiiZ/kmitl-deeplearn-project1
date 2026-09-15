# Source Implementation Contracts

The next model should create these modules without embedding machine-specific
paths. Configuration must drive every experiment.

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

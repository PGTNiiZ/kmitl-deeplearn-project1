# Notebooks

Use notebooks to launch and inspect experiments as well as create visuals.
Reusable audit, training, evaluation, and inference logic belongs in `src/`.

Start each model-owner notebook by copying `TEAM_MODEL_TEMPLATE.ipynb` and
setting its owner, hypothesis, experiment ID, and config path. It checks the
shared data/split gate and then calls `src.train` when that module exists.
Each owner hands in a config and `run_receipt.json` for the common leaderboard.

## Reading the shared dataset path

Start Jupyter from the repository root, then place this in the first cell of
every notebook:

```python
from src.paths import get_data_dir

DATA_DIR = get_data_dir()
print(DATA_DIR)
```

`DATA_DIR` resolves to `data/raw/` by default. Run `python3 -m src.download_data`
once to cache the shared Google Drive folder locally, then every notebook uses
the same code without committing an absolute personal path.

Suggested notebooks after the pipeline exists:

- `00_data_setup_audit.ipynb`
- `01_split_validation.ipynb`
- `02_baseline_cnn.ipynb`
- `03_backbone_benchmark.ipynb`
- `04_augmentation_imbalance.ipynb`
- `05_optuna_hpo.ipynb`
- `06_confusion_error_analysis.ipynb`
- `07_final_training_inference.ipynb`
- `08_presentation_figures.ipynb`

See `docs/TEAM_EXPERIMENT_PLAN.md` for the owner, inputs, and required outputs
of each notebook. Do not duplicate `src/` functions inside notebooks.

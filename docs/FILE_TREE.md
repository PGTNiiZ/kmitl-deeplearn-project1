# Target File Tree

Legend: `[ready]` exists now, `[build]` is for the implementing model, and
`[generated]` is produced by running the pipeline.

```text
kmitl-deeplearn-project1/
├── plan.md                              [ready] original requirements
├── readme.md                            [ready] entry point and commands
├── requirements.txt                    [ready] initial audit/Drive dependencies
├── .env.example                        [ready] shared Drive URL template
├── configs/
│   ├── first_experiment.yaml            [ready] E0 baseline
│   ├── experiments/                     [build] team benchmark/HPO configs
│   ├── transfer.yaml                    [build] E1/E2 transfer recipe
│   └── final.yaml                       [build] accepted final recipe
├── data/
│   ├── README.md                        [ready]
│   ├── raw/                             [user data; never modify]
│   ├── clean/                           [generated, only if required]
│   ├── splits/
│   │   ├── train.csv                    [generated]
│   │   ├── val.csv                      [generated]
│   │   └── label_to_index.json          [generated]
│   └── synthetic/                       [generated; training only]
├── docs/
│   ├── PROJECT_PLAN.md                  [ready] technical decisions
│   ├── EXECUTION_CHECKLIST.md           [ready] ordered tasks
│   ├── EXPERIMENTS.md                   [ready] ablation registry
│   ├── FILE_TREE.md                     [ready] this map
│   ├── NEXT_MODEL_HANDOFF.md             [ready] implementation prompt
│   └── TEAM_EXPERIMENT_PLAN.md           [ready] team/notebook experiment plan
├── notebooks/
│   ├── README.md                        [ready]
│   ├── TEAM_MODEL_TEMPLATE.ipynb        [ready] copy for each model owner
│   ├── 00_data_setup_audit.ipynb        [build]
│   ├── 01_split_validation.ipynb        [build]
│   ├── 02_baseline_cnn.ipynb            [build]
│   ├── 03_backbone_benchmark.ipynb      [build]
│   ├── 04_augmentation_imbalance.ipynb  [build]
│   ├── 05_optuna_hpo.ipynb              [build]
│   ├── 06_confusion_error_analysis.ipynb [build]
│   ├── 07_final_training_inference.ipynb [build]
│   └── 08_presentation_figures.ipynb    [build]
├── src/
│   ├── README.md                        [ready] module contracts
│   ├── __init__.py                      [ready]
│   ├── paths.py                         [ready] shared data-path resolver
│   ├── download_data.py                 [ready] Drive-to-local cache command
│   ├── audit.py                         [ready] first implementation module
│   ├── split.py                         [build first]
│   ├── dataset.py                       [build]
│   ├── augmentation.py                  [build]
│   ├── synthesis.py                     [build later]
│   ├── models.py                        [build]
│   ├── metrics.py                       [build]
│   ├── train.py                         [build]
│   ├── evaluate.py                      [build]
│   ├── error_analysis.py                [build]
│   ├── inference.py                     [build]
│   └── utils.py                         [build]
├── tests/
│   ├── test_audit.py                    [ready]
│   ├── test_paths.py                    [ready]
│   ├── test_split.py                    [build]
│   ├── test_dataset.py                  [build]
│   ├── test_models.py                   [build]
│   └── test_checkpoint_inference.py     [build]
├── results/
│   ├── README.md                        [ready]
│   ├── audit/                           [generated]
│   ├── runs/                            [generated]
│   ├── plots/                           [generated]
│   ├── confusion_matrix/                [generated]
│   ├── predictions/                     [generated]
│   └── error_analysis/                  [generated]
├── weights/
│   └── README.md                        [ready]
└── presentation/
    └── README.md                        [ready]
```

Do not create empty code files merely to match the tree. Add each module with
tests when its checklist phase begins.

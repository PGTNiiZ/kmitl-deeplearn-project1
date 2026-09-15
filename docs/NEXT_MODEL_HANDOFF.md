# Next Model Handoff

## Objective

Implement a reproducible 72-class Thai character/digit training and inference
pipeline while maximizing generalization to an unseen test distribution.

## Read first

1. `readme.md`
2. `docs/PROJECT_PLAN.md`
3. `docs/EXECUTION_CHECKLIST.md`
4. `docs/FILE_TREE.md`
5. `configs/first_experiment.yaml`

Use `plan.md` only as the original requirements source. The project plan has
already converted it into decisions and ordered work.

## Start here

Ask only for the local dataset directory if it cannot be discovered. The path
may be a downloaded folder, an external disk, or a Google Drive for desktop
mount; always pass it through `--data-dir` rather than hard-coding `data/raw`.
Then:

1. Implement `src/audit.py` and its tests.
2. Run the audit; replace no unknown with a guess.
3. Present the audit findings before finalizing transforms or class weighting.
4. Implement grouped-stratified splitting and prove no duplicate group crosses it.
   Save relative image paths so teammates with different local cache paths can
   use the same train/validation manifests.
5. Implement metrics and the E0 baseline from the YAML.
6. Smoke-test on a small subset, then run E0 and register its artifacts.

## Guardrails

- Do not download, move, rename, or delete user data without explicit scope.
- Do not let synthetic/augmented samples enter validation.
- Do not select a model using training accuracy.
- Do not change multiple experimental factors in one ablation.
- Do not claim transfer, synthesis, or an interesting technique helped until a
  fixed-split comparison supports it.
- Share preprocessing and label mapping between evaluation and inference.
- Keep E0 or the best SAFE model as a working fallback at all times.

## First stopping point

Stop and report after the audit and split artifacts exist. Include the exact
class distribution, imbalance ratio, image properties, corrupt/duplicate counts,
sample grids, split sizes, and remaining uncertainties. Those findings determine
whether the current 224 px input, augmentation ranges, and imbalance method
remain appropriate.

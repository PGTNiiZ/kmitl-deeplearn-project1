# Next Model Handoff

## Implementation update — 2026-09-16

`notebooks/01-model-search-lab.ipynb` implements the requested end-to-end notebook,
backed by `src/split.py`, `src/train.py`, `src/search.py` and `src/inference.py`.
Training dependencies are in `requirements-training.txt`. Start with the notebook
and `notebooks/README.md`; the tasks below describe the original handoff.
Synthetic smoke execution and offline tests verify mechanics, not real-data
performance. Real dataset audit/quality review and full GPU experiments remain
to be run. Near-duplicate/source grouping and separate unseen-test evaluation
remain dataset-specific work. Do not mix smoke, screening and confirmation scores.

## Locked DenseNet121 dataset handoff

The canonical dataset is not stored in GitHub. Copy the separately transferred
dataset to exactly:

`data/raw/1BnPkvJJlE7QDZ0sgEDujS23Pr8Cj3IKq/clean_32x32/`

The locked manifests define 345,569 training images and 86,103 validation images
(431,672 total across 72 classes) under split hash
`73c9c586053c6be848effe66676a9d9ba48125f9fc13bd7a842f74b5d331768e`.
After copying the dataset, run:

`python -m src.locked_train --check-files-only`

This preflight verifies the manifest counts, every canonical file's existence and
readability, labels, per-class counts, duplicate entries, train/validation path,
manifest SHA256 overlap, and source-ID isolation, plus the frozen split artifacts. Extra
unrelated image files are reported but are not used. Do not run E1 after a failed
preflight or with a partial dataset. Only after a full PASS should the teammate run
`python -m src.locked_train` in the locked RTX 5070 Ti environment.

The normal check does not decode all images or hash every byte. When a byte-level
copy verification is required, run the explicit optional check:
`python -m src.locked_train --check-files-only --verify-content-hashes`.

## Objective

Implement a reproducible 72-class Thai character/digit training and inference
pipeline while maximizing generalization to an unseen test distribution.

## Read first

1. `readme.md`
2. `docs/01-project-plan.md`
3. `docs/03-execution-checklist.md`
4. `docs/06-file-tree.md`
5. `configs/first_experiment.yaml`

Use `00-project-requirements.md` only as the original requirements source. The project plan has
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

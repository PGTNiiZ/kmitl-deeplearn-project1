# Experiment Registry

Do not overwrite runs. Use `results/runs/<experiment_id>/<timestamp>/` and save
the resolved config, environment, split hash, metrics, curves, confusion matrix,
predictions, and checkpoint reference.

| ID | Status | Single change | Hypothesis | Val accuracy | Macro F1 | Conclusion/artifact |
|---|---|---|---|---:|---:|---|
| E0 | planned | custom CNN reference | validate pipeline and establish baseline | TBD | TBD | TBD |
| E1 | planned | EfficientNet-B0 transfer | pretrained features improve generalization | TBD | TBD | TBD |
| E2 | planned | safe augmentation | nuisance invariance reduces overfit | TBD | TBD | TBD |
| E3 | planned | capped class-weighted CE | minority recall/macro F1 improve | TBD | TBD | TBD |
| E4 | conditional | targeted synthesis | minority variation helps real validation | TBD | TBD | TBD |
| E5 | planned | confusion-aware augmentation | top pair recall improves | TBD | TBD | TBD |
| E6 | planned | selected fine-tuning | controlled adaptation improves best recipe | TBD | TBD | TBD |
| E7 | optional | TTA | stable views improve predictions | TBD | TBD | TBD |
| E8 | optional | ensemble | complementary errors improve accuracy | TBD | TBD | TBD |

## DenseNet121 completion audit (pre-final, not a new controlled ablation)

The original registry above is historical planning material, including its earlier E1
EfficientNet-B0 definition. In this completion round, E1 means a DenseNet121
baseline. These are different experiment families; do not combine their numbers.

- Development root: `data/raw/1BnPkvJJlE7QDZ0sgEDujS23Pr8Cj3IKq/clean_32x32`.
  The source-v3 split records 432,000 source images, 431,672 included images,
  345,569 train and 86,103 validation images across 72 classes. Its recomputed
  split hash is `73c9c586053c6be848effe66676a9d9ba48125f9fc13bd7a842f74b5d331768e`.
  Train/validation source IDs and file hashes have zero cross-split overlap.
- Model indices 0–71 are assigned by the source-v3 `label_to_index.json`; the
  DenseNet checkpoint's embedded mapping matches it exactly. The labels contain
  Thai Unicode and numeric prefixes (index 0 `0_ก`, 1 `10_ฎ`, 71 `9_ญ`).
  Never derive this ordering from the external dataset.
- Historical DenseNet121 E1: `results/train-for-bestmodel/31e3890aa46ae116/runs/group_02_screening-densenet121.ra_in1k-18bddfb447/best.pt`.
  Architecture `densenet121.ra_in1k`, 7,027,656 parameters, image size 128,
  seed 42, 50-epoch budget, best epoch 26 of 36 completed, training time
  11,396.53 seconds on the recorded NVIDIA RTX 5070 Ti. Validation accuracy
  0.9979675505, macro F1 0.9979567744, macro precision 0.9979609964,
  macro recall/balanced accuracy 0.9979574981. Recomputed from all 86,103
  saved predictions and reconciled with the stored confusion matrix and metrics.
  This is validation evidence, not an external-test result.
- The 72 training class counts range 4,786–4,804 (mean 4,799.57, median 4,800;
  maximum/minimum 1.00376). Class weighting was not used because the
  development dataset does not show meaningful class imbalance requiring correction.
  E3 is skipped; `results/final/class_distribution.csv` records the counts.
- The historical baseline's largest bidirectional validation confusion is
  `43_า → 55_ๅ` (46) and `55_ๅ → 43_า` (28). The inspected image montage
  `results/final/e1_error_examples.png` shows visually similar glyph silhouettes
  with varied stroke weights/curves. Some edges look faint or soft, but the
  samples do not establish rotation, translation, blur, or contrast as a cause.
  These are historical E1 errors, **not** E5 targets selected after E2.
- The historical E1 environment is CUDA/PyTorch 2.15 dev and code hash
  `43c36a84c0b9badded27a820444d725ade29bf22a11cc54a93b92ef4a85ec706`;
  the audited local train/split code hash is
  `5e21c21210600d1d4ae6874359969593a860144ac578911883f14a5aecf0ff76`
  on MPS/PyTorch 2.14. A new local E2 cannot be called a controlled
  single-variable ablation against this historical E1. No new same-code,
  same-environment 50-epoch E1/E2 pair was run in this round. No E5 policy was
  defined or trained; first inspect errors from the strongest completed pre-E5
  local model and then freeze one policy before training.
- Existing inference entry: `python -m src.inference`. Default checkpoint is now
  `results/inference/densenet121_e1_candidate/best_model.pt`, packaged with its
  config and Thai label mapping. This is the historical E1 candidate, not a
  selected final model. The older MobileNetV3 export remains available with
  explicit `--weights`. An observed
  JSON list-versus-checkpoint tuple conflict for mean/std was fixed narrowly;
  the regression test and the 36-test suite pass. CPU self-check and single,
  MPS single/auto, folder, Top-3, confidence/margin, CSV and JSON were exercised on canonical
  validation samples; the malformed image produced an ERROR row without aborting.
  `results/final/inference_smoke.json` records commands and results. Inference
  uses deterministic `ResizePad` (bilinear, white pad), RGB, ToTensor and
  ImageNet mean/std; normal TTA is `none`. All 86,103 canonical validation
  images opened as grayscale (`L`), with no nontrivial EXIF orientation or
  unreadable files; both paths convert them to RGB. On three real validation
  images the inference and validation tensors were bit-identical. For other
  RGBA/EXIF images the loader policies differ, so equivalence is not claimed
  outside the canonical validation data.
- E2 is not complete, E5 is not complete, no final checkpoint has been selected
  or frozen, and `burapha_72` has not been read or evaluated. No external
  metrics or generalization gap exist. Do not populate `results/final/export/`
  or `external_metrics.json` from the historical baseline. A valid next run
  requires a frozen same-code DenseNet121 E1/E2 protocol and full same-budget
  executions, then pre-E5 validation error inspection and validation-only model
  selection before any external-test access.

## Run note template

```text
Experiment ID:
Commit/code snapshot:
Resolved config:
Dataset audit hash:
Split manifest hash:
Hypothesis:
Only changed variable:
Constants:
Hardware/runtime:
Best epoch/checkpoint:
Validation metrics:
Top confusion pairs:
Observed failure modes:
Conclusion:
Next experiment:
```

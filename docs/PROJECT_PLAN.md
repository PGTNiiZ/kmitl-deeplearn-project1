# Project Plan

Status date: 2026-09-11. Deadline: 2026-09-25. Dataset statistics are unknown
until the local audit is complete.

## 1. Problem Analysis

Build a 72-class image classifier for Thai characters/digits. The official test
distribution may differ from training, so validation quality, robustness, and
leakage control matter more than near-perfect training accuracy. Required work
includes a custom CNN reference, transfer learning, augmentation or synthesis,
imbalance handling, ablations, reliable inference, and presentation evidence.

## 2. Rubric Strategy

| Requirement | Score | Implementation | Evidence | Main risk |
|---|---:|---|---|---|
| Prediction performance | 5% | reproducible training and checkpointing | held-out accuracy/F1 | overfit validation |
| Performance ranking | 3% | competitive final model, optional TTA | locked experiment results | optimizing only train accuracy |
| Transfer learning | 1.5% | pretrained backbone plus staged fine-tuning | freeze/unfreeze ablation | merely replacing the head |
| Augmentation/synthesis | 1.5% | Thai-safe transforms plus targeted synthesis | examples and ablation | changing character identity |
| Interesting idea | 2% | confusion-aware targeted augmentation | confusion-pair before/after | complexity without benefit |
| Presentation/materials | 2% | concise slides and reproducible artifacts | timed rehearsal and plots | unsupported claims/overtime |

Guaranteed points: complete pipeline, transfer learning, safe augmentation,
correct split, and clear presentation. Competitive points: test accuracy and
ranking. Differentiating points: confusion-aware learning with an ablation.
Allocate most compute to E1–E5; do not spend scarce time on high-risk methods
until the competitive pipeline works.

## 3. Dataset Analysis Plan

Run an audit before model selection is finalized. Record total images, exact
class count, per-class count, min/max and imbalance ratio, formats, corruption,
width/height/aspect ratio, color mode, file hashes, perceptual hashes, and likely
duplicate groups. Generate class histograms, a per-class sample grid, minority
and majority examples, resolution/aspect plots, and suspected near-duplicate
pairs. Manually review background, fonts, handwriting/printing, blur, noise,
rotation, scale, position, crop quality, and visually similar classes.

Create a stable `label_to_index.json`. Group exact/near duplicates before the
stratified 80/20 split so related images cannot cross boundaries. Save manifests
with paths, labels, group IDs, and split names. Unknown until audit: every
numerical dataset statistic and every visual-data conclusion.

## 4. Dataset Challenges

Confirmed only: 72 classes and unequal class sizes. Plausible issues such as
small distinguishing marks, blur, crop variation, or font shift are hypotheses,
not findings. The audit must determine which are real. A valid final answer to
“what makes this dataset difficult?” must cite audit figures and sample images.

## 5. CNN Concepts Applied

| Class concept | Function | Project relevance/location |
|---|---|---|
| Convolution/kernel/feature map | detect local patterns | learns edges, strokes, loops, and marks in backbone |
| Stride/padding | control sampling and borders | preserves edge strokes while reducing resolution carefully |
| Pooling/receptive field | aggregate broader context | combines local strokes into full-character shape |
| ReLU variants | nonlinear representation | ReLU in baseline; backbone-native activation in transfer model |
| Batch normalization | stabilize feature scales | after baseline convolutions; inherited in pretrained backbone |
| Dropout | regularize classifier | before the 72-class head |
| Global average pooling | summarize channels compactly | avoids a large overfitting-prone dense stack |
| Fully connected/softmax | produce class logits/probabilities | final 72 outputs; softmax only for inference display |
| Cross entropy | train class logits | weighted or smoothed variant tested by ablation |
| Backprop/optimizer/LR/epoch/batch | update weights over passes | AdamW, staged LRs, reproducible training loop |
| Split/overfit/underfit | measure generalization | fixed grouped-stratified validation and curve diagnosis |
| Augmentation | perturb real images | training-only, label-preserving transforms |
| Transfer/fine-tuning | reuse and adapt features | pretrained backbone, then low-LR unfreezing |

## 6. Baseline Model

Input: RGB 224×224 after aspect-preserving resize and padding. Architecture:
four blocks of `3×3 conv, padding=1, stride=1 → batch norm → ReLU → 2×2 max
pool`, using 32/64/128/256 filters, then global average pooling, dropout 0.3,
and a 72-unit linear layer. Use cross entropy, AdamW (`lr=1e-3`, weight decay
`1e-4`), batch 32, at most 30 epochs, early stopping, and no pretrained weights.
This compact baseline exposes whether transfer learning adds real value.

## 7. Transfer Learning Candidates

| Backbone | Accuracy potential | Params/cost | Overfit risk | Character fit | Presentation value |
|---|---|---|---|---|---|
| EfficientNet-B0 | high | ~5.3M; moderate | low–moderate | strong multi-scale features at 224 | strong accuracy/efficiency story |
| ResNet-18 | good | ~11.7M; fast | moderate | simple, stable, easy to explain | excellent CNN teaching value |
| MobileNetV3-Large | good | ~5.5M; very fast | low–moderate | useful for demo constraints | clear efficiency comparison |

Parameter counts are approximate architecture references, not experimental
results. Confirm library-specific counts in the model summary.

## 8. Recommended Backbone

Primary: EfficientNet-B0 for its capacity/compute balance at 224 px. Backup:
ResNet-18 because it is robust and easy to debug. Baseline: the custom CNN
above. Reconsider after the audit if images are extremely small, the dataset is
very large, or compute is unusually limited.

## 9. Augmentation Strategy

Safe defaults: rotation ±7° (`p=.5`), translation up to 5% (`p=.5`), scale
0.9–1.1 (`p=.4`), mild brightness/contrast ±10% (`p=.3`), mild blur/noise
(`p=.15` each), and padding-preserving resize. Moderate experiments: affine
shear ±5°, perspective distortion ≤0.05, or small random erasing ≤5% area.
Risky and disabled by default: horizontal/vertical flips, 90° rotations,
aggressive crops, large perspective changes, and heavy erasing. Each may remove
or move a mark that defines a different class. Save augmented sample grids for
human semantic review before training.

## 10. Synthetic Data Strategy

Augmentation transforms an observed labeled image; synthesis creates a new image
by rendering a character with fonts, stroke widths, scales, positions,
backgrounds, blur, noise, lighting, and compression. After confirming labels
and glyph support, target minority classes toward a documented count threshold
rather than synthesizing all classes equally. Start with no more than 25%
synthetic examples per epoch and mix them with real images. Reduce domain gap by
sampling real background/noise statistics, using multiple fonts, randomizing the
rendering pipeline, and validating gains on real-only validation data.

## 11. Class Imbalance Strategy

Candidate methods: class-weighted cross entropy (simple but may amplify noisy
minorities), weighted sampling (balanced exposure but repeats examples), targeted
augmentation/synthesis (adds variation but may introduce domain shift), and
focal loss (focuses hard cases but adds tuning risk). Primary first test:
class-weighted cross entropy with weights proportional to inverse square-root
frequency and capped after the audit. Do not simultaneously add weighted
sampling; test it as an alternative so the effective distribution is not
over-corrected.

## 12. Generalization Strategy

Use a grouped, stratified, seeded split; train-only statistics; ImageNet
normalization for pretrained models; modest dropout, weight decay, and label
smoothing; early stopping; and staged fine-tuning. Monitor validation loss,
accuracy, macro F1, per-class recall, and train–validation gap. Select the best
checkpoint on macro F1, breaking close ties with accuracy. Training accuracy
99% and validation accuracy 75% indicates a 24-point generalization gap, likely
from memorization, leakage-free distribution shift, insufficient variation, or
excess capacity. Respond with safer augmentation, stronger regularization,
earlier stopping, balanced exposure, reduced capacity, or lower/less extensive
fine-tuning—one controlled change at a time.

## 13. Interesting Technique Candidates

| Technique | Gain/generalization | Difficulty/cost | Risk | Presentation |
|---|---|---|---|---|
| Confusion-aware augmentation | high | medium | low–medium | excellent |
| Progressive unfreezing | medium | low | low | strong |
| MixUp | medium | low | medium for character boundaries | good |
| TTA | small–medium | low training/high inference | low | good |
| ArcFace metric head | potentially high | high | high | excellent |

## 14. Recommended Interesting Technique

Safe first: confusion-aware targeted augmentation because it is driven directly
by validation errors and is easy to ablate. High-risk/high-reward: ArcFace-style
classification to increase separation of visually similar classes. Attempt the
latter only after E5 is stable.

## 15. Confusion-Aware Learning Strategy

Rank bidirectional confusion pairs using the validation confusion matrix. Review
their true/predicted images for small marks, blur, stroke width, crop, resolution,
and class frequency. Add class-specific safe transformations or targeted
synthetic examples that preserve the discriminating feature. Retrain from the E3
recipe and compare pairwise recall, macro F1, and overall accuracy. Never tune on
the official test set.

## 16. Experiment Matrix

| ID | Change from previous | Hypothesis | Selection/evidence |
|---|---|---|---|
| E0 | custom CNN, minimal augmentation | establishes reference | accuracy, macro F1, curves |
| E1 | EfficientNet-B0 transfer | pretrained features generalize better | same split/metrics |
| E2 | safe augmentation | reduces nuisance sensitivity | augmentation grid + delta |
| E3 | imbalance loss | raises minority recall | per-class recall + macro F1 |
| E4 | targeted synthesis | helps minorities without hurting real validation | real-only validation delta |
| E5 | confusion-aware augmentation | reduces top pair errors | pairwise before/after |
| E6 | selected fine-tuning recipe | best capacity adaptation | final validation comparison |
| E7 | optional TTA | stable transforms improve predictions | accuracy/F1 and latency |
| E8 | optional 2-model ensemble | complementary errors improve accuracy | gain versus cost |

All runs use the same split and seed unless the experiment explicitly studies
seed variance. Record actual result and conclusion in `docs/EXPERIMENTS.md`.

## 17. Metrics

Monitor train/validation loss and accuracy each epoch. Select checkpoints using
validation macro F1; report validation accuracy because ranking likely depends
on it. Also store macro precision/recall/F1, balanced accuracy, per-class recall
and accuracy, confusion matrix, top confused pairs, and optional top-3 accuracy.

## 18. Training Pipeline

`raw data → audit/corruption and duplicate checks → canonical labels → grouped
stratified 80/20 manifests → training-only augmentation → loaders → model →
frozen-head training → low-LR fine-tuning → validation → confusion/error
analysis → controlled ablation → best checkpoint → inference package`.

## 19. Inference Pipeline

Load the saved checkpoint, architecture metadata, and exact label mapping. Apply
the same aspect-preserving resize, RGB conversion, and normalization without
training augmentation. Support one file or a folder with batched processing.
Output predicted label, confidence, and top-3 predictions to console and CSV.
Before demo day, test offline in a clean environment, verify CPU fallback,
checksum weights, and keep the backup checkpoint and sample inputs locally.

## 20. Error Analysis Strategy

Export each important error with image path, thumbnail, true label, predicted
label, confidence, confusion-pair rank, and reviewer reason. Aggregate reasons
such as shape similarity, blur, crop, font, small stroke, imbalance, noise, or
resolution. Convert repeated causes into one experiment, then retrain and compare.

## 21. Final Model Selection Criteria

Compare validation accuracy, macro F1, parameters, latency on demo hardware,
train–validation gap, training cost, generalization risk, and contribution of the
interesting technique. Choose the simplest model within a small performance
margin of the leader unless the larger model shows repeatable gains across seeds.

## 22. Presentation Plan

| Slide | Content/visual | Time |
|---|---|---:|
| 1 | problem, team, generalization objective | 0:40 |
| 2 | dataset counts/distribution/sample grid | 0:55 |
| 3 | audited challenges and confusing samples | 0:55 |
| 4 | CNN + transfer architecture diagram | 1:00 |
| 5 | augmentation vs synthesis examples | 0:55 |
| 6 | staged training and imbalance strategy | 0:55 |
| 7 | confusion-aware technique diagram | 1:00 |
| 8 | ablation table and learning curves | 1:20 |
| 9 | final confusion matrix/errors/results | 1:05 |
| 10 | final model, demo, conclusion | 0:40 |

Target total: 9:25. Prefer visuals over paragraphs and rehearse transitions.

## 23. Q&A Preparation

Prepare 20–40 second answers grounded in artifacts: backbone choice (capacity and
audit), ImageNet transfer (generic edges/textures), freezing (protect pretrained
features), lower fine-tuning LR (avoid destructive updates), no flips (semantic
identity), macro F1 (minority visibility), synthesis domain gap, overfitting gap,
checkpoint selection, leakage prevention, and how confusion-aware augmentation
changed specific pairwise recall. Never claim causality without an ablation.

## 24. Team Task Allocation

For 5–8 people: data audit/split; custom baseline; transfer learning; augmentation
and synthesis; evaluation/error analysis; inference/demo; presentation and docs.
Pair-review split logic, inference preprocessing, and experiment conclusions.
Hold two whole-team walkthroughs so every member can explain the full pipeline.

## 25. Project Timeline

- Sep 11–12: acquire/audit data, confirm labels, duplicate policy, split manifest.
- Sep 13: E0 custom baseline and evaluation artifacts.
- Sep 14–15: E1 transfer head training and staged fine-tuning.
- Sep 16: E2 safe augmentation with reviewed sample grid.
- Sep 17: E3 imbalance experiment.
- Sep 18–19: E4 targeted synthesis if audit justifies it.
- Sep 20: E5 confusion-aware experiment; freeze optional scope.
- Sep 21: repeat best candidates/seeds and choose final model.
- Sep 22: inference package, CPU/offline verification, backup checkpoint.
- Sep 23: plots, ablation table, slides, and documentation.
- Sep 24: full timed rehearsals, clean-machine demo, submission package.
- Sep 25: checksum final package; no risky model changes before presentation.

## 26. Risks and Backup Plan

| Risk | Detection/prevention | Backup |
|---|---|---|
| GPU/time limit | benchmark one epoch early; mixed precision | ResNet-18/MobileNet and smaller batch |
| overfitting | gap/curves/per-class metrics | earlier E2 checkpoint, more regularization |
| unsafe augmentation | human grids and pair analysis | safe-only transform policy |
| synthesis hurts | real-only E3 vs E4 validation | omit synthesis and report negative result |
| imbalance persists | minority recall/macro F1 | compare capped weights vs sampler |
| interesting idea fails | strict E3/E5 ablation | present progressive unfreezing as safe idea |
| label mismatch | round-trip label tests | bundle mapping inside checkpoint package |
| demo environment fails | clean CPU rehearsal/offline assets | backup laptop/model/CSV screenshots |
| presentation overtime | recorded 9:25 rehearsal | pre-agreed slides to shorten |

## 27. SAFE / COMPETITIVE / HIGH-RISK Plan

- SAFE: E0–E3, EfficientNet-B0/ResNet-18, safe augmentation, capped class weights,
  fixed validation, and reliable inference.
- COMPETITIVE: SAFE plus targeted synthesis, progressive fine-tuning,
  confusion-aware augmentation, and optional TTA.
- HIGH-RISK: COMPETITIVE plus ArcFace, contrastive learning, complex ensembles,
  or distillation only if time remains.

Recommendation: COMPETITIVE, with the SAFE pipeline locked as the fallback.

## 28. Final Recommended Architecture

`RGB image → aspect-preserving resize/pad to 224 → safe train-only augmentation
→ ImageNet normalization → EfficientNet-B0 → global average pooling → dropout
0.3 → 72 logits → softmax for displayed probabilities`. Train the head while
the backbone is frozen, unfreeze the final blocks at one-tenth LR, and proceed
only while real validation macro F1 improves. Use capped inverse-square-root
class weights if E3 supports them. Mix targeted synthetic samples only into
training if E4 improves real-only validation. Apply confusion-aware augmentation
as E5 and retain it only with measured gains.

## 29. Five Actions to Complete Today

1. Download and freeze a read-only copy of the dataset; document its checksum.
2. Implement and run the audit, including corruption and duplicate detection.
3. Review labels and duplicate groups; save a seeded grouped-stratified 80/20 split.
4. Implement evaluation metrics, plots, artifact naming, and a smoke test.
5. Run E0 on a small subset, then start the complete custom-CNN baseline.

## Recommended First Experiment

| Setting | Value |
|---|---|
| Model | custom 4-block CNN from section 6 |
| Input size | 224×224 RGB, aspect-preserving pad |
| Pretrained weights | none |
| Split | grouped stratified 80/20, saved manifest |
| Seed | 42 |
| Batch size | 32; reduce only for memory |
| Loss | cross entropy, no weights in E0 |
| Optimizer | AdamW, weight decay `1e-4` |
| Initial LR | `1e-3` |
| Fine-tuning LR | not applicable to E0 |
| Scheduler | ReduceLROnPlateau on validation macro F1, factor .2, patience 2 |
| Epochs | maximum 30 |
| Augmentation | rotation ±5°, translate ≤3%, scale 0.95–1.05 |
| Normalization | training-set channel mean/std, computed from train only |
| Freeze/unfreeze | not applicable |
| Imbalance | none; expose the natural baseline weakness |
| Early stopping | patience 5 after minimum 8 epochs |
| Checkpoint metric | validation macro F1; accuracy tie-breaker |
| Record | all losses/accuracies, macro P/R/F1, balanced accuracy, per-class recall, confusion matrix, runtime |

E0 is deliberately simple: it validates labels, split integrity, metrics,
checkpointing, and inference compatibility before transfer learning adds hidden
complexity. E1 should keep the same split and evaluation code so its gain is
credible.

# Execution Checklist for the Implementing Model

Complete phases in order. Do not mark a phase complete without its artifacts.

## P0 — Reliable system

- [ ] Confirm local dataset path and actual directory/label convention.
- [ ] Implement `src/audit.py`; never infer statistics from folder names alone.
- [ ] Detect corrupt files, exact hashes, and perceptual near-duplicate candidates.
- [ ] Manually review duplicate candidates and save group IDs.
- [ ] Implement grouped-stratified 80/20 splitting with seed 42.
- [ ] Save `train.csv`, `val.csv`, `label_to_index.json`, and audit JSON/plots.
- [ ] Add unit tests for label mapping, split disjointness, transforms, and checkpoint reload.
- [ ] Implement metrics and artifact logging before full training.
- [ ] Implement E0 and verify on a tiny subset, then run it completely.
- [ ] Implement single-image and folder inference using shared preprocessing.

## P1 — Accuracy and generalization

- [ ] Implement EfficientNet-B0 and ResNet-18 through one model factory.
- [ ] Train the new head with the backbone frozen.
- [ ] Fine-tune final blocks at lower LR; stop if validation degrades.
- [ ] Review every augmentation using saved image grids.
- [ ] Run E1, E2, and E3 with one-factor changes.
- [ ] Export curves, per-class recall, confusion matrix, and top confusion pairs.
- [ ] Repeat leading recipe with seeds 7 and 2026 if compute permits.

## P2 — Interesting technique

- [ ] Review top confusion pairs with real samples.
- [ ] Define class-specific, semantics-preserving augmentation rules.
- [ ] Run E5 against the locked E3 recipe.
- [ ] Retain only if macro F1/pair recall improves without unacceptable accuracy loss.
- [ ] Implement targeted synthesis only after confirming Unicode labels/fonts.

## P3 — Optional

- [ ] Test safe test-time augmentation and report latency.
- [ ] Test a two-model ensemble only if errors are complementary.
- [ ] Attempt ArcFace only after all deliverables and fallback model are locked.

## Definition of done

- [ ] One command reproduces each accepted experiment from a config.
- [ ] Each result row points to a config, checkpoint, curves, and metrics JSON.
- [ ] Best and backup weights reload on CPU without network access.
- [ ] Demo handles one image and a folder and prints top-3 labels/confidence.
- [ ] Slides contain only audited statistics and recorded results.
- [ ] Presentation has been rehearsed under 9:30 at least twice.

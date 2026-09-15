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

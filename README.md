# APTOS Diabetic Retinopathy Classifier

Two-stage diabetic retinopathy screening pipeline built on the APTOS 2019 Blindness
Detection dataset, with an eventual generalization target of Messidor-2.

## Pipeline

1. **Binary screener** — flags presence/absence of diabetic retinopathy
2. **Severity classifier** — grades DR severity for images flagged positive

## Structure

- `logistic_regression_pt.ipynb` — Linear baseline (binary + multiclass)
- `shallow_nn_pt.ipynb` — MLP baseline (binary + multiclass)
- `preprocess.ipynb` — crop_fundus, pad_to_square, Ben Graham illumination
- `aptos2019/` — dataset (gitignored — see Data section)

## Data

This repo does not include the APTOS 2019 dataset or trained checkpoints.
Download via the [Kaggle APTOS 2019 competition](https://www.kaggle.com/c/aptos2019-blindness-detection)
and place under `aptos2019/`.

## Results so far

| Model | Task | Train Acc | Val Acc | Test Acc |
|---|---|---|---|---|
| Logistic Regression | Binary | 97.44% | 95.63% | 95.36% |
| Shallow NN | Binary | 98.53% | 94.54% | 95.63% |
| Logistic Regression | Multiclass | 56.28% | 46.39% | 42.51% |
| Shallow NN | Multiclass | 69.72% | 56.70% | 48.50% |

Binary screening performs strongly across both architectures. Multiclass severity
grading is harder and motivates moving to CNNs, which is the current work in progress.

## Next steps

- CNN architecture for severity classification
- Messidor-2 generalization testing (train/dev/train-dev distribution mismatch analysis)

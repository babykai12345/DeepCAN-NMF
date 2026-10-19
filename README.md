# DeepCAN-NMF

Implementation of *DeepCAN-NMF: A Deep Canonical Correlation and Non-negative Matrix Factorization Framework for Multimodal Imaging Genetics in Alzheimer's Disease*.

DeepCAN-NMF integrates SNP, structural MRI, FDG-PET and gene-expression data. Modality-specific encoders learn nonlinear representations; DCCA alignment and a shared nonnegative decoder jointly support cross-modal integration and interpretable latent modules. Graph, orthogonality and L1 penalties regularize training. Hub features are selected from decoder weights.

## Installation

Python 3.10+:

```bash
pip install -r requirements.txt
```

## Input files

Prepare the following CSVs in `data/` (samples in rows, features in columns; first column = participant ID):

| File | Modality |
| --- | --- |
| `c.csv` | SNP |
| `a.csv` | MRI |
| `b.csv` | PET |
| `d.csv` | Gene expression |
| `label.csv` | Diagnosis (`Label` column by default) |

Sample IDs must match across files. Check the label codebook before using the default positive class `1` and negative class `0`.

## Usage

```bash
python run.py --mode both --data-dir data
```

This runs selection-repeated cross-validation followed by an independently fitted final model. Use `--mode cv` or `--mode final` for either stage alone.

Common options: `--seed 42`, `--folds 5`, `--optuna-trials 50` (optional inner-fold tuning), `--top-n 12`, `--cumsum-threshold 0.80`, `--expected-n 323` (optional sample-count check). By default, hub lists are capped at 50 features per modality; use `--max-hub-features 0` for uncapped cumulative-weight selection.

Outputs are saved in `DeepCAN_NMF_selection_nested_output/` and include fold-specific selections, held-out ROC predictions, final-module hub features, and exploratory cross-modal correlations.

## Code structure

- `data.py`, `model.py`, `training.py`: input handling, architecture, optimization.
- `features.py`: module ranking and hub-feature selection.
- `evaluation.py`: fold-wise diagnostic evaluation and stability.
- `final_analysis.py`: independent final-model analysis and BH-FDR.
- `visualization.py`: plots; `cli.py`: arguments; `run.py`: entry point.
- `legacy.py`: original exploratory workflow (`--mode legacy`).

All scripts are in the repository root. For a synthetic-data check, install `requirements-dev.txt` and run `pytest -q test_smoke.py`.

## Scope and data availability

ADNI participant-level data are **not included**. This repository starts from prepared CSVs; it does not perform the manuscript's upstream limma DEG screening, initial 4:1 held-out partition, external GSE282742 evaluation or baseline comparisons. Fold-wise results cannot verify leakage-free DEG screening if gene inputs were selected upstream. The legacy ROC is exploratory, not an unbiased diagnostic estimate. Reproducing all manuscript results requires the original authorized data and preprocessing procedures. Do not upload subject-level outputs to GitHub.

## License

MIT (`LICENSE`).

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

## Main analysis: one participant-level 4:1 holdout

```bash
python run.py --mode holdout --data-dir data --expected-n 323 --seed 42
```

This implementation uses a single stratified 80/20 outer test partition. The
development subjects alone supply inner 20% validation, normalization,
model training, module ranking, Top-12 feature selection and logistic classifier
training. The outer test partition is used only for the diagnostic ROC/AUC.
The default configuration matches Table S1 (k=60, hidden=128, learning rate
0.001 and regularization coefficients), but **the published AUCs are not
hard-coded or guaranteed**. Existing mode aliases `both`, `cv`, and `final`
run the same holdout protocol; `legacy` retains exploratory legacy code.

Files in `DeepCAN_NMF_holdout_output/holdout_evaluation/` include the hashed
split manifest, predictions, training-selected features, rankings and weights.

### Manuscript reproducibility limitation

The original ADNI 469-participant data, original 4:1 identity-level split,
limma expression analysis and the archive predictions are **not contained
in this zip**. The 323-person label-filtered CSV input cannot reproduce the
upstream CN-assisted 610/327 DEG comparison. Therefore, if `d.csv` contains
globally pre-screened genes, the script **cannot establish absence of
upstream data leakage**. Supply verified training-only DEGs for the *same*
split and confirm it matches the historical 469-person split before claiming
this rerun reproduces the published estimates.

GSE282742, extra 160-person ADNI evaluation, Table 2, Table S7 and
Table S9 cannot be recreated from these inputs. Do not upload identifiable
ADNI outputs to a public repository.

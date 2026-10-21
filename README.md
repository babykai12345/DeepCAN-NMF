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

|File|Modality|
|-|-|
|`c.csv`|SNP|
|`a.csv`|MRI|
|`b.csv`|PET|
|`d.csv`|Gene expression|
|`label.csv`|Diagnosis (`Label` column by default)|

Sample IDs must match across files. Check the label codebook before using the default positive class `1` and negative class `0`.

## Main analysis

```bash
python run.py --mode holdout --data-dir data --expected-n 323 --seed 42
```

This implementation uses a single stratified 80/20 outer test partition. The
development subjects alone supply inner 20% validation, normalization,
model training, module ranking, Top-12 feature selection and logistic classifier
training. The outer test partition is used only for the diagnostic ROC/AUC.



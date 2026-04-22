# Credit Risk Feature Selection Using a Genetic Algorithm (GA) + SVM

Wrapper-based feature selection for credit risk classification using:
- **Baseline**: SVM trained on *all* engineered features
- **Traditional filter**: `SelectKBest` (mutual information) + SVM
- **GA wrapper**: DEAP Genetic Algorithm that directly optimizes **10-fold cross-validated accuracy** of an SVM

A small **Streamlit** app is included to run the pipeline and compare methods on two datasets.

## What this project does

- Loads a credit dataset CSV, preprocesses numeric + categorical columns (imputation, scaling, one-hot encoding)
- Evaluates a baseline SVM with **Stratified 10-fold CV accuracy**
- Runs a **traditional** feature-selection baseline (`SelectKBest`) and chooses the best `k` using the same CV objective
- Runs a **GA wrapper** that searches for a feature subset maximizing the same CV objective
- Produces a side-by-side comparison table and the selected feature subset

## Datasets

This repo includes two CSVs:

- `german_credit_data.csv` (German credit dataset in a Kaggle-friendly format)
  - Kaggle: https://www.kaggle.com/datasets/uciml/german-credit
- `credit_risk_dataset.csv` (secondary credit risk dataset)
  - Kaggle: https://www.kaggle.com/datasets/programmer3/credit-risk-dataset

### Target label handling (important)

The scripts look for a target/label column using common names (e.g., `Risk`, `Credit Risk`, `target`, `class`).

If no explicit target column is present:
- For the **German** dataset mode, the code first tries to download the **official UCI Statlog German Credit** labels and align them by row order.
- If that fails (or for generic datasets), it can generate a **proxy target** from credit attributes.

The proxy target is meant for **demo/testing only** and should not be used for research conclusions.

## How to run

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

Install the core dependencies used by the pipeline + app:

```powershell
pip install numpy pandas scikit-learn deap streamlit altair
```

### 3) Run the experiment (CLI)

Runs the GA wrapper feature selection on `german_credit_data.csv` by default:

```powershell
python .\ga_svm_feature_selection.py
```

### 4) Run the Streamlit app

```powershell
python -m streamlit run .\streamlit_app.py
```

The app runs the pipeline for:
- Dataset 1: German credit CSV
- Dataset 2: secondary credit dataset CSV (if present)

## Metrics (kept explicit)

- **Baseline Accuracy**: mean accuracy from Stratified **10-fold cross-validation** using all engineered features.
- **Traditional Accuracy**: best mean 10-fold CV accuracy among `k = 1..N` features selected by `SelectKBest(mutual_info)`.
- **Final GA Accuracy**: mean 10-fold CV accuracy on the GA-selected feature subset.

No “confidence score” is reported; all metrics shown are clearly defined CV accuracies.

## Project structure

- `ga_svm_feature_selection.py` — core pipeline (preprocessing, CV evaluation, SelectKBest baseline, GA wrapper)
- `streamlit_app.py` — Streamlit UI to run and compare both datasets
- `german_credit_data.csv` — dataset 1
- `credit_risk_dataset.csv` — dataset 2
- `run.txt` — original run notes/commands
- `kagle link.txt` — dataset links used

## Notes

- GA settings (population size, generations, mutation/crossover) are configurable constants in `ga_svm_feature_selection.py`.
- Results can vary slightly across runs due to stochastic GA operators, but a fixed random seed is used by default.

## Git ignore

`REPORT.md` is intentionally ignored and not pushed to GitHub.

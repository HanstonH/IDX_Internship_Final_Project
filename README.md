# Lending Club Loan Default Prediction

**English** | [Bahasa Indonesia](README.id.md)

Internship project: binary classification of Lending Club loans (2007–2014) to predict whether a loan will end as **good** (repaid) or **bad** (default / charged off).

---

## Problem

| Label | Meaning | Example `loan_status` values |
|-------|---------|------------------------------|
| **1 (good)** | Loan repaid successfully | Fully Paid |
| **0 (bad)** | Default or serious delinquency | Charged Off, Default, Late (31–120 days) |

The dataset is **imbalanced** (~79% good, ~21% bad after cleaning). Accuracy alone can look high while the model rarely flags risky loans. The notebook optimizes and reports **F1, precision, and recall for the bad class (0)**.

In-progress loans (`Current`, `In Grace Period`, etc.) are dropped so labels reflect **final** outcomes.

---

## Data

| File | Description |
|------|-------------|
| `loan_data_2007_2014.csv` | Raw Lending Club export (~466k rows). **Not in git** (see `.gitignore`). |
| `LCDataDictionary.xlsx` | Column definitions (optional reference). **Not in git**. |

Place `loan_data_2007_2014.csv` in this folder (`internship/`) before running the notebooks.

**After cleaning** (`CustomTransformer`): ~238k rows, **32 features** + `label`.

---

## Project structure

```
internship/
├── README.md              # This file (English)
├── README.id.md           # Indonesian version
├── main.ipynb             # End-to-end pipeline: clean → model → save
├── data_learning.ipynb    # Exploratory cleaning (reference; not required to run main)
├── LCDataDictionary.xlsx  # Data dictionary (local only)
├── loan_data_2007_2014.csv
└── saved_models/          # joblib artifacts after training (created by main.ipynb)
```

Parent repo: `requirements.txt` at `Part 2/requirements.txt`.

---

## Setup

From the repository root (`Part 2/`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install xgboost             # optional; grid search skips XGBoost if missing
pip install joblib seaborn      # used in main.ipynb (joblib for saves; seaborn for plots)
```

Start Jupyter from the `internship/` directory so relative paths resolve correctly:

```bash
cd internship
jupyter notebook main.ipynb
```

Run cells **top to bottom**. Grid search and MLP training can take a long time on CPU.

---

## Notebooks

### `main.ipynb` (primary deliverable)

1. **Load & clean** — `CustomTransformer` applies the same logic as `data_learning.ipynb` inside a scikit-learn–compatible transformer.
2. **Split** — Stratified 80/20 train/test (`random_state=42`).
3. **Preprocess** — `ColumnTransformer`: numeric imputation + `StandardScaler`; categorical imputation + `TargetEncoder` (binary target).
4. **EDA** — Correlation heatmaps on processed training data.
5. **Grid search** — Logistic Regression, Linear SVC, Random Forest, HistGradientBoosting, XGBoost (if installed); 3-fold stratified CV; scoring = **F1 for class 0**; `class_weight="balanced"` / `scale_pos_weight`.
6. **Save** — Artifacts under `saved_models/`.
7. **MLP** — TensorFlow/Keras feedforward net on the same processed matrices; early stopping; joblib save.
8. **Summary** — Indonesian narrative and infographics (last cells).

### `data_learning.ipynb`

Step-by-step exploratory data cleaning. **`main.ipynb` does not depend on running this file**; `CustomTransformer` mirrors its steps for reproducible pipelines.

---

## Pipeline overview

```
Raw CSV
  → CustomTransformer (clean, engineer features, map label)
  → train_test_split
  → preprocessor (fit on train only)
  → models (grid search + MLP)
  → saved_models/
```

### `CustomTransformer` (high level)

1. Drop index column and 100% empty columns  
2. Normalize blanks to NaN on text fields  
3. Parse `term`, dates, IDs  
4. Filter non-final statuses; map `loan_status` → `label`  
5. Feature engineering (`has_desc`, `emp_length`, `home_ownership`, `emp_pay_tier`, etc.)  
6. Encode month-since fields; handle `next_pymnt_d`  
7. Drop leakage / post-origination and sparse columns  
8. Final numeric encoding (epoch days, sentinels, flags)  
9. Fill `emp_length` and `annual_inc` NaNs with 0  

Leakage columns (e.g. `out_prncp`, `total_pymnt`, `last_pymnt_d`) are removed so the model only sees **origination-time** information.

### Modeling features (32)

- **Numeric (25):** all columns except the seven categorical ones below (e.g. `loan_amnt`, `int_rate`, `dti`, `fico_range_low`, …).  
- **Categorical (7):** `grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `emp_pay_tier`.

Default classification threshold: **0.5** on predicted probabilities (not tuned in the notebook).

---

## Models

| Model | Notes |
|-------|--------|
| Logistic Regression | Linear baseline, `class_weight="balanced"` |
| Linear SVC | Scalable linear margin classifier |
| Random Forest | Non-linear ensemble |
| HistGradientBoosting | sklearn gradient boosting |
| XGBoost | Optional; best CV/test F1 in sample runs |
| MLP (Keras) | 128 → 64 → 32 + dropout, sigmoid, `binary_crossentropy` |

**Class imbalance:** `class_weight="balanced"` (sklearn), XGBoost `scale_pos_weight`, Keras `class_weight` from training label counts.

---

## `saved_models/` artifacts

Created when you run the save cells in `main.ipynb`:

| File pattern | Contents |
|--------------|----------|
| `preprocessor.joblib` | Fitted `ColumnTransformer` |
| `{Model}_best_estimator.joblib` | Best estimator from grid search |
| `{Model}_gridsearch.joblib` | Full `GridSearchCV` object |
| `{Model}_pipeline.joblib` | `preprocess` + `classifier` for inference |
| `grid_search_summary.csv` | CV and test metrics per model |
| `loan_default_mlp.joblib` | Keras model + metadata |
| `mlp_test_metrics.joblib` | MLP test metrics dict |

Example inference with a saved sklearn pipeline:

```python
import joblib
import pandas as pd

pipe = joblib.load("saved_models/XGBoost_pipeline.joblib")
# X_raw: one row or many, same columns as X_train before preprocessing
# y_pred = pipe.predict(X_raw)
# proba = pipe.predict_proba(X_raw)[:, 1]  # P(good) if classifier supports predict_proba
```

Re-run `CustomTransformer` on new raw rows before passing data to a pipeline that expects pre-cleaned columns, or wrap cleaning + `*_pipeline.joblib` in your own outer pipeline.

---

## Example results (reference)

From a completed run (your numbers may differ slightly):

| Model | Test F1 (bad) | Test ROC-AUC |
|-------|----------------|--------------|
| XGBoost | ~0.497 | ~0.765 |
| HistGradientBoosting | ~0.493 | ~0.765 |
| MLP | ~0.483 | — |

Processed shapes: train **(190156, 32)**, test **(47539, 32)**.

---

## Design choices

- **No PCA** at 32 features — keeps interpretability.  
- **Target encoding** on categoricals (fit on train only).  
- **LinearSVC** instead of RBF SVC for scale on ~190k rows.  
- **Leakage avoidance** — central to which columns are dropped.  

`data_learning.ipynb` historically had a bug copying `emp_length` into `annual_inc`; `main.ipynb` correctly uses `annual_inc.fillna(0)`.

---

## License and data

Lending Club historical loan data is used for educational purposes. Check Lending Club / Kaggle terms for redistribution. Do not commit large CSV/XLSX files (see `.gitignore`).

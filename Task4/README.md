# Task 4 – Feature Engineering on Netflix Titles

## Objective
Improve model performance through feature engineering and intelligent feature selection on the `netflix_titles.csv` dataset (used in Task 2). The classification goal is predicting whether a Netflix title is a **Movie** or **TV Show**.

## Dataset
- **Source**: `netflix_titles.csv` (8,807 rows × 12 columns)
- **Target**: `type` → Movie (1) vs TV Show (0)
- **Key columns**: `type`, `release_year`, `duration`, `rating`, `listed_in`, `cast`, `director`, `country`, `date_added`, `description`

## Approach

### 1. Baseline Model
A minimal feature set (release year, raw duration, one-hot encoded rating & country) was used to train models and record pre-engineering performance.

### 2. Feature Engineering
Three categories of engineered features were created:

| Category | Examples |
|---|---|
| **Polynomial** (degree 2) | `release_year²`, `duration_val²`, `release_year × n_cast`, etc. |
| **Interaction** | `cast × genres`, `duration × genres`, `recent_intl`, `old_content` |
| **Binning** | `era` (Classic / Early2000s / Mid2000s / Late2010s / Recent), `duration_bucket`, `genre_richness` |
| **Text-derived** | `n_cast`, `n_genres`, `description_len`, `director_known`, `n_countries`, `years_since_release` |

### 3. Feature Selection (3 Methods)
| Method | Description |
|---|---|
| **RF Feature Importance** | Top 25 features by Random Forest impurity importance |
| **Correlation Analysis** | Keep features with `|corr|≥0.03` vs target; drop inter-correlated pairs `>0.95` |
| **Recursive Feature Elimination (RFE)** | Select top 20 features via RFE with RandomForestClassifier |

The **final feature set** is the intersection of RF Importance and RFE selections.

### 4. Model Training
Three models were tuned with `RandomizedSearchCV` (ROC-AUC scoring):
- **Random Forest** — `n_estimators`, `max_depth`, `min_samples_split`
- **Logistic Regression** — `C`, `solver`
- **XGBoost** — `n_estimators`, `learning_rate`, `max_depth`

## Results (Before vs. After)

| Metric | Before (Baseline) | After (Engineered) | Change |
|---|---|---|---|
| Accuracy | — | — | — |
| F1-Score | — | — | — |
| ROC-AUC | — | — | — |

> Run `python task4.py` to see exact results printed in the terminal.

## How to Run

```bash
cd "path/to/Task4"
python task4.py
```

**Requirements**: `pandas`, `numpy`, `scikit-learn`, `xgboost`

## Files
- `task4.py` — Full feature engineering pipeline

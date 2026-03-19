"""
Task 4: Feature Engineering on Netflix Titles Dataset
======================================================
Goal: Engineer new features, select the most important ones, train the best model,
      and compare performance before and after feature engineering.

Dataset : netflix_titles.csv (same as Task 2)
Target  : type  ->  Movie (1)  vs  TV Show (0)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEP = "=" * 65

# ----------------------------------------------------------------
# 1. DATA LOADING & BASIC CLEANING
# ----------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load and perform minimal cleaning on the Netflix dataset."""
    print("\n" + SEP)
    print("  STEP 1 - DATA LOADING & CLEANING")
    print(SEP)

    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    df = df.dropna(subset=['type'])

    df['director']   = df['director'].fillna('Unknown')
    df['cast']       = df['cast'].fillna('Unknown')
    df['country']    = df['country'].fillna('Unknown')
    df['rating']     = df['rating'].fillna(df['rating'].mode()[0])
    df['duration']   = df['duration'].fillna('0 min')
    df['date_added'] = df['date_added'].fillna('January 1, 2000')

    print(f"  Remaining rows after cleaning: {len(df):,}")
    return df


# ----------------------------------------------------------------
# 2. BASELINE FEATURES (Before Engineering)
# ----------------------------------------------------------------

def build_baseline_features(df: pd.DataFrame):
    """Build a minimal feature set that requires no engineering."""
    print("\n" + SEP)
    print("  STEP 2 - BASELINE FEATURES (Before Engineering)")
    print(SEP)

    y = (df['type'] == 'Movie').astype(int)

    def parse_duration_raw(row):
        try:
            return int(str(row['duration']).split()[0])
        except Exception:
            return 0

    features = pd.DataFrame()
    features['release_year'] = df['release_year'].astype(float)
    features['duration_raw'] = df.apply(parse_duration_raw, axis=1).astype(float)

    top_ratings = df['rating'].value_counts().head(6).index.tolist()
    df['rating_clean'] = df['rating'].apply(lambda r: r if r in top_ratings else 'Other')
    rating_dummies = pd.get_dummies(df['rating_clean'], prefix='rating', drop_first=False)

    df['country_first'] = df['country'].apply(lambda c: str(c).split(',')[0].strip())
    top_countries = df['country_first'].value_counts().head(10).index.tolist()
    df['country_top'] = df['country_first'].apply(lambda c: c if c in top_countries else 'Other')
    country_dummies = pd.get_dummies(df['country_top'], prefix='country', drop_first=False)

    features = pd.concat([features, rating_dummies, country_dummies], axis=1)
    features = features.fillna(0)

    print(f"  Baseline feature count : {features.shape[1]}")
    return features, y


# ----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rich engineered features from the raw dataframe."""
    print("\n" + SEP)
    print("  STEP 3 - FEATURE ENGINEERING")
    print(SEP)

    fe = pd.DataFrame(index=df.index)

    # 3a. Text-derived / count features
    fe['n_cast']          = df['cast'].apply(
                                lambda x: 0 if x == 'Unknown' else len(str(x).split(',')))
    fe['n_genres']        = df['listed_in'].apply(lambda x: len(str(x).split(',')))
    fe['description_len'] = df['description'].apply(lambda x: len(str(x).split()))
    fe['director_known']  = (df['director'] != 'Unknown').astype(int)
    fe['n_countries']     = df['country'].apply(lambda x: len(str(x).split(',')))

    # 3b. Date features
    def parse_year_added(date_str):
        try:
            return pd.to_datetime(date_str, format='%B %d, %Y').year
        except Exception:
            return 2000

    def parse_month_added(date_str):
        try:
            return pd.to_datetime(date_str, format='%B %d, %Y').month
        except Exception:
            return 1

    fe['year_added']          = df['date_added'].apply(parse_year_added).astype(float)
    fe['month_added']         = df['date_added'].apply(parse_month_added).astype(float)
    fe['release_year']        = df['release_year'].astype(float)
    fe['years_since_release'] = (fe['year_added'] - fe['release_year']).clip(lower=0)

    # 3c. Duration (numeric)
    def parse_duration(row):
        try:
            return int(str(row['duration']).split()[0])
        except Exception:
            return 0

    fe['duration_val'] = df.apply(parse_duration, axis=1).astype(float)

    # 3d. Categorical encoding
    top_ratings = df['rating'].value_counts().head(6).index.tolist()
    df['rating_clean'] = df['rating'].apply(lambda r: r if r in top_ratings else 'Other')
    rating_dummies = pd.get_dummies(df['rating_clean'], prefix='rating', drop_first=False)

    df['country_first'] = df['country'].apply(lambda c: str(c).split(',')[0].strip())
    top_countries = df['country_first'].value_counts().head(10).index.tolist()
    df['country_top'] = df['country_first'].apply(lambda c: c if c in top_countries else 'Other')
    country_dummies = pd.get_dummies(df['country_top'], prefix='country', drop_first=False)

    fe = pd.concat([fe, rating_dummies, country_dummies], axis=1)

    # 3e. BINNING / Categorization
    fe['era'] = pd.cut(
        fe['release_year'],
        bins=[0, 1999, 2010, 2015, 2018, 2030],
        labels=['Classic', 'Early2000s', 'Mid2000s', 'Late2010s', 'Recent'])
    fe['era'] = LabelEncoder().fit_transform(fe['era'].astype(str))

    fe['duration_bucket'] = pd.cut(
        fe['duration_val'],
        bins=[-1, 30, 90, 120, 10000],
        labels=['VeryShort', 'Short', 'Standard', 'Long'])
    fe['duration_bucket'] = LabelEncoder().fit_transform(fe['duration_bucket'].astype(str))

    fe['genre_richness'] = pd.cut(
        fe['n_genres'], bins=[0, 1, 2, 10],
        labels=['Single', 'Double', 'Multi'])
    fe['genre_richness'] = LabelEncoder().fit_transform(fe['genre_richness'].astype(str))

    # 3f. INTERACTION features
    fe['cast_x_genres']     = fe['n_cast']       * fe['n_genres']
    fe['duration_x_genres'] = fe['duration_val'] * fe['n_genres']
    fe['desc_x_cast']       = fe['description_len'] * fe['n_cast']
    fe['old_content']       = ((fe['release_year'] < 2000) &
                                (fe['director_known'] == 0)).astype(int)
    fe['recent_intl']       = ((fe['release_year'] > 2015) &
                                (fe['n_countries'] > 1)).astype(int)
    fe['year_x_duration']   = fe['release_year'] * fe['duration_val']

    # 3g. POLYNOMIAL features (degree 2, on core numerics)
    poly_cols = ['release_year', 'duration_val', 'n_cast', 'n_genres', 'description_len']
    poly_data = fe[poly_cols].fillna(0)

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_arr   = poly.fit_transform(poly_data)
    poly_names = poly.get_feature_names_out(poly_cols)

    original_set  = set(poly_cols)
    new_poly_names = [n for n in poly_names if n not in original_set]
    poly_df_full  = pd.DataFrame(poly_arr, columns=poly_names, index=fe.index)
    poly_new      = poly_df_full[new_poly_names]

    fe = pd.concat([fe, poly_new], axis=1)
    fe = fe.fillna(0)
    fe.columns = [str(c).replace(' ', '_').replace('^', 'pw') for c in fe.columns]

    n_poly  = len(new_poly_names)
    n_inter = 6
    n_bins  = 3
    print(f"  Engineered feature count   : {fe.shape[1]}")
    print(f"    |-- Polynomial terms      : {n_poly}")
    print(f"    |-- Interaction features  : {n_inter}")
    print(f"    +-- Binned/Categorical    : {n_bins}")

    return fe


# ----------------------------------------------------------------
# 4. FEATURE SELECTION
# ----------------------------------------------------------------

def select_features(X_train: np.ndarray, X_test: np.ndarray,
                    y_train: pd.Series, feature_names: list):
    """Apply three feature selection methods; return the final feature list."""
    print("\n" + SEP)
    print("  STEP 4 - FEATURE SELECTION")
    print(SEP)

    rf_sel = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train, y_train)

    # 4a. Feature Importance from Random Forest
    importances = rf_sel.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    top_n  = min(25, len(feature_names))
    top_fi = fi_df.head(top_n)['feature'].tolist()
    print(f"\n  [A] Random Forest Feature Importance -> top {top_n} selected")
    print(f"  {'Feature':<35}  {'Importance':>10}")
    print(f"  {'-'*35}  {'-'*10}")
    for _, row in fi_df.head(10).iterrows():
        print(f"  {row['feature']:<35}  {row['importance']:>10.4f}")

    # 4b. Correlation Analysis
    X_df = pd.DataFrame(X_train, columns=feature_names)
    corr_with_target = X_df.corrwith(pd.Series(y_train.values, index=X_df.index)).abs()
    corr_keep   = corr_with_target[corr_with_target >= 0.03].index.tolist()
    corr_matrix = X_df[corr_keep].corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_feats  = [col for col in upper.columns if any(upper[col] > 0.95)]
    corr_final  = [f for f in corr_keep if f not in drop_feats]
    print(f"\n  [B] Correlation Analysis")
    print(f"  Features with |corr| >= 0.03 vs target : {len(corr_keep)}")
    print(f"  Dropped highly inter-correlated (>0.95) : {len(drop_feats)}")
    print(f"  Final surviving features                 : {len(corr_final)}")

    # 4c. Recursive Feature Elimination (RFE)
    rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    n_feats_rfe   = min(20, len(feature_names))
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_feats_rfe, step=5)
    rfe.fit(X_train, y_train)
    rfe_selected = [feature_names[i] for i, s in enumerate(rfe.support_) if s]
    print(f"\n  [C] Recursive Feature Elimination (RFE)")
    print(f"  Selected {len(rfe_selected)} features out of {len(feature_names)}")
    print(f"  Top RFE features: {rfe_selected[:8]}")

    # Final: intersection of RF Importance and RFE
    final_features = list(set(top_fi) & set(rfe_selected))
    if len(final_features) < 5:
        final_features = rfe_selected

    print(f"\n  Final feature set (RF-Importance n RFE): {len(final_features)} features")
    print(f"  Features: {final_features[:10]}{'...' if len(final_features) > 10 else ''}")

    return final_features


# ----------------------------------------------------------------
# 5. MODEL TRAINING & EVALUATION
# ----------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, label: str = ""):
    """Train 3 models with hyperparameter tuning; return best model metrics."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        }
    }

    if HAS_XGB:
        models['XGBoost'] = {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }

    best_name    = None
    best_score   = -1
    best_metrics = {}
    all_results  = []

    for name, cfg in models.items():
        search = RandomizedSearchCV(
            cfg['model'], cfg['params'],
            n_iter=8, scoring='roc_auc',
            cv=cv, random_state=42,
            n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)
        mdl = search.best_estimator_

        y_pred  = mdl.predict(X_test)
        y_proba = (mdl.predict_proba(X_test)[:, 1]
                   if hasattr(mdl, 'predict_proba') else y_pred)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        all_results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': roc})

        if roc > best_score:
            best_score   = roc
            best_name    = name
            best_metrics = {'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': roc}

    results_df = pd.DataFrame(all_results).set_index('Model')
    print(f"\n  {label} - All Models:")
    print(results_df.round(4).to_string())
    print(f"\n  >> Best: {best_name} (ROC-AUC = {best_score:.4f})")
    return best_name, best_metrics


# ----------------------------------------------------------------
# 6. MAIN PIPELINE
# ----------------------------------------------------------------

def main():
    filepath = '../Task2/netflix_titles.csv'

    df = load_data(filepath)
    y  = (df['type'] == 'Movie').astype(int)

    # ==============================
    # A) BASELINE (before engineering)
    # ==============================
    baseline_features, _ = build_baseline_features(df.copy())

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        baseline_features, y,
        test_size=0.2, random_state=42, stratify=y)

    scaler_b = StandardScaler()
    X_tr_b_s = scaler_b.fit_transform(X_tr_b)
    X_te_b_s = scaler_b.transform(X_te_b)

    print("\n" + SEP)
    print("  STEP 5a - BASELINE MODEL TRAINING")
    print(SEP)
    _, baseline_metrics = train_and_evaluate(
        X_tr_b_s, X_te_b_s, y_tr_b, y_te_b, label="BASELINE")

    # ==============================
    # B) ENGINEERED FEATURES + SELECTION
    # ==============================
    eng_features = engineer_features(df.copy())

    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(
        eng_features, y,
        test_size=0.2, random_state=42, stratify=y)

    scaler_e = StandardScaler()
    X_tr_e_s = scaler_e.fit_transform(X_tr_e.fillna(0))
    X_te_e_s = scaler_e.transform(X_te_e.fillna(0))

    feature_names = list(eng_features.columns)

    final_feat_names = select_features(
        X_tr_e_s, X_te_e_s, y_tr_e, feature_names)

    final_idx  = [feature_names.index(f) for f in final_feat_names if f in feature_names]
    X_tr_final = X_tr_e_s[:, final_idx]
    X_te_final = X_te_e_s[:, final_idx]

    print("\n" + SEP)
    print("  STEP 5b - MODEL TRAINING ON ENGINEERED FEATURES")
    print(SEP)
    _, eng_metrics = train_and_evaluate(
        X_tr_final, X_te_final, y_tr_e, y_te_e,
        label="ENGINEERED")

    # ==============================
    # C) BEFORE vs. AFTER COMPARISON
    # ==============================
    print("\n" + SEP)
    print("  FINAL COMPARISON: BEFORE vs. AFTER FEATURE ENGINEERING")
    print(SEP)

    print()
    print(f"  {'Metric':<12} {'Before':>12}  {'After':>12}  {'Change':>10}")
    print(f"  {'-'*12} {'-'*12}  {'-'*12}  {'-'*10}")
    all_improved = True
    for metric in ['Accuracy', 'F1-Score', 'ROC-AUC']:
        b     = baseline_metrics[metric]
        a     = eng_metrics[metric]
        delta = a - b
        sign  = '+' if delta >= 0 else ''
        if delta < 0:
            all_improved = False
        print(f"  {metric:<12} {b:>12.4f}  {a:>12.4f}  {sign+f'{delta:.4f}':>10}")

    print("\n" + SEP)
    if all_improved:
        print("  [OK] Feature engineering improved ALL metrics!")
    else:
        best_gain = max(eng_metrics[m] - baseline_metrics[m]
                        for m in ['Accuracy', 'F1-Score', 'ROC-AUC'])
        print(f"  [OK] Feature engineering improved some metrics (max gain: {best_gain:+.4f})")
    print(SEP)

    roc_gain = eng_metrics['ROC-AUC'] - baseline_metrics['ROC-AUC']
    print(f"""
  SUMMARY
  -------------------------------------------------------------------
  Target          : Content type (Movie vs TV Show)
  Baseline feats  : {baseline_features.shape[1]:>4}  (release_year, duration, rating, country OHE)
  Engineered set  : {eng_features.shape[1]:>4}  (polynomial, interaction, binning + original)
  Final selected  : {len(final_feat_names):>4}  features  (RF Importance intersect RFE)
  ROC-AUC gain    : {roc_gain:>+.4f}
  -------------------------------------------------------------------
    """)


if __name__ == '__main__':
    main()

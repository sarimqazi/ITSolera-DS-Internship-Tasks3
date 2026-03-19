import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Drop Loan_ID as it's not useful
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        
    df = df.dropna(subset=['Loan_Status'])
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    X = df.drop('Loan_Status', axis=1)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    # Preprocessing pipelines for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    print("Fitting preprocessor...")
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    return X_train_pre, X_test_pre, y_train, y_test

def tune_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Models and their hyperparameter grids
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
    }

    results = []

    print("Starting hyperparameter tuning and evaluation...\n")
    best_overall_model = None
    best_overall_score = -1
    best_overall_name = ""
    
    for name, config in models.items():
        print(f"Tuning {name}...")
        search = RandomizedSearchCV(
            config['model'],
            config['params'],
            n_iter=10,
            scoring='roc_auc',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Best Params': search.best_params_
        })

        if roc_auc > best_overall_score:
            best_overall_score = roc_auc
            best_overall_model = best_model
            best_overall_name = name

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    # Print metrics without Best Params column for simpler table
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    print(results_df[display_cols].to_string(index=False))
    print("="*80 + "\n")
    
    print("BEST PARAMS:")
    for res in results:
        print(f"{res['Model']}: {res['Best Params']}")
    
    return results_df, best_overall_name, best_overall_score

def main():
    filepath = '../Task1/loan_prediction.csv.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    _, best_name, best_score = tune_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*80)
    print(f"BEST MODEL SELECTION: {best_name}")
    print("="*80)
    print(f"Justification: {best_name} achieved the highest ROC-AUC score ({best_score:.4f}) "
          "on the holdout test set. ROC-AUC is a strong indicator of the "
          "model's ability to distinguish between approved and rejected loan applications "
          "regardless of class imbalance thresholds.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def main():
    print("Starting Model Training Pipeline...")
    
    # 1. Load Data
    data_path = 'loan_prediction.csv.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Preprocess Data
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        
    target = 'Loan_Status'
    df = df.dropna(subset=[target])
    
    features = [c for c in df.columns if c != target]
    numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object']).columns.tolist()
    
    X = df[features]
    y = df[target]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
        
    X_processed = preprocessor.fit_transform(X)
    
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_features_encoded)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # 4. Train Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    accuracies = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy={best_accuracy:.4f}")
    
    # 5. Save Artifacts
    artifacts = {
        'preprocessor': preprocessor,
        'model': best_model,
        'model_name': best_model_name,
        'target_encoder': le,
        'accuracy': best_accuracy,
        'feature_names': all_feature_names,
        'accuracies': accuracies,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    joblib.dump(artifacts, 'best_model.pkl')
    print("Model and preprocessors saved to best_model.pkl")

if __name__ == "__main__":
    main()

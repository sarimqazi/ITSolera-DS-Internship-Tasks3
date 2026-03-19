# Task 5: Complete ML System with Interactive GUI

## Objective
The objective of this task is to build a complete, end-to-end Machine Learning system using the Loan Prediction dataset and deploy it with an interactive Graphical User Interface (GUI) for generating real-time predictions.

## Approach
1. **Data Preprocessing & Model Selection** (`train_model.py`):
   - Loaded and preprocessed the `loan_prediction` dataset.
   - Handled missing values using statistical imputation (median for numerical variables, most frequent for categorical variables).
   - Applied One-Hot Encoding to categorical inputs and scaled numerical features using an `sklearn` Pipeline and `ColumnTransformer`.
   - Trained three distinct classification algorithms:
     - Logistic Regression
     - Random Forest
     - XGBoost
   - Selected the best-performing model based on validation metrics.
   - Exported the best model alongside its preprocessor weights to `best_model.pkl` using `joblib`.

2. **Interactive Streamlit Web Dashboard** (`app.py`):
   - Built a real-time web interface using Streamlit (`app.py`).
   - Sourced dynamic input values (e.g., Applicant Income, Loan Term, Credit History) from a sidebar form.
   - Structured three functional interactive tabs:
     - **Prediction**: Calculates loan approval probability and renders an actionable outcome directly to the user.
     - **Dataset Preview**: Grants users the ability to investigate raw dataset schemas.
     - **Dashboard Info**: Showcases dynamic accuracy comparisons and visual feature importances of the underlying trained model architecture.

## Results
- **Winning Model**: Logistic Regression performed the highest with heavily robust outcomes, reaching an accuracy of ~78.86%.
- The comprehensive preprocessing data transformations handled previously unseen Streamlit GUI inputs seamlessly.
- An interactive, modern, browser-hosted GUI application was successfully linked up with the backend serialized ML artifacts. 

## How to Run the App
To start the Streamlit predictor dashboard, open your terminal inside the `Task1` directory and initialize the server with:
```bash
streamlit run app.py
```

# Task 3: Model Evaluation and Hyperparameter Tuning

## Objective
The objective of this task is to improve model performance and systematically evaluate multiple machine learning models.

## Dataset
This task utilizes the **Loan Prediction Dataset** (`loan_prediction.csv.csv`) previously cleaned and explored in Task 1. The target variable is `Loan_Status`, which dictates whether a loan is approved or rejected.

## Approach
1. **Data Preprocessing Pipeline**: Built a consistent `ColumnTransformer` pipeline that handles missing values and scales numerical features while one-hot encoding categorical features.
2. **Models Trained**:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
3. **Hyperparameter Tuning**: Utilized `RandomizedSearchCV` across a predefined hyperparameter grid to find the optimal configurations for each algorithm.
4. **Evaluation Metrics**: Models were evaluated on a 20% hold-out test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Results
The trained and tuned models yielded the following results on the test data:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.7886 | 0.7596 | 0.9875 | 0.8587 | **0.7494** |
| **Random Forest** | 0.7886 | 0.7596 | 0.9875 | 0.8587 | 0.7422 |
| **XGBoost** | 0.7724 | 0.7653 | 0.9375 | 0.8427 | 0.7488 |

### Best Model Selection
**Logistic Regression** was selected as the best model. 
- **Justification:** While its Accuracy, Precision, Recall, and F1-Score were identical to the Random Forest model, Logistic Regression achieved the highest **ROC-AUC score (0.7494)**. ROC-AUC is a strong indicator of the model's ability to distinguish between approved and rejected loan applications regardless of class imbalance thresholds. Furthermore, Logistic Regression is computationally simpler and highly interpretable compared to complex ensemble algorithms.

## How to Run
Navigate to the `Task3` directory and run the evaluation script:
```bash
cd Task3
python task3.py
```

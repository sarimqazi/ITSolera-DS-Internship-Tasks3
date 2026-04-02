import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# Apply dark mode style
st.set_page_config(page_title="Loan Prediction GUI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
        .reportview-container {
            background: #1e1e1e;
        }
        .sidebar .sidebar-content {
            background: #2b2b2b;
        }
        .Widget>label {
            color: white;
            font-family: monospace;
        }
        .stButton>button {
            color: #4F8BF9;
            border-radius: 50px;
            height: 3em;
            width: 100%;
            border: 2px solid #4F8BF9;
            background-color: transparent;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            color: white;
            background-color: #4F8BF9;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Intelligent Loan Prediction System")
st.markdown("Use this ML-powered GUI to predict loan approvals automatically.")

@st.cache_resource # # cache so it doesn't reload every time
def load_assets():
    model_path = 'best_model.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

assets = load_assets()

if assets is None:
    st.error("Model artifacts not found. Please run `train_model.py` first.")
    st.stop()

preprocessor = assets['preprocessor']
model = assets['model']
target_encoder = assets['target_encoder']
model_name = assets['model_name']
acc = assets['accuracy']
all_feature_names = assets['feature_names']

# Inputs
st.sidebar.header("Applicant Information")
    
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
prop_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

app_income = st.sidebar.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
coapp_income = st.sidebar.number_input("Co-Applicant Income ($)", min_value=0, value=0, step=100)
loan_amt = st.sidebar.number_input("Loan Amount (Thousands)", min_value=0, value=150, step=10)
loan_term = st.sidebar.number_input("Loan Amount Term (Days)", min_value=12, value=360, step=12)
credit_hist = st.sidebar.selectbox("Credit History", [1.0, 0.0])

input_data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'Property_Area': prop_area,
    'ApplicantIncome': app_income,
    'CoapplicantIncome': coapp_income,
    'LoanAmount': loan_amt,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_hist
}

input_df = pd.DataFrame([input_data])

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset Preview", "📈 Dashboard Info"])

with tab1:
    st.subheader("Your Input Profile")
    st.dataframe(input_df, use_container_width=True)
    
    if st.button("Predict Loan Status"):
        with st.spinner("Processing Application..."):
            X_proc = preprocessor.transform(input_df)
            pred = model.predict(X_proc)
            prob = model.predict_proba(X_proc)[0]
            
            pred_class = target_encoder.inverse_transform(pred)[0]
            confidence = np.max(prob) * 100
            
            if pred_class == 'Y':
                st.success(f"🎉 **APPROVED** with {confidence:.1f}% confidence!")
            else:
                st.error(f"❌ **REJECTED** with {confidence:.1f}% confidence.")

with tab2:
    st.subheader("Dataset Preview")
    data_path = 'loan_prediction.csv.csv'
    if os.path.exists(data_path):
        df_preview = pd.read_csv(data_path)
        st.dataframe(df_preview.head(50), use_container_width=True)
    else:
        st.write("Dataset file not available.")

with tab3:
    st.subheader("Model Information")
    st.markdown(f"**Winning Model**: {model_name}")
    st.markdown(f"**Test Set Accuracy**: {acc*100:.2f}%")
    
    st.write("### Comparison")
    acc_df = pd.DataFrame(list(assets['accuracies'].items()), columns=['Model', 'Accuracy'])
    acc_df['Accuracy'] = acc_df['Accuracy'] * 100
    st.bar_chart(acc_df.set_index('Model'))
    
    if hasattr(model, 'feature_importances_'):
        st.write("### Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(10, len(all_feature_names))
        top_features = [all_feature_names[i] for i in indices[:top_n]]
        top_importances = importances[indices][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        ax.bar(top_features, top_importances, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    elif hasattr(model, 'coef_'):
        st.write("### Feature Importance (Coefficients)")
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]
        
        top_n = min(10, len(all_feature_names))
        top_features = [all_feature_names[i] for i in indices[:top_n]]
        top_importances = importances[indices][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.magma(np.linspace(0, 1, top_n))
        ax.bar(top_features, top_importances, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS — clean card-based look
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .title-container {
        text-align: center;
        padding: 1.2rem 0 0.5rem 0;
    }
    .title-container h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .title-container p {
        color: #9aa0a6;
        font-size: 1rem;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 700;
        font-size: 1.05rem;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99,102,241,0.35);
    }

    .result-card {
        border-radius: 16px;
        padding: 1.6rem;
        margin-top: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .churn-high { background: rgba(239, 68, 68, 0.12); border-color: rgba(239,68,68,0.35); }
    .churn-low  { background: rgba(34, 197, 94, 0.12); border-color: rgba(34,197,94,0.35); }

    .result-label { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.3rem; }
    .result-sub { color: #9aa0a6; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load model + scaler (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("churn_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="title-container">
    <h1>📉 Customer Churn Predictor</h1>
    <p>Estimate the probability a bank customer will leave, using an ANN model.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_ok:
    st.error(
        "Model files not found. Make sure `churn_model.keras` and `scaler.pkl` "
        "are in the same folder as this app.\n\nDetails: " + load_error
    )
    st.stop()

# ----------------------------------------------------------------------
# Sidebar — inputs
# ----------------------------------------------------------------------
st.sidebar.header("Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.radio("Gender", ["Female", "Male"], horizontal=True)
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure (years with bank)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance", min_value=0.0, value=60000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=1)
has_cr_card = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
is_active = st.sidebar.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

predict_clicked = st.sidebar.button("🔮 Predict Churn")

# ----------------------------------------------------------------------
# Main panel — summary + prediction
# ----------------------------------------------------------------------
geo_map = {"France": 0, "Spain": 1, "Germany": 2}
gender_map = {"Female": 0, "Male": 1}
yesno_map = {"Yes": 1, "No": 0}

input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geo_map[geography],
    "Gender": gender_map[gender],
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": yesno_map[has_cr_card],
    "IsActiveMember": yesno_map[is_active],
    "EstimatedSalary": estimated_salary,
}])

with st.expander("View input summary"):
    st.dataframe(input_df, use_container_width=True)

if predict_clicked:
    scaled_input = scaler.transform(input_df)
    probability = float(model.predict(scaled_input, verbose=0)[0][0])
    will_churn = probability > 0.5

    css_class = "churn-high" if will_churn else "churn-low"
    label = "⚠️ Likely to Churn" if will_churn else "✅ Likely to Stay"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">{label}</div>
        <div class="result-sub">Churn probability</div>
        <h2 style="margin-top:0.3rem;">{probability:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        "This estimate comes from a neural network trained on historical bank "
        "customer data. Use it as a decision-support signal, not a guarantee."
    )
else:
    st.info("Fill in the customer details in the sidebar, then click **Predict Churn**.")
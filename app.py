import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Advanced Custom CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Animated Dark Mesh Background */
    .stApp {
        background: linear-gradient(-45deg, #090d16, #111827, #0f172a, #1e1b4b);
        background-size: 400% 400%;
        animation: gradientBg 18s ease infinite;
        color: #f8fafc;
    }

    @keyframes gradientBg {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Keyframe Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseRed {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* Header Container */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 9999px;
        color: #818cf8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.2rem;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.12);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Custom Streamlit Button */
    .stButton>button {
        width: 100%;
        border-radius: 14px;
        height: 3.4rem;
        font-weight: 700;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        background-size: 200% auto;
        color: white;
        border: none;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .stButton>button:hover {
        background-position: right center;
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 28px rgba(139, 92, 246, 0.5);
    }

    /* Result Glass Card Styling */
    .result-card {
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        animation: fadeInUp 0.5s ease-out forwards;
    }
    .churn-high {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        animation: pulseRed 2s infinite, fadeInUp 0.5s ease-out forwards;
    }
    .churn-low {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        animation: pulseGreen 2s infinite, fadeInUp 0.5s ease-out forwards;
    }
    .result-status {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.3rem;
    }
    .result-val {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.2rem 0;
        line-height: 1;
    }

    /* Sleek Input Controls Polish */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    div[data-baseweb="select"]:hover, div[data-baseweb="input"]:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load Model & Scaler
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
# Header Section
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🧠 Neural Network Driven</div>
    <div class="hero-title">Customer Churn Intelligence</div>
    <div class="hero-sub">Predict customer retention risks instantly with precise deep learning analytics.</div>
</div>
""", unsafe_allow_html=True)

if not artifacts_ok:
    st.error(f"⚠️ **Model Artifact Error:** Could not find necessary model files (`churn_model.keras` or `scaler.pkl`).\n\n`Details: {load_error}`")
    st.stop()

# ----------------------------------------------------------------------
# Main Dashboard Layout
# ----------------------------------------------------------------------
col_left, col_right = st.columns(2, gap="medium")

with col_left:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">👤 Customer Profile</div>
    """, unsafe_allow_html=True)
    
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
    is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">💳 Financial Metrics</div>
    """, unsafe_allow_html=True)
    
    credit_score = st.slider("Credit Score", 300, 900, 650)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Calculate Churn Risk")

# Data Mappings
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

# ----------------------------------------------------------------------
# Prediction Results & Visualizations
# ----------------------------------------------------------------------
if predict_clicked:
    scaled_input = scaler.transform(input_df)
    probability = float(model.predict(scaled_input, verbose=0)[0][0])
    will_churn = probability > 0.5

    css_class = "churn-high" if will_churn else "churn-low"
    label = "⚠️ Likely to Churn" if will_churn else "✅ Likely to Stay"
    text_color = "#ef4444" if will_churn else "#22c55e"

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

    # Result Header Card
    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-status" style="color: {text_color};">{label}</div>
        <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Calculated Probability</div>
        <div class="result-val" style="color: {text_color};">{probability:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CHART ROW 1: Gauge & Benchmark
    # ------------------------------------------------------------------
    chart_col1, chart_col2 = st.columns(2, gap="medium")

    # 1. Risk Gauge Chart
    with chart_col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">⏱ Risk Probability Gauge</div>
        """, unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'color': text_color, 'size': 38, 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': text_color, 'thickness': 0.25},
                'bgcolor': "rgba(15, 23, 42, 0.4)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(34, 197, 94, 0.2)'},
                    {'range': [35, 65], 'color': 'rgba(234, 179, 8, 0.2)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc", 'family': 'Plus Jakarta Sans'},
            height=260,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Benchmark Comparison Bar Chart
    with chart_col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📊 Customer vs Bank Average</div>
        """, unsafe_allow_html=True)
        
        metrics = ['Credit Score', 'Age', 'Balance ($k)', 'Salary ($k)']
        customer_vals = [credit_score, age, balance / 1000, estimated_salary / 1000]
        avg_vals = [650, 38, 76.4, 100.0]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=metrics, y=customer_vals, name='Current Customer',
            marker_color='#818cf8', opacity=0.9
        ))
        fig_bar.add_trace(go.Bar(
            x=metrics, y=avg_vals, name='Bank Average',
            marker_color='rgba(255, 255, 255, 0.2)'
        ))

        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc", 'family': 'Plus Jakarta Sans'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CHART ROW 2: Radar Chart & Landscape Scatter Plot
    # ------------------------------------------------------------------
    chart_col3, chart_col4 = st.columns(2, gap="medium")

    # 3. Behavioral Profile Radar Chart
    with chart_col3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🕸 Customer Behavioral Radar</div>
        """, unsafe_allow_html=True)

        categories = ['Credit Rating', 'Age Index', 'Tenure', 'Products', 'Activity']
        
        # Scaling customer values onto 0-100 radar axis
        c_credit = ((credit_score - 300) / 600) * 100
        c_age = ((age - 18) / 74) * 100
        c_tenure = (tenure / 10) * 100
        c_prod = (num_products / 4) * 100
        c_act = 100 if is_active == "Yes" else 10

        radar_vals = [c_credit, c_age, c_tenure, c_prod, c_act]
        high_risk_benchmark = [50, 65, 30, 75, 20] # Typical high churn profile

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals,
            theta=categories,
            fill='toself',
            name='Current Customer',
            line_color='#6366f1',
            fillcolor='rgba(99, 102, 241, 0.35)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=high_risk_benchmark,
            theta=categories,
            fill='toself',
            name='High Churn Risk Threshold',
            line_color='#ef4444',
            fillcolor='rgba(239, 68, 68, 0.15)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='rgba(255,255,255,0.1)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc", 'family': 'Plus Jakarta Sans'},
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=300,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Customer Landscape Position Scatter Plot
    with chart_col4:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📍 Portfolio Position Map (Age vs Balance)</div>
        """, unsafe_allow_html=True)

        # Synthetic representative sample points for context background
        np.random.seed(42)
        sample_ages = np.random.randint(20, 75, 40)
        sample_balances = np.random.randint(10000, 180000, 40)
        
        fig_scatter = go.Figure()
        
        # Background Customer Pool
        fig_scatter.add_trace(go.Scatter(
            x=sample_ages,
            y=sample_balances,
            mode='markers',
            name='Existing Customers',
            marker=dict(size=8, color='rgba(255, 255, 255, 0.25)', symbol='circle')
        ))
        
        # Highlighted Current Customer
        fig_scatter.add_trace(go.Scatter(
            x=[age],
            y=[balance],
            mode='markers+text',
            name='Current Customer',
            text=["📍 THIS CUSTOMER"],
            textposition="top center",
            textfont=dict(color=text_color, size=12, family="Plus Jakarta Sans"),
            marker=dict(size=18, color=text_color, symbol='star', line=dict(width=2, color='#ffffff'))
        ))

        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc", 'family': 'Plus Jakarta Sans'},
            xaxis=dict(title="Age", gridcolor='rgba(255, 255, 255, 0.05)', range=[18, 80]),
            yaxis=dict(title="Balance ($)", gridcolor='rgba(255, 255, 255, 0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Raw Vector Expander
    with st.expander("🔍 View Raw Input Vector"):
        st.dataframe(input_df, use_container_width=True)

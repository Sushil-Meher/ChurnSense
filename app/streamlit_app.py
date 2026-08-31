import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import json
import joblib
import pandas as pd
import streamlit as st

from src.features import create_features
from src.predict import predict_customer
from src.explain import explain_prediction,clean_feature_name

st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .section-card {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .recommendation {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Load model and threshold
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/churnsense_catboost.pkl"
    )


@st.cache_data
def load_threshold():
    with open("models/threshold.json", "r") as f:
        return json.load(f)["threshold"]


model = load_model()
threshold = load_threshold()


# -----------------------------
# Page
# -----------------------------

st.markdown(
    '<div class="main-title">📊 ChurnSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer churn prediction and retention intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.success(
    f"✅ CatBoost model loaded • Decision threshold: {threshold:.2f}"
)

st.info(
    "Enter a customer's current profile, service usage, and billing details "
    "to estimate churn risk and receive targeted retention actions."
)

st.write(
    "Enter customer information to estimate churn risk "
    "and generate retention recommendations."
)

st.success(
    f"Model loaded successfully • Decision threshold: {threshold:.2f}"
)






st.divider()

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )




st.divider()

st.header("Services")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )





st.divider()
st.header("💳 Billing")

col1, col2 = st.columns(2)

with col1:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )

with col2:

    st.caption(
        "Total historical charges for the customer."
    )


customer_data = {
    "customerID": "STREAMLIT_USER",
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

customer_df = pd.DataFrame([customer_data])




st.divider()

if st.button(
    "🔍 Predict Churn Risk",
    type="primary",
    use_container_width=True
):

    # Apply feature engineering
    customer_features = create_features(
        customer_df
    )

    # Remove non-model columns
    customer_features = customer_features.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    # Generate prediction
    result = predict_customer(
        model,
        customer_features
    )

    probability = result["churn_probability"] * 100
    risk = result["risk"]


    # -----------------------------
    # Prediction Results
    # -----------------------------

    st.divider()

    st.header("📈 Churn Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk
        )

    with col3:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


    # Probability bar
    st.progress(
        min(probability / 100, 1.0)
    )


    # Risk message
    if risk == "HIGH":

        st.error(
            "⚠️ High churn risk — immediate retention intervention recommended."
        )

    elif risk == "MEDIUM":

        st.warning(
            "⚠️ Medium churn risk — proactive engagement recommended."
        )

    else:

        st.success(
            "✅ Low churn risk — customer appears relatively stable."
        )


    # -----------------------------
    # SHAP Explanation
    # -----------------------------

    st.subheader("🔎 Why this prediction?")

    feature_names, customer_shap_values, customer_feature_values = (
        explain_prediction(
            model,
            customer_features
        )
    )

    explanation_df = pd.DataFrame({
    "Feature": [
        clean_feature_name(name)
        for name in feature_names
    ],
    "SHAP_Value": customer_shap_values,
    "Value": customer_feature_values
    })


    # Factors increasing churn risk
    risk_factors = (
        explanation_df[
            explanation_df["SHAP_Value"] > 0
        ]
        .sort_values(
            "SHAP_Value",
            ascending=False
        )
        .head(5)
    )


    # Factors reducing churn risk
    protective_factors = (
        explanation_df[
            explanation_df["SHAP_Value"] < 0
        ]
        .sort_values(
            "SHAP_Value"
        )
        .head(5)
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### 🔴 Increasing Churn Risk")

        if len(risk_factors) == 0:
            st.write("No strong risk factors identified.")
        else:
            for _, row in risk_factors.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"({row['SHAP_Value']:.3f})"
                )


    with col2:

        st.markdown("### 🟢 Reducing Churn Risk")

        if len(protective_factors) == 0:
            st.write("No strong protective factors identified.")
        else:
            for _, row in protective_factors.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"({row['SHAP_Value']:.3f})"
                )

        


    # -----------------------------
    # Recommended Actions
    # -----------------------------

    st.subheader(
        "🎯 Recommended Retention Actions"
    )

    for recommendation in result["recommendations"]:

        st.markdown(
            f"""
            <div class="recommendation">
                ✅ {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )
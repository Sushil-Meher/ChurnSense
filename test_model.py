import json

import joblib
import pandas as pd

from src.features import create_features
from src.predict import predict_customer


# Load the saved model
model = joblib.load(
    "models/churnsense_catboost.pkl"
)

# Load the saved decision threshold
with open("models/threshold.json", "r") as f:
    threshold_data = json.load(f)

print(
    "Loaded threshold:",
    threshold_data["threshold"]
)


# Load raw customer data
df = pd.read_csv(
    "data/telco_churn.csv"
)


# Apply the same feature engineering used during training
df = create_features(df)


# Select one customer for testing
customer = df.drop(
    columns=["customerID", "Churn"]
).iloc[[0]]


# Generate prediction
prediction = predict_customer(
    model,
    customer
)


# Display prediction
print("\nPrediction:")
print(prediction)
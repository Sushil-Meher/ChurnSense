import shap


def explain_prediction(model_pipeline, customer):
    """
    Generate SHAP feature contributions for one customer.
    """

    # Get preprocessing and trained model
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    # Apply the same preprocessing used during training
    X_transformed = preprocessor.transform(customer)

    # Convert sparse output to dense if necessary
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_transformed)

    # Handle binary-classification SHAP output
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    return (
        feature_names,
        shap_values[0],
        X_transformed[0]
    )


def clean_feature_name(feature_name):
    """
    Convert preprocessing-generated feature names
    into user-friendly labels.
    """

    name = feature_name

    # Remove preprocessing prefixes
    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    # Make common one-hot encoded features easier to read
    replacements = {
        "Contract_Month-to-month": "Contract: Month-to-month",
        "Contract_One year": "Contract: One year",
        "Contract_Two year": "Contract: Two year",
        "OnlineSecurity_No": "Online Security: No",
        "OnlineSecurity_Yes": "Online Security: Yes",
        "OnlineBackup_No": "Online Backup: No",
        "OnlineBackup_Yes": "Online Backup: Yes",
        "DeviceProtection_No": "Device Protection: No",
        "DeviceProtection_Yes": "Device Protection: Yes",
        "TechSupport_No": "Tech Support: No",
        "TechSupport_Yes": "Tech Support: Yes",
        "PaymentMethod_Electronic check": "Payment Method: Electronic check",
        "PaymentMethod_Mailed check": "Payment Method: Mailed check",
        "PaymentMethod_Bank transfer (automatic)": "Payment Method: Bank transfer",
        "PaymentMethod_Credit card (automatic)": "Payment Method: Credit card",
        "gender_Male": "Gender: Male",
        "gender_Female": "Gender: Female",
        "Partner_Yes": "Partner: Yes",
        "Partner_No": "Partner: No",
        "Dependents_Yes": "Dependents: Yes",
        "Dependents_No": "Dependents: No",
    }

    return replacements.get(name, name)
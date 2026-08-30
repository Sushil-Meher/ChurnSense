from .recommendations import (
    get_risk_level,
    generate_recommendations
)


def predict_customer(model, customer):
    probability = model.predict_proba(customer)[0, 1]

    risk = get_risk_level(probability)

    recommendations = generate_recommendations(
        customer.iloc[0]
    )

    return {
        "churn_probability": float(probability),
        "risk": risk,
        "recommendations": recommendations
    }
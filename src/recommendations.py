def get_risk_level(probability):
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


def generate_recommendations(customer):
    recommendations = []

    if customer["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer an incentive to switch to a long-term contract."
        )

    if customer["OnlineSecurity"] == "No":
        recommendations.append(
            "Offer an Online Security package or trial."
        )

    if customer["TechSupport"] == "No":
        recommendations.append(
            "Offer a discounted Tech Support package."
        )

    if customer["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Consider promoting automatic bank transfer or card payment."
        )

    if customer["InternetService"] == "Fiber optic":
        recommendations.append(
            "Review fiber plan pricing and offer a retention discount if appropriate."
        )

    if customer["tenure"] <= 6:
        recommendations.append(
            "Provide an early-tenure loyalty offer and proactive onboarding support."
        )

    return recommendations
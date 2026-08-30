import numpy as np
import pandas as pd


def create_features(df):
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Service-related features
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        df[service_cols]
        .eq("Yes")
        .sum(axis=1)
    )

    # New customer indicator
    df["IsNewCustomer"] = (
        df["tenure"] <= 6
    ).astype(int)

    # Long-term contract indicator
    df["LongTermContract"] = (
        df["Contract"].isin(
            ["One year", "Two year"]
        )
    ).astype(int)

    # Average monthly spending
    df["AvgMonthlySpend"] = (
        df["TotalCharges"]
        / df["tenure"].replace(0, np.nan)
    )

    df["AvgMonthlySpend"] = (
        df["AvgMonthlySpend"]
        .fillna(df["MonthlyCharges"])
    )

    # Charge per tenure month
    df["ChargePerTenureMonth"] = (
        df["TotalCharges"]
        / df["tenure"].replace(0, np.nan)
    )

    df["ChargePerTenureMonth"] = (
        df["ChargePerTenureMonth"]
        .fillna(df["MonthlyCharges"])
    )

    # Support indicator
    df["HasSupport"] = (
        df["TechSupport"] == "Yes"
    ).astype(int)

    # Security indicator
    df["HasSecurity"] = (
        df["OnlineSecurity"] == "Yes"
    ).astype(int)

    # Tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=[
            "0-6",
            "7-12",
            "13-24",
            "25-48",
            "49-72"
        ]
    )

    return df
import numpy as np
import pandas as pd

#to load the path
def load_data(path):
     df = pd.read_csv(path)
     return df

#to create cyclic features
def create_cyclic_features(df):

    # Hour cycle
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)

    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Weekly cycle
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)

    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # Monthly cycle
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)

    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df

# Encode categorical variables and split X and y
def encode_and_split_features_targets(df):

    # Encode categorical variables
    new_df = pd.get_dummies(
        df,
        columns=[
            "DayName",
            "Season",
            "Weather_Condition",
            "House_Type",
            "HVAC_Status",
            "Water_Heater_Status",
            "Day_Type"
            
        ],
        drop_first=True,
        dtype=int
    )

    # Input features
    X = new_df.drop(
        columns=[
            "Anomaly",
            "Energy_Cost",
            "Timestamp"
        ]
    )

    # Target 1: Energy forecasting
    y_energy = new_df[
        "Energy_Consumption_kWh"
    ]

    # Target 2: Anomaly detection
    y_anomaly = new_df[
        "Anomaly"
    ]

    return X, y_energy, y_anomaly, new_df
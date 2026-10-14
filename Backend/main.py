from fastapi import FastAPI, HTTPException
import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model #type: ignore

from Backend.schemas import(
    PredictionRequest,
    PredictionResponse
)
#fastapi application
app = FastAPI(
    title="Energy Consumption Intelligence API",
    description=(
        "API for energy consumption forecasting,"
        "anomaly detection and optimization"
    ),
    version = "1.0.0"
)
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)
FORECASTING_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model_forecasting.keras"
)
ANOMALY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model_Anomaly.keras"
) 
SCALAR_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)
FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)
THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "anomaly_threshold.pkl"
)
required_files = [
    FORECASTING_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    SCALAR_PATH,
    FEATURE_COLUMNS_PATH,
    THRESHOLD_PATH
]
for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )
#load trained models
forecasting_model = load_model(FORECASTING_MODEL_PATH)    
anomaly_model = load_model(ANOMALY_MODEL_PATH)

scaler = joblib.load(
    SCALAR_PATH
)
feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)
anomaly_threshold = joblib.load(
    THRESHOLD_PATH
)

print("Forecasting model load successfully.")
print("Anomaly model load successfully.")
print("Scaler load successfully.")
print("Feature columns model load successfully.")
print(f"Anomaly threshold: {anomaly_threshold}")

#endpoint
@app.get("/")
def home():
    return{
        "message": "Energy consumption Intelligence API is running",
        "services":[
            "Energy Consumption Forecasting",
            "Anomaly detection",
            "Energy Optimization"
        ]
    }
@app.post(
    "/predict",
     response_model = PredictionResponse

)
def predict(request: PredictionRequest):
    if len(request.records) != 24:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 24 hourly records are required"
                "for prediction"
            )
        )
    records = [
        record.model_dump()
        for record in request.records
    ]

    data = pd.DataFrame(records)

    data["Hour_sin"] = np.sin(
        2 * np.pi * data["Hour"] / 24
    )

    data["Hour_cos"] = np.cos(
        2 * np.pi * data["Hour"] / 24
    )

    data["DayOfWeek_sin"] = np.sin(
        2 * np.pi * data["DayOfWeek"] / 7
    )

    data["DayOfWeek_cos"] = np.cos(
        2 * np.pi * data["DayOfWeek"] / 7
    )

    data["Month_sin"] = np.sin(
        2 * np.pi * data["Month"] / 12
    )

    data["Month_cos"] = np.cos(
        2 * np.pi * data["Month"] / 12
    )  

    data = pd.get_dummies(
        data,
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

    data = data.drop(
        columns=[
            "Anomaly",
            "Energy_Cost"
        ],
        errors="ignore"
    ) 
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    try:

        X_scaled = scaler.transform(data)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Scaling error: {str(e)}"
        )
    X_sequence = X_scaled.reshape(
        1,
        24,
        X_scaled.shape[1]
    )
    try: #forecasting prediction

        energy_prediction = forecasting_model.predict(
            X_sequence,
            verbose=0
        )

        predicted_energy = float(
            energy_prediction[0][0]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Forecasting error: {str(e)}"
        )
    try: #anomaly detection

        reconstructed = anomaly_model.predict(
            X_sequence,
            verbose=0
        )

        anomaly_score = float(
            np.mean(
                np.square(
                    X_sequence - reconstructed
                )
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection error: {str(e)}"
        )
    if anomaly_score > anomaly_threshold:

        anomaly = 1

    else:

        anomaly = 0
    latest_record = records[-1]

    current_energy = float(
        latest_record[
            "Energy_Consumption_kWh"
        ]
    )

    current_price = float(
        latest_record[
            "Electricity_Price_per_kWh"
        ]
    )

    occupancy = int(
        latest_record[
            "Occupancy"
        ]
    )

    hvac_status = str(
        latest_record[
            "HVAC_Status"
        ]
    ) 
    water_heater_status = str(
        latest_record[
            "Water_Heater_Status"
        ]
    ) 
    current_energy_cost = (
        current_energy * current_price
    )

    predicted_energy_cost = (
        predicted_energy * current_price 
    ) 
    recommendations = []

    estimated_saving = 0.0 

    if predicted_energy > current_energy * 1.20:

        recommendations.append(
            "Predicted energy consumption is significantly "
            "higher than the current consumption."
        ) 
    if current_price > 0.25:

        recommendations.append(
            "Electricity price is relatively high. "
            "Consider shifting flexible loads to "
            "lower-price periods."
        )

        estimated_saving += (
            predicted_energy
            * current_price
            * 0.10
        ) 

    if (
        hvac_status.lower() == "on"
        and occupancy == 0
    ):

        recommendations.append(
            "HVAC is ON while occupancy is zero. "
            "Consider turning off or reducing HVAC usage."
        )

        estimated_saving += (
            predicted_energy
            * current_price
            * 0.05
        ) 
    if water_heater_status.lower() == "on":

        recommendations.append(
            "Consider scheduling water-heater operation "
            "during lower-price periods."
        )

        estimated_saving += (
            predicted_energy
            * current_price
            * 0.03
        )
    if anomaly == 1:

        recommendations.append(
            "Anomalous energy behavior detected. "
            "Inspect high-consumption appliances or "
            "HVAC equipment."
        )  
    if anomaly == 1:

        optimization_status = "Attention Required"

    elif predicted_energy > current_energy * 1.20:

        optimization_status = "High Consumption"

    elif current_price > 0.25:

        optimization_status = (
            "Cost Optimization Recommended"
        )

    else:

        optimization_status = "Energy Usage Normal" 
    return {

        "predicted_energy_kwh": round(
            predicted_energy,
            2
        ),

        "anomaly_score": round(
            anomaly_score,
            6
        ),

        "anomaly": anomaly,

        "current_energy_cost": round(
            current_energy_cost,
            2
        ),

        "predicted_energy_cost": round(
            predicted_energy_cost,
            2
        ),

        "optimization_status": optimization_status,
        "optimization_recommendation": (
            " ".join(recommendations)
        ),

        "estimated_saving": round(
            estimated_saving,
            2
        )
    }
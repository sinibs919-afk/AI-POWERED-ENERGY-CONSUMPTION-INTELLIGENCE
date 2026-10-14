from pydantic import BaseModel
from typing import List

class EnergyRecord(BaseModel):
    Energy_Consumption_kWh: float#(Historical energy consumption)
    #numerical features
    Temperature_C: float
    Humidity_Percent: float
    Occupancy: int
    Electricity_Price_per_kWh: float

    #time features
    Hour: int
    DayOfWeek: int
    Month: int
    IsWeekend: int

    #Categorical features
    DayName : str
    Season: str
    Weather_Condition: str
    House_Type: str
    HVAC_Status: str
    Water_Heater_Status: str
    Day_Type: str

class PredictionRequest(BaseModel):
    records: List[EnergyRecord]

class PredictionResponse(BaseModel):
    #forecasting
    predicted_energy_kwh: float
    #anomaly detection
    anomaly_score: float
    anomaly: int
    #cost
    current_energy_cost: float
    predicted_energy_cost: float
    #optimization
    optimization_status: str
    optimization_recommendation: str
    estimated_saving: float

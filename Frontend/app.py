import streamlit as st
import requests
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI POWERED ENERGY CONSUMPTION INTELLIGENCE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS for the color
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #F8F5FC;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #5B2C83;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        color: #5B2C83;
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #E4D8EF;
        box-shadow: 0px 3px 10px rgba(91, 44, 131, 0.08);
        text-align: center;
        min-height: 145px;
    }

    .metric-icon {
        font-size: 30px;
    }

    .metric-title {
        font-size: 14px;
        color: #777777;
        margin-top: 5px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        color: #5B2C83;
        margin-top: 5px;
    }

    /* Information box */
    .info-box {
        background-color: #F0E8F8;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #7B4BA3;
        margin-bottom: 15px;
    }

    /* Optimization box */
    .optimization-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #E4D8EF;
        box-shadow: 0px 3px 10px rgba(91, 44, 131, 0.08);
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F1EAF7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "http://backend:8000/predict"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ AI-POWERED ENERGY CONSUMPTION INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered Energy Forecasting, Anomaly Detection and
    Intelligent Energy Optimization.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚡ Energy AI")

    st.markdown(
        """
        ### What this app does

        🔮 **Forecasting**  
        Predicts the next-hour energy consumption.

        🚨 **Anomaly Detection**  
        Detects unusual energy behavior.

        💰 **Cost Analysis**  
        Calculates current and predicted energy cost.

        🧠 **Optimization**  
        Provides recommendations to reduce energy consumption
        and cost.
        """
    )

    st.divider()

    st.markdown("### 🔗 Backend Status")

    try:
        response = requests.get(
            "http://127.0.0.1:8000/",
            timeout=3
        )

        if response.status_code == 200:
            st.success("Backend Connected")
        else:
            st.warning("Backend Responding")

    except requests.exceptions.RequestException:
        st.error("Backend Offline")

    st.divider()

    st.caption(
        "Powered by LSTM, LSTM Autoencoder and FastAPI"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📊 Enter 24-Hour Energy Data</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    💡 <b>Important:</b> The model requires exactly
    <b>24 consecutive hourly records</b>. These records represent
    the previous 24 hours used by the AI models.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "records" not in st.session_state:
    st.session_state.records = []


# ============================================================
# INPUT MODE
# ============================================================

input_mode = st.radio(
    "Choose input method",
    [
        "📝 Enter data manually",
        "📄 Use sample data"
    ],
    horizontal=True
)


# ============================================================
# SAMPLE DATA
# ============================================================

def generate_sample_data():

    records = []

    energy_values = [
        2.10, 1.95, 1.90, 1.88,
        1.85, 2.00, 2.25, 2.70,
        3.10, 3.40, 3.60, 3.80,
        4.00, 4.20, 4.30, 4.40,
        4.30, 4.10, 3.80, 3.50,
        3.20, 2.90, 2.60, 2.30
    ]

    temperatures = [
        24.5, 24.2, 23.9, 23.7,
        23.5, 23.8, 24.2, 25.0,
        26.0, 27.0, 28.0, 29.0,
        30.0, 30.5, 31.0, 31.5,
        31.0, 30.5, 29.5, 28.5,
        27.5, 26.5, 25.5, 25.0
    ]

    humidity = [
        68, 70, 71, 72,
        73, 72, 70, 67,
        64, 61, 58, 55,
        53, 51, 50, 49,
        50, 52, 55, 58,
        61, 64, 66, 68
    ]

    occupancy = [
        1, 1, 1, 1,
        1, 1, 2, 2,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 2,
        2, 2, 1, 1
    ]

    prices = [
        0.12, 0.12, 0.11, 0.11,
        0.11, 0.11, 0.12, 0.13,
        0.14, 0.15, 0.15, 0.16,
        0.16, 0.17, 0.17, 0.18,
        0.18, 0.17, 0.16, 0.15,
        0.14, 0.13, 0.12, 0.12
    ]

    for hour in range(24):

        record = {
            "Energy_Consumption_kWh": energy_values[hour],
            "Temperature_C": temperatures[hour],
            "Humidity_Percent": humidity[hour],
            "Occupancy": occupancy[hour],
            "Electricity_Price_per_kWh": prices[hour],
            "Hour": hour,
            "DayOfWeek": 0,
            "Month": 8,
            "IsWeekend": 0,
            "DayName": "Monday",
            "Season": "Summer",
            "Weather_Condition": (
                "Hot"
                if temperatures[hour] >= 29
                else "Clear"
            ),
            "House_Type": "Apartment",
            "HVAC_Status": (
                "ON"
                if 7 <= hour <= 19
                else "OFF"
            ),
            "Water_Heater_Status": (
                "ON"
                if hour in [5, 6, 18, 19, 20, 21]
                else "OFF"
            ),
            "Day_Type": "Weekday"
        }

        records.append(record)

    return records


# ============================================================
# MANUAL INPUT
# ============================================================

if input_mode == "📝 Enter data manually":

    st.markdown(
        "### 🕐 Select the hour to enter"
    )

    selected_hour = st.selectbox(
        "Hour",
        list(range(24)),
        format_func=lambda x: f"{x:02d}:00"
    )

    st.markdown(
        f"#### 🕐 Data for {selected_hour:02d}:00"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        energy = st.number_input(
            "⚡ Energy Consumption (kWh)",
            min_value=0.0,
            value=2.5,
            step=0.1,
            help=(
                "Enter the energy consumed during this hour "
                "in kilowatt-hours (kWh). Example: 2.5"
            )
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            value=25.0,
            step=0.1,
            help=(
                "Enter the outdoor temperature in Celsius. "
                "Example: 28.5"
            )
        )

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help=(
                "Enter relative humidity between 0 and 100%. "
                "Example: 65"
            )
        )

        occupancy = st.number_input(
            "👥 Occupancy",
            min_value=0,
            value=2,
            step=1,
            help=(
                "Enter the number of people present in the house. "
                "Example: 2"
            )
        )

    with col2:

        price = st.number_input(
            "💰 Electricity Price",
            min_value=0.0,
            value=0.15,
            step=0.01,
            help=(
                "Enter electricity price per kWh. "
                "Example: 0.15"
            )
        )

        day_of_week = st.number_input(
            "📅 Day of Week",
            min_value=0,
            max_value=6,
            value=0,
            step=1,
            help=(
                "0 = Monday, 1 = Tuesday, ..., 6 = Sunday."
            )
        )

        month = st.number_input(
            "🗓️ Month",
            min_value=1,
            max_value=12,
            value=8,
            step=1,
            help=(
                "Enter month number. "
                "January = 1 and December = 12."
            )
        )

        is_weekend = st.selectbox(
            "🏖️ Weekend?",
            [0, 1],
            format_func=lambda x: (
                "No" if x == 0 else "Yes"
            ),
            help=(
                "Select Yes for Saturday/Sunday and "
                "No for weekdays."
            )
        )

    with col3:

        day_name = st.selectbox(
            "📆 Day Name",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ],
            help="Select the day of the week."
        )

        season = st.selectbox(
            "🌤️ Season",
            [
                "Spring",
                "Summer",
                "Autumn",
                "Winter"
            ],
            help="Select the current season."
        )

        weather = st.selectbox(
            "☁️ Weather Condition",
            [
                "Clear",
                "Sunny",
                "Cloudy",
                "Rainy",
                "Hot",
                "Cold"
            ],
            help="Select the current weather condition."
        )

        house_type = st.selectbox(
            "🏠 House Type",
            [
                "Apartment",
                "Independent House"
            ],
            help="Select the type of residential building."
        )

    col4, col5 = st.columns(2)

    with col4:

        hvac = st.selectbox(
            "❄️ HVAC Status",
            ["OFF", "ON"],
            help=(
                "Select ON if heating/cooling system "
                "was operating during this hour."
            )
        )

    with col5:

        water_heater = st.selectbox(
            "🚿 Water Heater Status",
            ["OFF", "ON"],
            help=(
                "Select ON if the water heater "
                "was operating during this hour."
            )
        )

    day_type = st.selectbox(
        "🏢 Day Type",
        ["Weekday", "Weekend"],
        help=(
            "Select Weekday for Monday-Friday "
            "and Weekend for Saturday-Sunday."
        )
    )

    if st.button(
        "➕ Add This Hour",
        use_container_width=True
    ):

        record = {
            "Energy_Consumption_kWh": energy,
            "Temperature_C": temperature,
            "Humidity_Percent": humidity,
            "Occupancy": occupancy,
            "Electricity_Price_per_kWh": price,
            "Hour": selected_hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend,
            "DayName": day_name,
            "Season": season,
            "Weather_Condition": weather,
            "House_Type": house_type,
            "HVAC_Status": hvac,
            "Water_Heater_Status": water_heater,
            "Day_Type": day_type
        }

        existing = [
            r["Hour"]
            for r in st.session_state.records
        ]

        if selected_hour in existing:

            st.warning(
                "This hour has already been added."
            )

        else:

            st.session_state.records.append(record)

            st.success(
                f"Hour {selected_hour:02d}:00 added successfully."
            )


# ============================================================
# SAMPLE DATA
# ============================================================

else:

    st.markdown(
        """
        <div class="info-box">
        🚀 <b>Demo Mode:</b> Click the button below to automatically
        create 24 realistic hourly records for testing the complete
        AI pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 Load 24-Hour Demo Data",
        use_container_width=True
    ):

        st.session_state.records = generate_sample_data()

        st.success(
            "24-hour demo data loaded successfully!"
        )


# ============================================================
# CURRENT RECORDS
# ============================================================

if len(st.session_state.records) > 0:

    st.markdown(
        "### 📋 Entered Data"
    )

    sorted_records = sorted(
        st.session_state.records,
        key=lambda x: x["Hour"]
    )

    df = pd.DataFrame(sorted_records)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"📊 {len(sorted_records)} / 24 hourly records entered"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗑️ Clear All Data",
            use_container_width=True
        ):

            st.session_state.records = []

            st.rerun()

    with col2:

        if len(sorted_records) == 24:

            st.success(
                "✅ Ready for prediction!"
            )

        else:

            st.warning(
                f"⚠️ Add {24 - len(sorted_records)} "
                "more hourly records."
            )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 AI Analysis</div>',
    unsafe_allow_html=True
)


if len(st.session_state.records) == 24:

    if st.button(
        "🚀 Run Energy Intelligence Analysis",
        type="primary",
        use_container_width=True
    ):

        records = sorted(
            st.session_state.records,
            key=lambda x: x["Hour"]
        )

        payload = {
            "records": records
        }

        with st.spinner(
            "🧠 AI models are analyzing your energy data..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state.result = result

                    st.success(
                        "Analysis completed successfully!"
                    )

                else:

                    st.error(
                        f"Backend Error {response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ The request timed out. "
                    "Please try again."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )

else:

    st.info(
        "👆 Enter exactly 24 hourly records to enable prediction."
    )


# ============================================================
# RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state.result

    st.divider()

    st.markdown(
        '<div class="section-title">📊 AI Results</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-title">
                    Predicted Energy
                </div>
                <div class="metric-value">
                    {result["predicted_energy_kwh"]:.2f} kWh
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        anomaly = result["anomaly"]

        if anomaly == 1:
            anomaly_text = "Anomaly Detected"
            anomaly_icon = "🚨"
        else:
            anomaly_text = "Normal"
            anomaly_icon = "✅"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{anomaly_icon}</div>
                <div class="metric-title">
                    Anomaly Status
                </div>
                <div class="metric-value">
                    {anomaly_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">💰</div>
                <div class="metric-title">
                    Predicted Cost
                </div>
                <div class="metric-value">
                    {result["predicted_energy_cost"]:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">💡</div>
                <div class="metric-title">
                    Estimated Saving
                </div>
                <div class="metric-value">
                    {result["estimated_saving"]:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    st.markdown(
        "### 🔮 Energy Forecast"
    )

    st.info(
        f"""
        The LSTM model predicts approximately
        **{result["predicted_energy_kwh"]:.2f} kWh**
        of energy consumption for the next hour.
        """
    )


    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Anomaly Detection"
    )

    anomaly_score = result["anomaly_score"]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Anomaly Score",
            f"{anomaly_score:.6f}"
        )

    with col2:

        if result["anomaly"] == 1:

            st.error(
                "🚨 Unusual energy behavior detected."
            )

        else:

            st.success(
                "✅ Energy behavior appears normal."
            )


    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    st.markdown(
        "### 💰 Energy Cost Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Current Energy Cost",
            f"{result['current_energy_cost']:.2f}"
        )

    with col2:

        st.metric(
            "Predicted Energy Cost",
            f"{result['predicted_energy_cost']:.2f}"
        )


    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Energy Optimization"
    )

    status = result["optimization_status"]

    if status == "Attention Required":

        st.error(
            f"🚨 {status}"
        )

    elif status == "High Consumption":

        st.warning(
            f"⚠️ {status}"
        )

    elif status == "Cost Optimization Recommended":

        st.warning(
            f"💰 {status}"
        )

    else:

        st.success(
            f"✅ {status}"
        )


    st.markdown(
        f"""
        <div class="optimization-box">

        <h4>💡 Recommendation</h4>

        <p>
        {result["optimization_recommendation"]}
        </p>

        <hr>

        <b>💰 Estimated Saving:</b>
        {result["estimated_saving"]:.2f}

        </div>
        """,
        unsafe_allow_html=True
    )
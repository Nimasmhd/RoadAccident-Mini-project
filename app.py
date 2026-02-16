import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚦",
    layout="wide"
)

# ------------------------------
# Load Model & Preprocessing Objects
# ------------------------------
model = joblib.load("road_accident_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ------------------------------
# Get Current Date & Time
# ------------------------------
now = datetime.now()
current_year = now.year
current_month = now.month
current_hour = now.hour

# ------------------------------
# Title Section
# ------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#d32f2f;'>
    🚦 Road Accident Severity Prediction System
    </h1>
    <p style='text-align:center; font-size:18px;'>
    Machine Learning–based accident severity prediction (Sri Lanka context)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("📝 Accident Details")

year = st.sidebar.number_input(
    "📅 Year",
    min_value=current_year,
    max_value=current_year + 10,
    value=current_year
)

month = st.sidebar.slider(
    "🗓️ Month",
    min_value=1,
    max_value=12,
    value=current_month
)

hour = st.sidebar.slider(
    "⏰ Hour of Day",
    min_value=0,
    max_value=23,
    value=current_hour
)

longitude = st.sidebar.number_input("🌍 Longitude", value=80.62)
latitude = st.sidebar.number_input("📍 Latitude", value=8.78)

road_type = st.sidebar.selectbox(
    "🛣️ Road Type",
    label_encoders["Road Type"].classes_
)

weather = st.sidebar.selectbox(
    "🌦️ Weather",
    label_encoders["Weather"].classes_
)

gender = st.sidebar.selectbox(
    "👤 Gender",
    label_encoders["Gender"].classes_
)

vehicle = st.sidebar.selectbox(
    "🚗 Vehicle Type",
    label_encoders["Vehicle"].classes_
)

# ------------------------------
# Main Layout
# ------------------------------
col1, col2 = st.columns(2)

# ------------------------------
# Input Summary
# ------------------------------
with col1:
    st.subheader("📥 Input Summary")
    st.info(f"""
    **Year:** {year}  
    **Month:** {month}  
    **Hour:** {hour}  
    **Latitude:** {latitude}  
    **Longitude:** {longitude}  
    **Road Type:** {road_type}  
    **Weather:** {weather}  
    **Gender:** {gender}  
    **Vehicle:** {vehicle}
    """)

# ------------------------------
# Prediction Section
# ------------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if st.button("🚀 Predict Severity", use_container_width=True):

        # Create raw input DataFrame
        raw_new_data = pd.DataFrame({
            "Year": [year],
            "longitude": [longitude],
            "latitude": [latitude],
            "Road Type": [road_type],
            "Weather": [weather],
            "Gender": [gender],
            "Vehicle": [vehicle],
            "Hour": [hour],
            "Month": [month]
        })

        # Encode categorical variables
        for col in ["Road Type", "Weather", "Gender", "Vehicle"]:
            raw_new_data[col] = label_encoders[col].transform(raw_new_data[col])

        # Scale numeric features
        new_data_scaled = scaler.transform(raw_new_data)

        # Predict
        prediction = model.predict(new_data_scaled)[0]

        # Probability (if supported)
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(new_data_scaled)[0].max() * 100
            st.progress(int(confidence))
            st.caption(f"Prediction Confidence: {confidence:.2f}%")

        # Display Result
        if prediction == 0:
            st.error("⚠️ HIGH SEVERITY ACCIDENT RISK")
            st.markdown("🚨 **Warning:** High probability of severe accident outcome.")
        else:
            st.success("✅ LOW SEVERITY ACCIDENT RISK")
            st.markdown("🟢 **Safe:** Lower probability of severe consequences.")

st.divider()

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    <p style='text-align:center; font-size:14px;'>
    Developed By MRM Nimas Using Streamlit | Data Mining University Project
    </p>
    """,
    unsafe_allow_html=True
)

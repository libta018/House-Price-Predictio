import streamlit as st
import pandas as pd
import joblib
import os

# Load model and pipeline
if not os.path.exists("model.pkl") or not os.path.exists("pipeline.pkl"):
    st.error("⚠️ Model or pipeline file missing! Run House_Price.py first to generate them.")
    st.stop()

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Load dataset for defaults
if os.path.exists("housing.csv"):
    df = pd.read_csv("housing.csv")
else:
    df = None

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction App")
st.markdown("Fill in the details below to predict the **median house value** in California.")

# Helper: get default value from dataset
def get_default(col, default=0.0):
    if df is not None and col in df.columns:
        return float(df[col].median())
    return default

# User Inputs
longitude = st.number_input("🌍 Longitude", value=get_default("longitude", -120.0))
latitude = st.number_input("🌍 Latitude", value=get_default("latitude", 35.0))
housing_median_age = st.number_input("🏘️ Median Age of Houses", value=get_default("housing_median_age", 20))
total_rooms = st.number_input("🚪 Total Rooms", value=get_default("total_rooms", 2000))
total_bedrooms = st.number_input("🛏️ Total Bedrooms", value=get_default("total_bedrooms", 400))
population = st.number_input("👨‍👩‍👧 Population", value=get_default("population", 800))
households = st.number_input("🏠 Households", value=get_default("households", 300))
median_income = st.number_input("💰 Median Income (in tens of thousands)", value=get_default("median_income", 3.5))

ocean_proximity = st.selectbox(
    "🌊 Ocean Proximity",
    ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

# Predict Button
if st.button("🔮 Predict House Price"):
    # Prepare input as DataFrame
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    # Transform and predict
    transformed_data = pipeline.transform(input_data)
    prediction = model.predict(transformed_data)[0]

    st.success(f"🏠 Predicted Median House Value: **${prediction:,.2f}**")


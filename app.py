import streamlit as st
import pandas as pd
import joblib
import os
import gdown  

# --- Google Drive Model Download ---
MODEL_FILE_ID = "1uen3ir3lX9Y_-2iUU1r9ACdAVB8i1BvV"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

if not os.path.exists("model.pkl"):
    st.info("Downloading model from Google Drive...")
    output = gdown.download(MODEL_URL, "model.pkl", quiet=False)
    if output is None:
        st.error("⚠️ Model download failed! Check the Google Drive link.")
        st.stop()

# --- Load Pipeline ---
if not os.path.exists("pipeline.pkl"):
    st.error("⚠️ Pipeline file missing in repo!")
    st.stop()

# --- Load Model and Pipeline ---
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# --- Load dataset for default values ---
if os.path.exists("housing.csv"):
    df = pd.read_csv("housing.csv")
else:
    df = None

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction App")
st.markdown("Fill in the details below to predict the **median house value** in California.")

# --- Helper: get default value ---
def get_default(col, default=0.0):
    if df is not None and col in df.columns:
        return float(df[col].median())
    return default

# --- First row: longitude, latitude, housing_median_age ---
col1, col2, col3 = st.columns(3)
with col1:
    longitude = st.text_input("🌍 Longitude", value=str(get_default("longitude", -120.0)))
with col2:
    latitude = st.text_input("🌍 Latitude", value=str(get_default("latitude", 35.0)))
with col3:
    housing_median_age = st.text_input("🏘️ Median Age of Houses", value=str(get_default("housing_median_age", 20)))

# --- Second row: total_rooms, total_bedrooms ---
col4, col5 = st.columns(2)
with col4:
    total_rooms = st.text_input("🚪 Total Rooms", value=str(get_default("total_rooms", 2000)))
with col5:
    total_bedrooms = st.text_input("🛏️ Total Bedrooms", value=str(get_default("total_bedrooms", 400)))

# --- Third row: population, households ---
col6, col7 = st.columns(2)
with col6:
    population = st.text_input("👨‍👩‍👧 Population", value=str(get_default("population", 800)))
with col7:
    households = st.text_input("🏠 Households", value=str(get_default("households", 300)))

# --- Last row: median_income and ocean_proximity ---
col8, col9 = st.columns(2)
with col8:
    median_income = st.text_input("💰 Median Income (in tens of thousands)", value=str(get_default("median_income", 3.5)))
with col9:
    ocean_proximity = st.selectbox(
        "🌊 Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    )

# --- Predict Button ---
if st.button("🔮 Predict House Price"):
    try:
        # Convert inputs to float and prepare DataFrame
        input_data = pd.DataFrame([{
            "longitude": float(longitude),
            "latitude": float(latitude),
            "housing_median_age": float(housing_median_age),
            "total_rooms": float(total_rooms),
            "total_bedrooms": float(total_bedrooms),
            "population": float(population),
            "households": float(households),
            "median_income": float(median_income),
            "ocean_proximity": ocean_proximity
        }])

        # Transform and predict
        transformed_data = pipeline.transform(input_data)
        prediction = model.predict(transformed_data)[0]

        st.success(f"🏠 Predicted Median House Value: **${prediction:,.2f}**")
    except ValueError:
        st.error("⚠️ Please enter valid numbers in all fields!")



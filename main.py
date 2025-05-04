# prompt: buatkan code streamlit dari code saya di atas

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load the trained model and scaler
model = keras.models.load_model('plant_growth_model.h5')
scaler = StandardScaler()  # You'll need to load the scaler you used during training
label_encoder = LabelEncoder() # Load your label encoder

# Function to simulate plant growth (same as in your original code)
def simulate_plant_growth(soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity, ph):
    sample_input = np.array([[soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity, ph]])
    sample_input_df = pd.DataFrame(sample_input)
    sample_input_scaled = scaler.transform(sample_input_df)
    predicted_class = np.argmax(model.predict(sample_input_scaled))
    predicted_personality = label_encoder.inverse_transform([predicted_class])
    return predicted_personality[0]

# Streamlit app
st.title("Plant Growth Prediction")

# Input fields
soil_type = st.number_input("Soil Type (Encoded)", min_value=0, max_value=100, value=2)  # Adjust min/max as needed
water_frequency = st.number_input("Water Frequency (Encoded)", min_value=0, max_value=100, value=37)
fertilizer_type = st.number_input("Fertilizer Type (Encoded)", min_value=0, max_value=100, value=0)
sunlight_hours = st.number_input("Sunlight Hours (Encoded)", min_value=0, max_value=100, value=2)
temperature = st.number_input("Temperature (Encoded)", min_value=0, max_value=1000, value=169)
humidity = st.number_input("Humidity (Encoded)", min_value=0, max_value=1000, value=113)
ph = st.number_input("pH (Encoded)", min_value=0, max_value=100, value=0)

# Prediction
if st.button("Predict Growth Milestone"):
    predicted_growth = simulate_plant_growth(soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity, ph)
    st.write(f"Predicted plant growth milestone: {predicted_growth}")

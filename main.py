# prompt: buatkan streamlit tanpa tensorflow

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model (replace with your actual model loading)
# Assuming the model is saved as 'plant_growth_model.h5'
# model = tf.keras.models.load_model('plant_growth_model.h5')

# Dummy model and scaler for demonstration (replace with your actual model and scaler)
class DummyModel:
    def predict(self, X):
        return np.random.rand(X.shape[0], 1)
model = DummyModel()
scaler = StandardScaler()


# Load label encoders (replace with your actual label encoders)
# le_Soil_Type = LabelEncoder()
# (load other label encoders)
joblib.dump(le_Soil_Type, 'le_Soil_Type.pkl')
joblib.dump(le_Water_Frequency, 'le_Water_Frequency.pkl')
joblib.dump(le_Fertilizer_Type, 'le_Fertilizer_Type.pkl')
joblib.dump(le_Sunlight_Hours, 'le_Sunlight_Hours.pkl')
joblib.dump(le_Temperature, 'le_Temperature.pkl')
joblib.dump(le_Humidity, 'le_Humidity.pkl')

# Dummy label encoders for demonstration
class DummyLabelEncoder:
    def fit_transform(self, data):
        return np.random.randint(0, 3, size=len(data))
    def transform(self, data):
        return np.random.randint(0, 3, size=len(data))
    
le_Soil_Type = DummyLabelEncoder()
le_Water_Frequency = DummyLabelEncoder()
le_Fertilizer_Type = DummyLabelEncoder()
le_Sunlight_Hours = DummyLabelEncoder()
le_Temperature = DummyLabelEncoder()
le_Humidity = DummyLabelEncoder()


def predict_growth_milestone(model, scaler, soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity):
    input_data = np.array([[soil_type, water_frequency, fertilizer_type, sunlight_hours, temperature, humidity]])
    scaled_input = scaler.fit_transform(input_data) # Use fit_transform here as well
    prediction = model.predict(scaled_input)
    predicted_class = (prediction > 0.5).astype(int)[0][0]
    return predicted_class

# Streamlit app
st.title("Plant Growth Milestone Prediction")

# Input features
soil_type = st.selectbox("Soil Type", ["Sandy", "Clayey", "Loamy"])
water_frequency = st.selectbox("Water Frequency", ["Daily", "Weekly", "Bi-Weekly"])
fertilizer_type = st.selectbox("Fertilizer Type", ["Organic", "Inorganic", "Balanced"])
sunlight_hours = st.number_input("Sunlight Hours", min_value=0, max_value=24)
temperature = st.number_input("Temperature", min_value=0)
humidity = st.number_input("Humidity", min_value=0, max_value=100)

# Encode input features
soil_type_encoded = le_Soil_Type.transform(np.array([soil_type]))[0]
water_frequency_encoded = le_Water_Frequency.transform(np.array([water_frequency]))[0]
fertilizer_type_encoded = le_Fertilizer_Type.transform(np.array([fertilizer_type]))[0]
sunlight_hours_encoded = le_Sunlight_Hours.transform(np.array([sunlight_hours]))[0]
temperature_encoded = le_Temperature.transform(np.array([temperature]))[0]
humidity_encoded = le_Humidity.transform(np.array([humidity]))[0]


# Prediction button
if st.button("Predict"):
    # Make prediction
    predicted_milestone = predict_growth_milestone(model, scaler, soil_type_encoded, water_frequency_encoded, fertilizer_type_encoded, sunlight_hours_encoded, temperature_encoded, humidity_encoded)

    # Display prediction
    st.write(f"Predicted Growth Milestone: {predicted_milestone}")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained classification model, dataset, scaler, and feature names
def load_model():
    try:
        data = joblib.load("laptop_classification_model.joblib")
        if isinstance(data, tuple) and len(data) == 4:
            return data  # Returns (df, model, scaler, feature_names)
        else:
            st.error("Error: The joblib file does not contain the expected data. Please retrain and save the model correctly.")
            return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

df, model, scaler, feature_names = load_model()

# Ensure the model and dataset are loaded before proceeding
if df is None or model is None or scaler is None or feature_names is None:
    st.stop()

# Streamlit UI
st.title("Laptop Classification & Recommendation System")
st.write("Select your laptop specifications to predict the processor brand and get recommendations.")

# User Inputs for Classification
num_cores = st.number_input("Number of CPU Cores", min_value=1, max_value=16, value=4)
num_threads = st.number_input("Number of CPU Threads", min_value=1, max_value=32, value=8)
ram_memory = st.selectbox("Select RAM Size (GB)", sorted(df['ram_memory'].unique()))
primary_storage = st.selectbox("Primary Storage Capacity (GB)", sorted(df['primary_storage_capacity'].unique()))
display_size = st.selectbox("Select Display Size (inches)", sorted(df['display_size'].unique()))
resolution_width = st.number_input("Resolution Width", min_value=800, max_value=3840, value=1920)
resolution_height = st.number_input("Resolution Height", min_value=600, max_value=2160, value=1080)

# Fix: Ensure price input default is within valid range
default_price = max(min(500.0, float(df['Price'].max())), float(df['Price'].min()))
price = st.number_input(
    "Price (USD)", 
    min_value=float(df['Price'].min()), 
    max_value=float(df['Price'].max()), 
    value=default_price
)

# Create input dataframe for prediction with correct feature names
input_data = pd.DataFrame({
    "num_cores": [num_cores],
    "num_threads": [num_threads],
    "ram_memory": [ram_memory],
    "primary_storage_capacity": [primary_storage],
    "display_size": [display_size],
    "resolution_width": [resolution_width],
    "resolution_height": [resolution_height],
    "Price": [price]
})

# Ensure input_data has all necessary columns (handle missing columns)
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0  # Default value for missing columns

# Ensure feature order matches training
input_data = input_data[feature_names]

# Use the **same scaler** from training to transform the input data
X_scaled = scaler.transform(input_data)

# Predict processor brand
if st.button("Predict Processor Brand"):
    prediction = model.predict(X_scaled)
    st.success(f"Predicted Processor Brand: {prediction[0]}")

# Laptop Recommendations based on similar specifications
st.subheader("Laptop Recommendations")
filtered_df = df[(df['ram_memory'] == ram_memory) & (df['primary_storage_capacity'] == primary_storage) &
                 (df['display_size'] == display_size) & (df['Price'] >= price * 0.9) & (df['Price'] <= price * 1.1)]

st.dataframe(filtered_df[['brand', 'Model', 'Price', 'processor_brand', 'ram_memory', 'gpu_brand']])

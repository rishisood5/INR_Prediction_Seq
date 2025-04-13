# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import calendar
import tensorflow as tf  # Ensure TensorFlow is imported (needed for Keras models)


# Load trained model
with open('inr_seq_model.pkl', 'rb') as file:
    model = pickle.load(file)
    

st.title("INR Exchange Rate Predictor")
st.write("Predict future USD to INR closing rate using the LSTM model.")

# --- User Input Fields ---
# Day input via number input
day = st.number_input("Select Day", min_value=1, max_value=31, value=1)

# Month selector from January to December
months = list(calendar.month_name)[1:]
selected_month = st.selectbox("Select Month", months)

# Year selector for years 2023 to 2026
year = st.selectbox("Select Year", list(range(2023, 2027)))

# Convert the selected month to its numerical value
month_num = list(calendar.month_name).index(selected_month)

# When user clicks the Predict button
if st.button("Predict INR Rate"):
    # Create target date from user inputs
    try:
        target_date = datetime(year, month_num, int(day))
    except Exception as e:
        st.error("Invalid date selected. Please check the day, month, and year inputs.")
        st.stop()
    
    # --- Load and Process Historical Data ---
    try:
        data = pd.read_csv('HistoricalData.csv')
    except Exception as e:
        st.error("Error loading HistoricalData.csv. Please ensure it is in the same directory.")
        st.stop()

    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data = data.sort_values('Date')
    data = data[['Date', 'Close/Last']].dropna()
    data.set_index('Date', inplace=True)

    # Show last historical date
    last_date = data.index[-1]
    st.write(f"Last historical date: {last_date.date()}")

    if target_date <= last_date:
        st.error("Prediction date must be after the last historical date in the dataset.")
        st.stop()
    
    # --- Load the Trained Model ---
    try:
        with open('inr_seq_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error("Error loading the model file (inr_seq_model.pkl).")
        st.stop()

    # --- Prepare the Data for Prediction ---
    closing_prices = data['Close/Last'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(closing_prices)

    seq_length = 60  # Same sequence length used during training
    if len(scaled_prices) < seq_length:
        st.error("Insufficient historical data for prediction.")
        st.stop()

    # Determine forecast horizon: number of days from last historical date to target date
    forecast_days = (target_date - last_date).days
    st.write(f"Forecasting {forecast_days} day(s) ahead.")

    # --- Iterative Multi-step Forecasting ---
    # Starting sequence is the last 'seq_length' days from the scaled historical data.
    seq = scaled_prices[-seq_length:].copy()  # shape: (seq_length, 1)
    last_pred = None

    # Iterate forecast_days times, updating the sequence each time with the new prediction.
    for _ in range(forecast_days):
        # Reshape to match model input: (1, seq_length, 1)
        pred_scaled = model.predict(seq.reshape(1, seq_length, 1))
        last_pred = pred_scaled[0, 0]  # Extract predicted scaled value
        # Append prediction to sequence and remove the oldest entry
        seq = np.append(seq, [[last_pred]], axis=0)
        seq = seq[1:]

    # Inverse transform the final predicted scaled value back to INR rate
    predicted_value = scaler.inverse_transform([[last_pred]])[0, 0]

    st.success(f"Predicted USD to INR rate on {target_date.date()} is: ₹{predicted_value:.4f}")

# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import calendar
import pickle

# Load trained model
with open('inr_seq_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Title
st.title("USD to INR Exchange Rate Predictor")
st.write("Forecast the USD to INR rate using a trained Random Forest model.")

# User Inputs
months = list(calendar.month_name)[1:]  # January to December
years = list(range(2025, 2051))

selected_month = st.selectbox("Select Month", months)
selected_year = st.selectbox("Select Year", years)

# Convert selected month to numerical value
month_num = list(calendar.month_name).index(selected_month)

# Load data
@st.cache_data
def load_and_train_model():
    data = pd.read_csv("HistoricalData.csv")
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data = data.sort_values('Date')
    data = data[['Date', 'Close/Last']].dropna()
    data.set_index('Date', inplace=True)

    # Feature Engineering
    for lag in range(1, 6):
        data[f'lag_{lag}'] = data['Close/Last'].shift(lag)
    data['rolling_mean_3'] = data['Close/Last'].rolling(window=3).mean()
    data['rolling_std_3'] = data['Close/Last'].rolling(window=3).std()
    data.dropna(inplace=True)

    X = data.drop('Close/Last', axis=1)
    y = data['Close/Last']

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, data

model, data = load_and_train_model()

# Generate input features for prediction
if st.button("Predict Exchange Rate"):
    try:
        last_known = data.iloc[-5:].copy()  # Last known data for lag features
        latest_close = data['Close/Last'].iloc[-1]

        # Simulate next date's features using previous values
        new_row = {
            'lag_1': latest_close,
            'lag_2': data['Close/Last'].iloc[-2],
            'lag_3': data['Close/Last'].iloc[-3],
            'lag_4': data['Close/Last'].iloc[-4],
            'lag_5': data['Close/Last'].iloc[-5],
            'rolling_mean_3': data['Close/Last'].iloc[-3:].mean(),
            'rolling_std_3': data['Close/Last'].iloc[-3:].std(),
        }

        input_df = pd.DataFrame([new_row])

        prediction = model.predict(input_df)[0]

        st.success(f"Predicted USD to INR exchange rate for {selected_month} {selected_year}: ₹{prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

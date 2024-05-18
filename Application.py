import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# Load the trained LSTM model
model_path = 'water_consumption_lstm_model.h5'  # Update with your model path
model = load_model(model_path)

# Load the scaler
scaler_path = 'scaler.pkl'  # Update with your scaler path
scaler = joblib.load(scaler_path)

# Function to preprocess data and make predictions
def predict_water_consumption(input_data):
    # Ensure input_data is a 2D array
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)
    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    # Reshape input data for LSTM model
    input_data_reshaped = np.reshape(input_data_scaled, (1, input_data_scaled.shape[0], input_data_scaled.shape[1]))
    # Make prediction
    predicted_consumption_scaled = model.predict(input_data_reshaped)
    # Inverse transform the predicted values
    predicted_consumption = scaler.inverse_transform(predicted_consumption_scaled)
    return predicted_consumption

# Streamlit app
st.title('Water Consumption Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Show the uploaded data
    st.subheader('Uploaded Data')
    st.write(df)

    # Get the latest 10 days' data
    latest_data = df.tail(10)

    # Show the latest data
    st.subheader('Latest 10 Days Data')
    st.write(latest_data)

    # Predict water consumption
    predicted_consumption = predict_water_consumption(latest_data.values)
    
    # Show the predicted consumption
    st.subheader('Predicted Water Consumption for the Next Day')
    st.write(predicted_consumption)

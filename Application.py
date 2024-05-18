import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime

# Load the model
model_path = 'water_consumption_lstm_model.h5'
model = load_model(model_path)

# Load the scaler
scaler = joblib.load('scaler.save')

# Function to create sequences
def create_sequences(data, time_steps=1):
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Function to predict future water consumption
def predict_future_consumption(last_known_data, future_steps=1):
    try:
        # Scale the last known data
        scaled_last_known_data = scaler.transform(last_known_data.reshape(-1, 1))
        # Create sequences for prediction
        input_data = create_sequences(scaled_last_known_data, time_steps=10)
        if len(input_data) == 0:
            raise ValueError("Input data is empty.")
        # Predict future consumption
        future_predictions_scaled = []
        current_sequence = input_data[-1]
        for i in range(future_steps):
            # Reshape data for model input
            current_sequence = current_sequence.reshape((1, 10, 1))
            # Make prediction
            future_prediction_scaled = model.predict(current_sequence)[0, 0]
            # Add prediction to sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[-1][-1] = future_prediction_scaled
            future_predictions_scaled.append(future_prediction_scaled)
        # Inverse transform predictions to original scale
        future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
        return future_predictions
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

        
# Streamlit UI
st.title('Future Water Consumption Predictor')

# Sidebar for user input
st.sidebar.header('User Input')
last_known_date = st.sidebar.date_input('Last Known Date', datetime.date.today())
last_known_data = st.sidebar.number_input('Last Known Water Consumption in Liters', min_value=0.0, value=0.0, step=0.01)
future_steps = st.sidebar.slider('Number of Steps into the Future', min_value=1, max_value=100, value=1)

# Predict future consumption
if st.sidebar.button('Predict'):
    last_known_data_array = np.array([last_known_data])
    future_predictions = predict_future_consumption(last_known_data_array, future_steps)
    future_dates = [last_known_date + datetime.timedelta(days=i) for i in range(1, future_steps + 1)]
    
    # Display predictions
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Consumption': future_predictions.flatten()
    })
    st.subheader('Future Water Consumption Predictions')
    st.write(prediction_df)

# Streamlit app footer
st.sidebar.text('Powered by Your Model')

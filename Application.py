import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load pre-trained model and scaler
model_path = 'water_consumption_lstm_model.h5'  # Update with your model path
scaler_path = 'scaler.save'  # Update with your scaler path

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Function to preprocess input data
def preprocess_input(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Function to make predictions
def predict_consumption(input_data):
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    # Preprocess input data for LSTM
    X = preprocess_input(scaled_input, time_steps)
    # Make predictions
    y_pred_scaled = model.predict(X)
    # Inverse transform the predictions to original scale
    y_pred = scaler.inverse_transform(y_pred_scaled)
    return y_pred.flatten()

# Streamlit app
st.title('Water Consumption Prediction')

# User input for historical consumption data
st.write('Enter the last {} days of water consumption data:'.format(time_steps))
input_data = []
for i in range(time_steps):
    value = st.number_input('Day {}'.format(i+1), min_value=0.0)
    input_data.append(value)

# Predict consumption
if st.button('Predict'):
    prediction = predict_consumption(np.array(input_data).reshape(-1, 1))
    st.write('Predicted water consumption for the next day:', prediction[0])

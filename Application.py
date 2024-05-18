import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the LSTM model and scaler
model_path = 'water_consumption_lstm_model.h5'  # Adjust the path accordingly
scaler_path = 'scaler.save'  # Adjust the path accordingly

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Function to preprocess input data
def preprocess_data(data, scaler, time_steps):
    scaled_data = scaler.transform(data)
    sequences = []
    for i in range(len(data) - time_steps):
        seq = scaled_data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Function to make predictions
def make_predictions(model, X):
    y_pred = model.predict(X)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    return y_pred_inverse

# Streamlit app
st.title('Water Consumption Prediction')

# Input parameters
time_steps = st.number_input('Time Steps', min_value=1, value=10)

# Upload data
uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display data
    st.subheader('Data Preview')
    st.write(data.head())

    # Preprocess data
    X = preprocess_data(data.values, scaler, time_steps)

    # Make predictions
    y_pred = make_predictions(model, X)

    # Display predictions
    st.subheader('Predictions')
    prediction_df = pd.DataFrame({
        'Timestamp': data.index[-len(y_pred):],
        'Predicted Consumption': y_pred[:, 0]
    })
    st.write(prediction_df)

    # Plot predictions
    st.subheader('Predicted Consumption Plot')
    st.line_chart(prediction_df.set_index('Timestamp'))

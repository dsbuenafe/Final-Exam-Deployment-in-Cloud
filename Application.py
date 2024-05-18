import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib

# Load the model and scaler
model_path = 'water_consumption_lstm_model.h5'  # Adjust the path accordingly
scaler_path = 'scaler.save'  # Adjust the path accordingly
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Function to create sequences
def create_sequences(data, time_steps=1):
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Function to predict water consumption
def predict_consumption(data, model, scaler, time_steps=10):
    # Scale the data
    scaled_data = scaler.transform(data)
    # Create sequences
    sequences = create_sequences(scaled_data, time_steps)
    # Predict
    predictions_scaled = model.predict(sequences)
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions_scaled)
    return predictions

# Streamlit UI
st.title('Water Consumption Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    st.write(df.head())

    # Predict button
    if st.button('Predict'):
        # Prepare data for prediction
        data = df.values.reshape(-1, 1)  # Reshape data for scaling
        predictions = predict_consumption(data, model, scaler)
        
        # Create a DataFrame with predictions
        timestamp_index = df.index[-len(predictions):]
        output_df = pd.DataFrame({
            'Timestamp': timestamp_index,
            'Predicted Values': predictions[:, 0]
        })

        # Display predictions
        st.write(output_df)

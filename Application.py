import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model
model_path = 'water_consumption_lstm_model.h5'
model = load_model(model_path)

# Load the scaler
scaler_path = 'scaler.save'
scaler = joblib.load(scaler_path)

# Function to prepare data for prediction
def prepare_data(data, time_steps=10):
    scaled_data = scaler.transform(data)
    sequences = []
    for i in range(len(data) - time_steps):
        seq = scaled_data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Streamlit App
st.title('Water Consumption Prediction')

# Input interface
st.header('Enter Historical Water Consumption Data')
data_input = st.text_area("Enter historical water consumption data separated by commas")

if st.button("Predict"):
    # Convert input to numpy array
    data = np.array(list(map(float, data_input.split(','))))

    # Prepare data for prediction
    sequences = prepare_data(data.reshape(-1, 1))

    # Make predictions
    predictions = model.predict(sequences)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)

    # Display predicted values
    st.subheader("Predicted Water Consumption:")
    st.write(predictions)

    # Plot predictions
    st.line_chart(predictions)

import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Function to load the LSTM model and scaler
@st.cache(allow_output_mutation=True)
def load_model_and_scaler(model_path, scaler_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to preprocess input data for prediction
def preprocess_input(input_data, time_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(input_data.reshape(-1, 1))
    sequences = []
    for i in range(len(scaled_data) - time_steps):
        seq = scaled_data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences), scaler

# Function to make predictions
def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Streamlit App
def main():
    st.title('Water Consumption Prediction')

    # Load LSTM model and scaler
    model_path = 'water_consumption_lstm_model.h5'
    scaler_path = 'scaler.save'
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # Input features
    st.sidebar.header('Input Parameters')
    time_steps = st.sidebar.slider('Time Steps', min_value=1, max_value=50, value=10)

    # Get input data from user
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with water consumption data", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        input_data['Date'] = pd.to_datetime(input_data['Date'])
        input_data.set_index('Date', inplace=True)
        st.subheader('Input Data')
        st.write(input_data.head())

        # Preprocess input data
        X, _ = preprocess_input(input_data.values, time_steps)
        
        # Make predictions
        predictions = make_predictions(model, X)

        # Inverse transform predictions to original scale
        predictions_inverse = scaler.inverse_transform(predictions.reshape(-1, 1))

        # Display predictions
        st.subheader('Predictions')
        prediction_df = pd.DataFrame({
            'Date': input_data.index[-len(predictions_inverse):],
            'Predicted Consumption': predictions_inverse.flatten()
        })
        st.write(prediction_df)

if __name__ == '__main__':
    main()

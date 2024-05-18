import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_path = 'water_consumption_lstm_model.h5'
model = load_model(model_path)

# Load your dataset for reference (used for fitting the scaler)
df = pd.read_csv('water_consumption.csv')

# Fit the scaler on the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[['feature1', 'feature2', 'feature3']])  # Replace with your actual feature columns

# Function to preprocess user input
def preprocess_input(user_input, scaler):
    scaled_input = scaler.transform(user_input)
    return scaled_input

def main():
    st.title("Water Consumption Prediction")

    # User input
    user_input_date = st.date_input("Select a date:", value=pd.to_datetime('today'))
    # Additional input fields can be added based on your dataset columns
    user_input_feature1 = st.number_input("Input feature 1 value:")
    user_input_feature2 = st.number_input("Input feature 2 value:")
    user_input_feature3 = st.number_input("Input feature 3 value:")

    # Preprocess the user input
    user_input = pd.DataFrame({
        'feature1': [user_input_feature1],
        'feature2': [user_input_feature2],
        'feature3': [user_input_feature3]
    })
    
    scaled_input = preprocess_input(user_input, scaler)

    # Reshape the input to (1, timesteps, features)
    reshaped_input = np.reshape(scaled_input, (1, 1, scaled_input.shape[1]))

    # Predict water consumption
    prediction = model.predict(reshaped_input)

    # Inverse transform the prediction to get the actual consumption value
    predicted_consumption = scaler.inverse_transform(prediction.reshape(-1, 1))

    st.subheader("Predicted Water Consumption:")
    st.write(predicted_consumption[0][0])

if __name__ == "__main__":
    main()


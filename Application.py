import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_path = '/content/drive/MyDrive/Final_Model_Emtech/water_consumption_lstm_model.h5'
model = load_model(model_path)

# Function to preprocess user input
def preprocess_input(user_input, scaler):
    # Perform any necessary preprocessing on the user input
    # For example, convert the input into the appropriate format

    # In this example, assuming user_input is a DataFrame with the same features as your original dataset
    # You may need to adjust this based on your actual input

    # Normalize the user input using the same scaler used during training
    scaled_input = scaler.transform(user_input)

    return scaled_input

def main():
    st.title("Water Consumption Prediction")

    # Load your dataset for reference
    # Replace 'your_dataset.csv' with the path to your dataset
    df = pd.read_csv('/content/drive/MyDrive/Final_Model_Emtech/water_consumption.csv')

    # Assuming 'Date' is one of the columns for user input
    user_input_date = st.date_input("Select a date:", value=pd.to_datetime('today'))

    # Additional input fields can be added based on your dataset columns

    # Preprocess the user input
    user_input = pd.DataFrame({'Date': [user_input_date]})
    scaler = MinMaxScaler(feature_range=(0, 1))  # Use the same scaler as used during training
    scaled_input = preprocess_input(user_input, scaler)

    # Predict water consumption
    prediction = model.predict(np.reshape(scaled_input, (1, scaled_input.shape[0], scaled_input.shape[1])))

    # Inverse transform the prediction to get the actual consumption value
    predicted_consumption = scaler.inverse_transform(prediction.reshape(-1, 1))

    st.subheader("Predicted Water Consumption:")
    st.write(predicted_consumption[0][0])

if __name__ == "__main__":
    main()

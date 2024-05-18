import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

model_path = 'water_consumption_lstm_model.h5'
model = load_model(model_path)

scaler_path = 'scaler.save'
scaler = joblib.load(scaler_path)

def preprocess_input(user_input, scaler):

    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

def main():
    st.title("Water Consumption Prediction")
    df = pd.read_csv('water_consumption.csv')
    user_input_date = st.date_input("Select a date:", value=pd.to_datetime('today'))
    user_input = pd.DataFrame({'Date': [user_input_date]})
    user_input['Date'] = pd.to_datetime(user_input['Date']).map(pd.Timestamp.toordinal)
    scaled_input = preprocess_input(user_input, scaler)
    prediction = model.predict(np.reshape(scaled_input, (1, scaled_input.shape[0], scaled_input.shape[1])))
    predicted_consumption = scaler.inverse_transform(prediction.reshape(-1, 1))
    st.subheader("Predicted Water Consumption:")
    st.write(predicted_consumption[0][0])

if __name__ == "__main__":
    main()

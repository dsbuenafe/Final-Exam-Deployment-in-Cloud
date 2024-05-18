import streamlit as st
import pandas as pd
import tensorflow as tf

# Set up the Streamlit app
st.write("Final Exam: Deployment in Cloud")
st.write("Name: Dhafny Buenafe and Mayah Catorce")
st.write("Section: CPE32S3")
st.write("Instructor: Engr. Roman Richard")

# Load the pre-trained model
model = tf.keras.models.load_model('water_consumption_lstm_model.h5')

# Define function to make predictions
def predict_consumption(previous_consumption):
    # Reshape the input to match the model's expected input shape
    previous_consumption = np.array([[previous_consumption]])  # Assuming the model expects a 2D array
    predicted_consumption = model.predict(previous_consumption)
    return predicted_consumption[0][0]  # Adjust based on your model's output shape

# Streamlit app
def main():
    st.title('Consumption Prediction App')

    # User input for previous consumption
    previous_consumption = st.number_input('Enter your previous consumption:', value=0.0)

    # Predict consumption
    if st.button('Predict Consumption'):
        # Make predictions
        predicted_consumption = predict_consumption(previous_consumption)
        st.write('Predicted consumption:', predicted_consumption)

if __name__ == "__main__":
    main()

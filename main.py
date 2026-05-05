import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and transformers
model = joblib.load('train_arrival_model.pkl')
le_source = joblib.load('le_source.pkl')
le_dest = joblib.load('le_dest.pkl')
le_type = joblib.load('le_type.pkl')

# Application title
st.title("🚂 Train Arrival Time Prediction (Egypt)")
st.write("Enter the trip details to predict the expected arrival time.")

# List of available cities and train types
sources = le_source.classes_
destinations = le_dest.classes_
train_types = le_type.classes_

# User Interface
col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("From City:", sources)
    train_type = st.selectbox("Train Type:", train_types)

with col2:
    destination = st.selectbox("To City:", destinations)
    distance = st.number_input("Approximate Distance (km):", min_value=10, max_value=1000, value=200)

if st.button("Predict Time"):
    try:
        # Transform inputs
        s_enc = le_source.transform([source])[0]
        d_enc = le_dest.transform([destination])[0]
        t_enc = le_type.transform([train_type])[0]
        
        # Prediction
        input_data = np.array([[s_enc, d_enc, distance, t_enc]])
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success(f"⏱️ Estimated travel time is: {prediction:.2f} hours")
        
        # Add a simple note
        st.info("Note: This prediction is based on simplified data and may vary in reality.")
        
    except Exception as e:
        st.error(f"An error occurred: Please ensure you select valid cities present in the database.")
import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load preprocess and model from MLflow
# Load preprocessor
scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')

    # Add user input components for 13 features
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex (1 = male, 0 = female)', [1, 0])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=250, value=120)
    chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)', [0, 1])
    restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250, value=150)
    exang = st.selectbox('Exercise Induced Angina (1 = yes, 0 = no)', [0, 1])
    oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (0-3)', [0, 1, 2, 3])
    
    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()



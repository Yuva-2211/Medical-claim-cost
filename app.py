import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer


model = joblib.load('/Users/yuvashankarnarayana/Documents/Helath_insurance_LM/trained_model(tunned).pkl')  
imputer = joblib.load('/Users/yuvashankarnarayana/Documents/Helath_insurance_LM/imputer.pkl')  

SEX_MAP = {'male': 0, 'female': 1}
SMOKER_MAP = {'yes': 1, 'no': 0}
DIABETES_MAP = {'yes': 1, 'no': 0}
REGULAR_EX_MAP = {'yes': 1, 'no': 0}

# Streamlit form
def user_input():
    st.title("Claim Cost Prediction")

    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
    diabetes = st.selectbox("Diabetes", ["yes", "no"])
    regular_ex = st.selectbox("Regular Exercise", ["yes", "no"])

# input data to data frame
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [SEX_MAP[sex]],
        "weight": [weight],
        "no_of_dependents": [no_of_dependents],
        "smoker": [SMOKER_MAP[smoker]],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetes": [DIABETES_MAP[diabetes]],
        "regular_ex": [REGULAR_EX_MAP[regular_ex]]
    })

    return input_data


def predict_claim_cost(input_data):
    # Impute missing values (if any & )
    input_data_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
    
    # prediction Part
    prediction = model.predict(input_data_imputed)
    return prediction[0]  

def main():

    input_data = user_input()


    if st.button("Predict Insurance Cost"):
        result = predict_claim_cost(input_data)
        st.write(f"Predicted Claim Cost: {result:.2f}")


if __name__ == '__main__':
    main()

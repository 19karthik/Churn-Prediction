import streamlit as st
import pickle
import numpy as np
import pandas as pd
 
# Load your trained model
model_path = 'Churn.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the expected columns
columns_path = 'model_columns.pkl'
with open(columns_path, 'rb') as file:
    expected_columns = pickle.load(file)

def u_input():
    # Define four columns for the layout
    col1, col2, col3, col4 = st.columns(4)

    # Distribute the inputs equally among the four columns
    # Column 1 inputs
    with col1:
        cust_id = st.text_input('Customer ID')
        gender = st.selectbox('Gender', ('Male', 'Female'))
        SeniorCitizen = st.selectbox('Senior Citizen', ('0', '1'))
        Partner = st.selectbox('Partner', ('Yes', 'No'))

    # Column 2 inputs
    with col2:
        Dependents = st.selectbox('Dependents', ('Yes', 'No'))
        PhoneService = st.selectbox('Phone Service', ('Yes', 'No'))
        MultipleLines = st.selectbox('Multiple Lines', ('Yes', 'No', 'No phone'))
        InternetService = st.selectbox('Internet Service', ('DSL', 'No', 'Fiber optic'))

    # Column 3 inputs
    with col3:
        OnlineSecurity = st.selectbox('Online Security', ('Yes', 'No'))
        OnlineBackup = st.selectbox('Online Backup', ('Yes', 'No'))
        DeviceProtection = st.selectbox('Device Protection', ('Yes', 'No'))
        TechSupport = st.selectbox('Tech Support', ('Yes', 'No'))

    # Column 4 inputs
    with col4:
        StreamingTV = st.selectbox('Streaming TV', ('Yes', 'No'))
        StreamingMovies = st.selectbox('Streaming Movies', ('Yes', 'No'))
        Contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        PaperlessBilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Credit card (automatic)', 
        'Bank transfer (automatic)'), index=1)

    # Sliders in a separate row
    tenure = st.slider('Tenure', 0, 72, 1)
    MonthlyCharges = st.slider('Monthly Charges', 0.0, 150.0, 1.0)
    TotalCharges = st.slider('Total Charges', 0.0, 10000.0, 1.0)

    data = {
        'customerID': [cust_id],
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    features = pd.DataFrame(data)
    return features

# Main Streamlit app
st.title('Customer Churn Prediction')

# Get user input
input_df = u_input()

# Perform encoding
input_df_encoded = pd.get_dummies(input_df, 
    columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod'])

# Ensure all expected columns are present
for col in expected_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Reorder columns to match training data
input_df_encoded = input_df_encoded[expected_columns]

# Prediction
prediction = model.predict(input_df_encoded)

# Display prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

# Display input features
st.subheader('Input Features')
st.write(input_df)

# Prediction probability
prediction_proba = model.predict_proba(input_df_encoded)
st.subheader('Prediction Probability')
st.write(prediction_proba)

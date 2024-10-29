from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

@st.cache_resource(show_spinner="Fetching the model")
def loading_model():
    loaded_scaler = joblib.load('STR_model_building/scaler.joblib')
    loaded_model = joblib.load('STR_model_building/model.joblib')
    loaded_encoder = joblib.load('STR_model_building/encoder.joblib')
    return loaded_scaler, loaded_model, loaded_encoder

st.title("Suspicious Transaction Detection App")

st.write("This app detects any suspicious financial transaction")

left, middle, right = st.columns(3)

date = left.date_input("Select the date of transaction")

time = middle.time_input("Select the time of transaction")

transaction_amount = right.number_input("Enter the amount of transaction", min_value=50.0, step=0.01, format="%.2f")

left1, middle1, right1 = st.columns(3)

transaction_type = left1.selectbox("Transaction Type", ['Cash Deposit', 'Online Payment', 'Wire Transfer', 'Withdrawal'])

currency = middle1.selectbox("Currency of transaction", ['NGN', 'GBP', 'EUR', 'USD'])

age = right1.number_input("Age of customer", min_value=18)

left2, middle2, right2 = st.columns(3)

occupation = left2.selectbox("Occupation of customer", ['Engineer', 'Teacher', 'Business Owner', 'Doctor', 'Lawyer'])

balance = middle2.number_input("Enter Account Balance", min_value=0.0, step=0.01, format="%.2f")

avg_trans_amount = right2.number_input("Average transaction amount", min_value=0.0, step=0.01, format="%.2f")

left3, middle3, right3 = st.columns(3)

frequency = left3.number_input("Transaction frequency", min_value=1)

tenure = middle3.number_input("Account Tenure", help="Enter the number of years the account has been registered with the bank", min_value=1)

origin = right3.selectbox("Country of Origin", ['Nigeria', 'Germany', 'UK', 'USA', 'China'], help="Country where transaction was initialized")

left4, right4 = st.columns(2)

destination = left4.selectbox("Country of Destination", ['Nigeria', 'Germany', 'UK', 'USA', 'China'], help="Country where transaction was recieved")

third_party = right4.radio("Does transaction involves multiple accounts?", ['Yes', 'No'])

left5, right5 = st.columns(2)

linked_acc = left5.radio("Does transaction involve linked accounts?", ['Yes', 'No'])

pep_status = right5.radio("Is the customer a Political Exposed Person (PEP)?", ['Yes', 'No'])

def prediction(date, time, transaction_amount, transaction_type, currency, age, occupation, balance, avg_trans_amount, frequency, tenure, origin, destination, third_party, linked_acc, pep_status):
    scaler, model, encoder = loading_model()
    dic = {
        'Transaction Amount': [transaction_amount],
        'Transaction Type': [transaction_type],
        'Currency': [currency], 
        'Customer Age': [age], 
        'Customer Occupation': [occupation],
        'Account Balance': [balance],
        'Average Transaction Amount': [avg_trans_amount],
        'Transaction Frequency': [frequency],
        'Account Tenure': [tenure],
        'Country of Origin': [origin],
        'Country of Destination': [destination],
        'Third-Party Involvement': [third_party],
        'Linked Accounts': [linked_acc],
        'PEP Status': [pep_status],
        'date': [date],
        'time': [time]
    }
    
    user_dataframe = pd.DataFrame(dic)

    user_dataframe['date'] = pd.to_datetime(user_dataframe['date'])
    user_dataframe['time'] = user_dataframe['time'].apply(lambda t: datetime.combine(datetime.today().date(), t))
    user_dataframe['time'] = pd.to_datetime(user_dataframe['time'])

    user_dataframe['Transaction_hour'] = user_dataframe['time'].dt.hour
    user_dataframe['Transaction_day_of_week'] = user_dataframe['date'].dt.dayofweek
    user_dataframe['Transaction_day_of_month'] = user_dataframe['date'].dt.day

    user_dataframe['Third-Party Involvement'] = user_dataframe['Third-Party Involvement'].map({'Yes': 1, 'No': 0})
    user_dataframe['Linked Accounts'] = user_dataframe['Linked Accounts'].map({'Yes': 1, 'No': 0})
    user_dataframe['PEP Status'] = user_dataframe['PEP Status'].map({'Yes': 1, 'No': 0})

    user_data = user_dataframe.drop(columns = ['date', 'time'])

    numerical_columns = ['Transaction Amount', 'Customer Age', 'Account Balance', 'Average Transaction Amount', 'Transaction Frequency', 'Account Tenure', 'Third-Party Involvement', 'Linked Accounts', 'PEP Status', 'Transaction_hour', 'Transaction_day_of_week', 'Transaction_day_of_month']
    categorcial_columns = ['Transaction Type', 'Currency', 'Customer Occupation', 'Country of Origin', 'Country of Destination']

    user_data[numerical_columns] = scaler.transform(user_data[numerical_columns])

    encoded_cols = list(encoder.get_feature_names_out(categorcial_columns))
    user_data[encoded_cols] = encoder.transform(user_data[categorcial_columns])
    user_data = user_data.drop(columns = categorcial_columns)

    pred = model.predict(user_data)
    pred_proba = model.predict_proba(user_data)
    
    if pred == 1:
        return "This is a Suspicious Transaction with a {:.2f}% probability.\nSending alert...".format(pred_proba[:, 0][0] * 100)
    
    else:
        return "This is not a Suspicious Transaction with a {:.2f}% probability.".format(pred_proba[:, 1][0] * 100)

if st.button("Detect Transaction", key= 'my_button'):
    value = prediction(date, time, transaction_amount, transaction_type, currency, age, occupation, balance, avg_trans_amount, frequency, tenure, origin, destination, third_party, linked_acc, pep_status)
    st.write(value)
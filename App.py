import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Set page config
st.set_page_config(page_title='Medical Insurance Prediction', page_icon=':hospital:')



# Load and preprocess the data
medical_df = pd.read_csv('insurance.csv')
medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

# Split the data
X = medical_df.drop('charges',axis=1)
y = medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train the model
lg = LinearRegression()
lg.fit(X_train,y_train)

# App title
st.title("Medical Insurance Prediction Model")

# User inputs
st.header('Enter Person Details')
age = st.number_input('Age', min_value=1, max_value=100)
sex = st.selectbox('Sex', options=['Male', 'Female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0)
children = st.number_input('Number of Children', min_value=0, max_value=10)
smoker = st.selectbox('Smoker', options=['Yes', 'No'])
region = st.selectbox('Region', options=['Southeast', 'Southwest', 'Northwest', 'Northeast'])

# Convert sex, smoker, and region to binary
sex = 0 if sex == 'Male' else 1
smoker = 0 if smoker == 'Yes' else 1
region = {'Southeast':0, 'Southwest':1, 'Northwest':2, 'Northeast':3}[region]

# Predict button
if st.button('Predict'):
    # Make prediction
    input_data = np.array([age, sex, bmi, children, smoker, region])
    prediction = lg.predict(input_data.reshape(1,-1))

    # Display prediction
    st.success(f'Medical Insurance for this person is: {prediction[0]}')

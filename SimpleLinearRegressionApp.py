import streamlit as st
import pickle
import numpy as np

#load model
model = pickle.load(open(r'C:\Users\Lenovo\Desktop\ML\SimpleLinearRegression\linear_regression_model.pkl','rb'))

#title
st.title("Salary Prediction App")

#description
st.write("This app predicts the salary based on the years of experience")

#input
years_experience = st.number_input("Enter the years of experience", min_value = 0.0, max_value =50.0, value = 1.0, step = 0.5 )

if st.button("Predict Salary"):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)
    
    #display result
    st.success(f'The predicted salary for {years_experience} years of experience is :${prediction[0]:,.2f}')
st.write("The Model was developed by using Simple Linear Regression by a dataset having salary and experience column")
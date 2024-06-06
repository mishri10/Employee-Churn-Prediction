#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Organize the layout
import streamlit as st
import pandas as pd
import pickle

st.sidebar.title("Select")
app_selection = st.sidebar.selectbox("Select", ["Single Prediction"])

if app_selection == "Single Prediction":
    # Load the pre-trained model
    with open('data.pkl','rb') as f:
            data = pickle.load(f)
            
    with open('pipeline.pkl','rb') as f:
        pipeline = pickle.load(f)
    
    # Function to show prediction result
    def show_prediction():
        p1 = float(e1)
        p2 = float(e2)
        p3 = float(e3)
        p4 = float(e4)
        p5 = float(e5)
        p6 = float(e6)
        p7 = float(e7)
        p8 = str(e8)
        p9 = str(e9)
    
        sample = pd.DataFrame({
            'satisfaction_level': [p1],
            'last_evaluation': [p2],
            'number_project': [p3],
            'average_montly_hours': [p4],
            'time_spend_company': [p5],
            'Work_accident': [p6],
            'promotion_last_5years': [p7],
            'departments': [p8],
            'salary': [p9]
        })
    
        result = pipeline.predict(sample)
        
        if result == 1:
            st.write("An employee may leave the organization.")
        else:
            st.write("An employee may stay with the organization.")
    
    # Streamlit app
    st.title("!! Employee Churn Prediction Using Machine Learning !!")
    
    # Employee data input fields
    
    
    e1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
    e2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
    e3 = st.slider("Number of projects assigned to", 1, 10, 5)
    e4 = st.slider("Average monthly hours worked", 50, 300, 150)
    e5 = st.slider("Time spent at the company", 1, 10, 3)
    e6 = st.radio("Whether they have had a work accident", [0, 1])
    e7 = st.radio("Whether they have had a promotion in the last 5 years", [0, 1])
    
    options = ('sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
           'RandD', 'accounting', 'hr', 'management')
    e8 = st.selectbox("Department name", options)
    
    options1 = ('low','meduim','high')
    e9 = st.selectbox("Salary category", options1)
    
    # Predict button
    if st.button("Predict"):
        show_prediction()



# In[ ]:





import streamlit as st
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random
import pickle

model = pickle.load(open('recommender.pkl', 'rb'))

st.subheader('Ad recommender')
with st.form('form1'):

    col1, col2 = st.columns(2)
    with col1:
        time = st.selectbox('Time', ("Morning", "Afternoon", "Evening", "Midnight"))
        age = st.number_input("Age", min_value=1, max_value=100, step=1)
        income = st.number_input('Income', min_value=0, max_value=100000, step=1)
        area = st.selectbox('Area', ('Rural', 'Suburban', 'Urban'))

        
    with col2:
        sex_ = st.radio('Sex', ["Male", "Female"])
        personality = st.radio('Personality', ["Introvert", "Extrovert"])
        day = st.radio('Day', ["Weekday", "Weekend"])

    submitted = st.form_submit_button("Submit")


if submitted == True:

    if sex_ == "Male":
        sex = 1
    else:
        sex = 0

    if personality == "Extrovert":
        extrovert = 1
    else:
        extrovert = 0

    if day == "Weekday":
        weekday = 1
    else:
        weekday = 0

    X_dict = {'Age': age, 'Male': sex, 'Income': income, "Area": area, 'Extrovert': extrovert, 'Weekday': weekday, "Time": time}
    X = pd.DataFrame(X_dict, index=[0])
    X = pd.get_dummies(X, dtype='int')

    df_template = pd.DataFrame(None, columns=['Age', 'Male', 'Income', 'Extrovert', 'Weekday', 'Area_Rural',
    'Area_Suburban', 'Area_Urban', 'Time_Afternoon', 'Time_Evening',
    'Time_Midnight', 'Time_Morning'])

    df_merged = pd.concat([df_template, X]).fillna(0)
    predicted = model.predict(df_merged)

    st.write('Recommended Ad')
    st.write(pd.DataFrame(predicted, columns=['Ad spot', 'Ad variant']))
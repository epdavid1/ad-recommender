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

    col1, col2, col3 = st.columns(3)
    with col1:
        time = st.selectbox('Time', ("Morning", "Afternoon", "Evening", "Midnight"))
        age = st.number_input("Age", min_value=1, max_value=100, step=1)
        income = st.number_input('Income', min_value=0, max_value=100000, step=1)
        area = st.selectbox('Area', ('Rural', 'Suburban', 'Urban'))

        
    with col2:
        sex_ = st.radio('Sex', ["Male", "Female"])
        personality = st.radio('Personality', ["Introvert", "Extrovert"])
        day = st.radio('Day', ["Weekday", "Weekend"])
        bank = int(st.checkbox('Bank account linked'))

    with col3:
        with st.container():
            st.write('Billers used')
            biller_meralco = int(st.checkbox('Meralco'))
            biller_pldt = int(st.checkbox('PLDT'))
            biller_maynilad = int(st.checkbox('Maynilad'))
        

        loan = st.number_input('Total outstanding loan', min_value=0, max_value=100000, step=1)
        trans_ave = st.number_input('Ave. transaction count', min_value=0, max_value=1000, step=1)
        trans_sd = st.number_input('SD transaction count', min_value=0, max_value=1000, step=1)

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

    columns = ['Age', 'Male', 'Income', 'Extrovert', 'Weekday', 'Biller_Meralco',
       'Biller_PLDT', 'Biller_Maynilad', 'Total Outstanding Loan',
       'Linked Bank Account', 'Ave Transaction Count', 'SD Transaction Count',
       'Area', 'Time']

    values = [age, sex, income, extrovert, weekday, biller_meralco, biller_pldt, biller_maynilad, loan, bank, trans_ave, trans_sd, area, time]

    X_dict = dict(zip(columns,values))

    X = pd.DataFrame(X_dict, index=[0])
    X = pd.get_dummies(X, dtype='int')

    df_template = pd.DataFrame(None, columns=['Age', 'Male', 'Income', 'Extrovert', 'Weekday', 'Biller_Meralco',
       'Biller_PLDT', 'Biller_Maynilad', 'Total Outstanding Loan',
       'Linked Bank Account', 'Ave Transaction Count', 'SD Transaction Count',
       'Area_Rural', 'Area_Suburban', 'Area_Urban', 'Time_Afternoon',
       'Time_Evening', 'Time_Midnight', 'Time_Morning'])

    df_merged = pd.concat([df_template, X]).fillna(0)
    predicted = model.predict(df_merged)
    predicted_df = pd.DataFrame(predicted, columns=['Push notif', 'Banner', 'Popup'])
    predicted_df = predicted_df.rename_axis('CustomerID')

    st.write('Recommended Ad')
    st.write(predicted_df)


    reward_dict = {'Sports': 30, 'Fashion':20, 'Finance': 25, 'Travel': 25}

    ad_variants = ['Sports', 'Fashion', 'Finance', 'Travel']
    actual_df = predicted_df.copy()
    actual_df['Push notif'] = random.choice(ad_variants)
    actual_df['Banner'] = random.choice(ad_variants)
    actual_df['Popup'] =  random.choice(ad_variants)
    equal_df = predicted_df == actual_df


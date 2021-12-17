import streamlit as st
import numpy as np
import pandas as pd
import joblib 

def run_ml_app() :
    classifier = joblib.load('data/best_model.pkl')
    scaler_X = joblib.load('data/scaler_X.pkl')

    st.subheader('데이터를 입력하면 당뇨병을 예측!')

    # 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin'
    # 'BMI', 'DiabetesPedigreeFunction', 'Age'
     
    preg = st.number_input('임신횟수', min_value=0, max_value=17)
    glucose = st.number_input('Glucose', min_value= 44, max_value=199)
    bp = st.number_input('BloodPressure', min_value=24, max_value=122)
    skin = st.number_input('SkinThickness', min_value=7, max_value=99)
    insulin = st.number_input('Insulin', min_value=0, max_value=846)
    BMI = st.number_input('BMI', min_value=18.2, max_value=67.1)
    DPF = st.number_input('DiabetesPedigreeFunction', min_value=0.078, max_value=2.42)
    age = st.number_input('Age', min_value=21, max_value=81)

    if st.button('결과보기') :
        new_data = np.array([preg, glucose, bp, skin, insulin, BMI, DPF, age])  

        new_data = new_data.reshape(1,8)
        new_data = scaler_X.transform(new_data)

        y_pred = classifier.predict(new_data)

        print(y_pred)
        if y_pred[0] == 0 :
            st.write('예측 결과는, 당뇨병이 아닙니다.')
        else :
            st.write('예측 결과는, 당뇨병입니다.')
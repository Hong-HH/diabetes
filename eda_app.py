import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def run_eda_app() :
    st.subheader('데이터 분석')

    df = pd.read_csv('data/diabetes.csv')

    si = SimpleImputer(missing_values=0)

    df_si = si.fit_transform(df[['Glucose','BloodPressure', 'SkinThickness', 'BMI']])

    df[['Glucose','BloodPressure', 'SkinThickness', 'BMI']] = df_si
    
    st.dataframe(df)

    st.text('Nan 데이터 확인')

    st.dataframe(df.isna().sum())

    st.text('각 컬럼별 히스토그램 확인')

    ## 이거 워닝 뜬다.
    # loc로도 된대 
    #fig1 = plt.figure()
    df.hist(figsize=(10,8))
    plt.show()
    st.pyplot()

    # 빈의 갯수를 조절하는 슬라이더
    bins = st.slider('bin의 갯수 조절', min_value=10, max_value=50)

    selected_columns = st.selectbox('컬럼을 선택하세요', df.columns)
    fig1 = plt.figure()
    df[selected_columns].hist(bins=bins)
    st.pyplot(fig1)


    # describe
    st.subheader('각 컬럼별 통계치')
    st.dataframe(df.describe())

    

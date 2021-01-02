import streamlit as st
import pandas as pd
from sklearn.datasets import load_boston
import models

# streamlit run app.py
# Page layout
st.set_page_config(page_title='데이터 예측')

st.write("""
# 데이터 예측
이 사이트는 RandomForestRegressor 모델을 기반으로 단일 목표값에 대한 중요 변수를 알려 드립니다.(v0.1.0-beta)
""")

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('CSV 파일 불러오기'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Sidebar - Specify parameter settings
with st.sidebar.header('데이터 분할'):
    split_size = st.sidebar.slider('데이터 분할 비율(학습 데이터 %)', 10, 90, 60, 5)

#with st.sidebar.subheader('Model'):
#    model_type = st.sidebar.selectbox(
#        "Prediction Model",
#        ("Random forest", "Logistic regression")
#    )
#
#    #parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
model_type = "Random forest"

with st.sidebar.subheader('교차 검증'):
    parameter_cv = st.sidebar.slider('폴더 수 (cv)', 2, 10, 3, 1)
    #parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])

with st.sidebar.subheader('손실 함수'):
    parameter_criterion = st.sidebar.select_slider('종류', options=['mse', 'mae'])

# Main
# Displays the dataset
st.subheader('1. 데이터 세트')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. 데이터 프레임(5줄)**')
    st.write(df.head())
    models.model(df, model_type, parameter_criterion, parameter_cv, split_size)

else:
    st.info('데이터를 불러오거나 예제 데이터를 사용해 보세요. ')
    if st.button('예제 데이터 불러오기'):
        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='PRICES')
        df = pd.concat([X, Y], axis=1)

        st.markdown('보스턴 집 값 데이터')
        st.markdown('**1.1. 데이터 프레임(5줄)**')
        st.write(df.head())

        models.model(df, model_type, parameter_criterion,parameter_cv, split_size)

        st.write("""
        ** 보스턴 집 값 데이터 변수 설명 **
        
        CRIM: 범죄율
        INDUS: 비소매상업지역 면적 비율
        NOX: 일산화질소 농도
        RM: 주택당 방 수
        LSTAT: 인구 중 하위 계층 비율
        B: 인구 중 흑인 비율
        PTRATIO: 학생/교사 비율
        ZN: 25,000 평방피트를 초과 거주지역 비율
        CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
        AGE: 1940년 이전에 건축된 주택의 비율
        RAD: 방사형 고속도로까지의 거리
        DIS: 직업센터의 거리
        TAX: 재산세율
        """)
# ...existing code...
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

col1, col2 = st.columns([4, 1])
with col2:
    st.image("https://yt3.googleusercontent.com/ytc/AIdro_lpQ3F79hQhruFACNbiC06LnBXo7SNAMtnT3UFqZaOSbg=s900-c-k-c0x00ffffff-no-rj", width=50,)
with col1:
    st.title("안동중앙고 소프웨어동아리")
ad = pd.read_csv("AD.csv")
st.header("산불 예측 모델")

X = ad.drop('피해면적_등급', axis=1)
y = ad['피해면적_등급']

# 1) 숫자/범주형 컬럼 자동 분리
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

# 2) 전처리 파이프라인 설정
model_pipe = Pipeline(steps=[
    ("clf", LogisticRegression(
        max_iter=5000,
        solver='saga',
        random_state=42
    ))
])

# 모델 학습 함수
def train_logistic_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = model_pipe
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

model, test_score = train_logistic_model(X, y)

st.sidebar.header("입력값 (예측용)")

# 숫자 입력 제한 제거
input_data = {}
for col in X.columns:
    col_vals = X[col]
    if col in numeric_cols:
        # 기본값은 데이터 중앙값
        finite_vals = pd.to_numeric(col_vals, errors="coerce").dropna()
        if len(finite_vals) == 0:
            mean_v = 0.0
        else:
            mean_v = float(finite_vals.median())

        input_data[col] = st.sidebar.number_input(
            col,
            value=float(mean_v),
            step=1.0,            
            format="%.2f"        
        )

if st.sidebar.button("산불 확률 예측"):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    probs = model.predict_proba(input_df)

    classes = model.named_steps['clf'].classes_
    prob_pct = (probs[0] * 100).round(2)

    result = pd.DataFrame({
        "등급": classes,
        "확률(%)": prob_pct
    })

    st.subheader("예측 결과")
    st.write("정확도:", round(test_score * 100, 2), "%")
    st.table(result.set_index("등급"))

    pred_class = classes[np.argmax(probs[0])]
    max_prob = float(prob_pct.max())
    if pred_class == 0:
        st.success(f"예측 등급: {pred_class}")
    elif pred_class == 1:
        st.warning(f"예측 등급: {pred_class}")
    else:
        st.error(f"예측 등급: {pred_class}")
        
    if pred_class == 0:
        warning_g = "낮음"
    elif pred_class == 1:
        warning_g = "보통"
    else:
        warning_g = "높음"
    
    st.info(f"해당 조건에서 '{pred_class}'({warning_g}) 등급일 확률은 {max_prob:.2f}% 입니다.")
st.markdown("<span style='font-size:10px;'>made by 김규태, 손창수, 이상우, 이지훈</span>", unsafe_allow_html=True)

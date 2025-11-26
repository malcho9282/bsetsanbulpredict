import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
from folium import Map, CircleMarker, LayerControl
from branca.colormap import LinearColormap
import streamlit.components.v1 as components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from datetime import datetime

# Streamlit UI 설정
st.set_page_config(page_title="산불 예측 및 위험도", layout="wide")
col1, col2 = st.columns([4, 1])
with col2:
    st.image("https://yt3.googleusercontent.com/ytc/AIdro_lpQ3F79hQhruFACNbiC06LnBXo7SNAMtnT3UFqZaOSbg=s900-c-k-c0x00ffffff-no-rj", width=50,)
with col1:
    st.title("안동중앙고 소프웨어동아리")
ad = pd.read_csv("AD.csv")
st.header("실시간 일자료 기반 산불 예측")

# API 및 인증키
KMA_SFCDD_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php"
auth_key = "lUd7Swg_SC-He0sIP2gvzA"

# 사용자 입력 UI
st.subheader("예측 설정")
tm = st.text_input(
    "예측 날짜 (YYYYMMDD)",
    value=datetime.today().strftime("%Y%m%d"),
    placeholder="예: 20251125"
)

csv_path = "AD.csv"

# 관측소 목록
STATIONS = {
    108: ("서울", 37.571, 126.965),
    133: ("대전", 36.372, 127.374),
    143: ("대구", 35.877, 128.656),
    156: ("광주", 35.173, 126.891),
    159: ("부산", 35.104, 129.032),
    112: ("인천", 37.474, 126.625),
    184: ("제주", 33.514, 126.529),
    105: ("강릉", 37.751, 128.890),
    283: ("안동", 36.5665, 128.7292),
    146: ("전주", 35.8206, 127.148),
    138: ("포항", 36.032, 129.381),
    168: ("여수", 34.735, 127.550),
    152: ("울산", 35.580, 129.334),
    140: ("군산", 35.967, 126.736),
    239: ("세종", 36.51, 127.25),
    221: ("제천", 37.1593, 128.1906),
    165: ("목포", 34.811, 126.382),
    131: ("청주", 36.642, 127.443),
    277: ("영덕", 36.4156, 129.365),
}

stn_choices = list(STATIONS.keys())

# 학습 데이터 로드 (캐싱)
@st.cache_data(show_spinner=False)
def load_ad(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


ad = load_ad(csv_path)

# 타겟 변수 체크
if "피해면적_등급" not in ad.columns:
    st.error("AD.csv에 '피해면적_등급' 열이 없습니다.")
    st.stop()

# 입력 변수 선택
feature_cols = [
    "평균기온(°C)", "일강수량(mm)", "평균 풍속(m/s)", "평균 상대습도(%)",
    "평균 증기압(hPa)", "평균 현지기압(hPa)", "합계 일사량(MJ/m2)",
    "일 최심신적설(cm)", "평균 전운량(1/10)", "합계 대형증발량(mm)"
]

X = ad[feature_cols]
y = ad["피해면적_등급"]

# 모델 생성
model_pipe = Pipeline([
    ("clf", LogisticRegression(max_iter=5000, solver="saga", random_state=42))
])

@st.cache_resource(show_spinner=True)
def train_logistic_model(X, y):
    # train/test 분리 후 학습
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_pipe
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)

model, test_score = train_logistic_model(X, y)
classes = model.named_steps["clf"].classes_
st.sidebar.write("모델 정확도:", f"{test_score*100:.2f}%")

# 텍스트 인코딩 처리
def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "euc-kr", "cp949"):
        try: return b.decode(enc)
        except: continue
    return b.decode("utf-8", errors="ignore")

# API 데이터 → DataFrame 변환
def parse_sfcdd_table(text: str) -> pd.DataFrame:
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), sep=r"\s+|,", engine="python", header=None)
    df.columns = ["TM", "STN"] + [f"C{i}" for i in range(2, len(df.columns))]
    return df

# API 호출 함수
def fetch_sfcdd(tm, key):
    r = requests.get(KMA_SFCDD_URL, params={"tm": tm, "stn": 0, "authKey": key, "help": 0}, timeout=20)
    txt = decode_bytes(r.content)
    return parse_sfcdd_table(txt), txt[:600]

# API 열명 → 학습데이터 열명 매핑
ALIAS = {
    "평균기온(°C)": ["TA_AVG","C10"], "일강수량(mm)": ["RN_DAY","C38"], "평균 풍속(m/s)": ["WS_AVG","C2"],
    "평균 상대습도(%)": ["HM_AVG","C18"], "평균 증기압(hPa)": ["PV_AVG","C21"], "평균 현지기압(hPa)": ["PA_AVG","C25"],
    "합계 일사량(MJ/m2)": ["SI_DAY","C35"], "일 최심신적설(cm)": ["SD_NEW","C47"], "평균 전운량(1/10)": ["CA_TOT","C31"],
    "합계 대형증발량(mm)": ["EV_L","C23"]
}

# 결측값 대체 기준 계산
col_defaults = {c: float(X[c].median()) for c in X.columns}

# 피처 추출
def pick_by_alias(df, alias):
    for key in alias:
        m = df.filter(regex=key, axis=1)
        if not m.empty:
            return pd.to_numeric(m.iloc[:,0], errors="coerce")
    return None

def sfcdd_to_features(df):
    out = {}
    for feat, alias in ALIAS.items():
        s = pick_by_alias(df, alias)
        out[feat] = float(s.mean()) if isinstance(s, pd.Series) and s.notna().any() else col_defaults[feat]
    return out

# 위험도 레벨 설명 생성
def risk_description(pred_class, probs_row, classes_array):
    if str(pred_class) == "0": return "안전"
    if str(pred_class) == "2": return "매우 위험"
    return "매우 위험" if max(probs_row) >= 0.75 else "위험"


# 실행 버튼
if st.button("모델 예측 보기", type="primary"):

    df_all, _ = fetch_sfcdd(tm, auth_key)

    df_all["STN"] = pd.to_numeric(df_all["STN"], errors="coerce").astype("Int64")

    rows_model, rows_map = [], []

    # 관측소별 데이터 → 예측 입력 형식으로 변환
    for stn in stn_choices:
        name, lat, lon = STATIONS[stn]
        df_one = df_all[df_all["STN"] == stn]

        if df_one.empty:
            st.warning(f"{name} 데이터 없음")
            continue

        feat = sfcdd_to_features(df_one)
        rows_model.append({"stn": stn, "name": name, **feat})
        rows_map.append({"stn": stn, "name": name, "lat": lat, "lon": lon, **feat})

    pred_df = pd.DataFrame(rows_model)

    # 예측 수행
    probs = model.predict_proba(pred_df[X.columns])
    pred_label = [classes[i] for i in probs.argmax(axis=1)]
    pred_prob = probs.max(axis=1)

    risk_desc = [risk_description(pred_label[i], probs[i], classes) for i in range(len(pred_label))]

    result = pred_df[["stn", "name"]].copy()
    for i, cls in enumerate(classes): result[f"P({cls})"] = probs[:, i]
    result["예측등급"], result["최대확률"], result["위험설명"] = pred_label, pred_prob, risk_desc

    st.markdown("### 예측 결과")
    st.dataframe(result)

    # 지도 표시
    vis_df = pd.DataFrame(rows_map)
    m = Map(location=[vis_df["lat"].mean(), vis_df["lon"].mean()], zoom_start=7, tiles="CartoDB positron")

    cmap = LinearColormap(["#2b83ba", "#ffffbf", "#d7191c"], vmin=0, vmax=2)
    cmap.caption = "위험 등급"
    cmap.add_to(m)

    # result의 name 컬럼 제거하여 충돌 방지
    result_no_name = result.drop(columns=["name"])

    merged = vis_df.merge(result_no_name, on="stn", how="left")

    for _, r in merged.iterrows():
        CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + 3 * int(r["예측등급"]),
            color=cmap(float(r["예측등급"])),
            fill=True,
            popup=(
                f"<b>{r['name']}</b><br>"
                f"등급:{r['예측등급']}<br>"
                f"확률:{r['최대확률']:.2f}<br>"
                f"{r['위험설명']}"
            ),
        ).add_to(m)

    LayerControl().add_to(m)
    components.html(m.get_root().render(), height=650)


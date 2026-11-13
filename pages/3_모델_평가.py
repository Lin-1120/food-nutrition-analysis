
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="모델 평가 | 식품 영양 데이터 분석실",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 모델 채점표")
st.caption(
    "로지스틱 회귀와 랜덤 포레스트가 고나트륨 음식을 "
    "얼마나 정확하게 분류하는지 평가합니다."
)

st.divider()


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/Lin-1120/food-nutrition-analysis/main/%EC%A0%84%EA%B5%AD%ED%86%B5%ED%95%A9%EC%8B%9D%ED%92%88%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84%EC%A0%95%EB%B3%B4_%EC%9D%8C%EC%8B%9D_%ED%91%9C%EC%A4%80%EB%8D%B0%EC%9D%B4%ED%84%B0.csv"


@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_URL,
        encoding="cp949",
        low_memory=False
    )

    required_columns = [
        "식품명",
        "영양성분함량기준량",
        "에너지(kcal)",
        "단백질(g)",
        "당류(g)",
        "포화지방산(g)",
        "나트륨(mg)"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV 파일에 필요한 열이 없습니다: {missing_columns}"
        )

    # 100g 기준 데이터만 선택
    reference = (
        data["영양성분함량기준량"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data[reference == "100g"].copy()

    # 영양성분 열 숫자로 변환
    numeric_columns = [
        "에너지(kcal)",
        "단백질(g)",
        "당류(g)",
        "포화지방산(g)",
        "나트륨(mg)"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col].astype(str).str.replace(
                ",", "", regex=False
            ),
            errors="coerce"
        )

    # 나트륨 결측 행 제거
    data = data.dropna(subset=["나트륨(mg)"]).copy()

    # 나트륨 상위 25% 기준 계산
    sodium_threshold = data["나트륨(mg)"].quantile(0.75)

    # 정답 열 생성
    data["고나트륨 여부"] = (
        data["나트륨(mg)"] >= sodium_threshold
    ).astype(int)

    return data, sodium_threshold


try:
    df, sodium_threshold = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 3. 입력 변수 선택
# --------------------------------------------------

st.subheader("1. 평가에 사용할 입력 변수")

st.write(
    "2번 분류 모델 페이지에서 사용한 것과 동일한 "
    "입력 변수를 선택해 주세요."
)

feature_options = {
    "에너지(kcal)": "에너지",
    "단백질(g)": "단백질",
    "당류(g)": "당류",
    "포화지방산(g)": "포화지방산"
}

selected_features = st.multiselect(
    "모델 입력 변수 선택",
    options=list(feature_options.keys()),
    default=list(feature_options.keys()),
    format_func=lambda x: feature_options[x]
)

st.info(
    "선택한 변수: "
    + (
        ", ".join(
            feature_options[x] for x in selected_features
        )
        if selected_features
        else "없음"
    )
)

st.caption(
    "※ 나트륨(mg)은 정답을 만드는 데만 사용하며 "
    "모델 입력 변수에는 포함하지 않습니다."
)

st.divider()


# --------------------------------------------------
# 4. 입력 변수 개수 확인
# --------------------------------------------------

if len(selected_features) < 2:
    st.warning(
        "모델을 평가하려면 입력 변수를 2개 이상 선택해야 합니다."
    )
    st.stop()


# --------------------------------------------------
# 5. 데이터 분할
# --------------------------------------------------

st.subheader("2. 평가 데이터 준비")

# 입력 변수 X
X = df[selected_features].copy()

# 정답 y
y = df["고나트륨 여부"].copy()

# 학습 데이터 80%, 테스트 데이터 20%
# 2번 페이지와 동일한 random_state와 층화 추출 사용
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

except ValueError as e:
    st.error("데이터를 학습용과 테스트용으로 나누지 못했습니다.")
    st.code(str(e))
    st.stop()


# 학습 데이터 중앙값으로 결측값 처리
train_medians = X_train.median()

if train_medians.isna().any():
    empty_columns = train_medians[
        train_medians.isna()
    ].index.tolist()

    st.error(
        f"학습 데이터에 유효한 값이 없는 입력 변수: {empty_columns}"
    )
    st.stop()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# 데이터 개수 표시
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 음식 데이터", f"{len(df):,}개")

with col2:
    st.metric("학습 데이터", f"{len(X_train):,}개")

with col3:
    st.metric("테스트 데이터", f"{len(X_test):,}개")

st.write(
    f"**고나트륨 분류 기준:** "
    f"100g당 나트륨 {sodium_threshold:,.2f}mg 이상"
)

st.caption(
    "학습 데이터와 테스트 데이터는 8:2로 분할했으며, "
    "두 모델 모두 동일한 테스트 데이터를 사용합니다."
)

st.divider()


# --------------------------------------------------
# 6. 로지스틱 회귀 모델 학습 및 예측
# --------------------------------------------------

st.subheader("3. 로지스틱 회귀 모델 평가")

# 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 생성
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

# 학습
logistic_model.fit(
    X_train_scaled,
    y_train
)

# 테스트 데이터 예측
logistic_pred = logistic_model.predict(X_test_scaled)


# --------------------------------------------------
# 7. 랜덤 포레스트 모델 학습 및 예측
# --------------------------------------------------

st.subheader("4. 랜덤 포레스트 모델 평가")

# 2번 페이지와 동일한 설정
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# 학습
random_forest_model.fit(
    X_train,
    y_train
)

# 테스트 데이터 예측
rf_pred = random_forest_model.predict(X_test)


# --------------------------------------------------
# 8. 혼동 행렬 계산
# --------------------------------------------------

logistic_cm = confusion_matrix(
    y_test,
    logistic_pred,
    labels=[0, 1]
)

rf_cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=[0, 1]
)

# 혼동 행렬 값 추출
# 행: 실제 정답 / 열: 모델 예측
# [[TN, FP],
#  [FN, TP]]

logistic_tn = int(logistic_cm[0, 0])
logistic_fp = int(logistic_cm[0, 1])
logistic_fn = int(logistic_cm[1, 0])
logistic_tp = int(logistic_cm[1, 1])

rf_tn = int(rf_cm[0, 0])
rf_fp = int(rf_cm[0, 1])
rf_fn = int(rf_cm[1, 0])
rf_tp = int(rf_cm[1, 1])


# --------------------------------------------------
# 9. 성능 지표 계산
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "정확도": accuracy_score(y_true, y_pred),
        "정밀도": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "재현율": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 점수": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


logistic_metrics = calculate_metrics(
    y_test,
    logistic_pred
)

rf_metrics = calculate_metrics(
    y_test,
    rf_pred
)


# --------------------------------------------------
# 10. 혼동 행렬 시각화
# --------------------------------------------------

st.divider()

st.subheader("5. 혼동 행렬로 모델 채점하기")

st.write(
    "혼동 행렬은 실제 음식의 정답과 모델이 예측한 결과를 "
    "비교하여 보여 주는 표입니다."
)

st.markdown("""
- **TN (진짜 음성)**: 실제 일반 음식을 일반 음식이라고 정확히 예측한 경우
- **FP (거짓 양성)**: 실제 일반 음식인데 고나트륨 음식이라고 잘못 예측한 경우
- **FN (거짓 음성)**: 실제 고나트륨 음식인데 일반 음식이라고 잘못 예측한 경우
- **TP (진짜 양성)**: 실제 고나트륨 음식을 고나트륨 음식이라고 정확히 예측한 경우
""")

st.info(
    "이 프로젝트에서는 FN도 중요합니다. "
    "실제로 고나트륨인 음식을 일반 음식이라고 판단하면 "
    "고나트륨 음식을 놓치게 되기 때문입니다."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### 로지스틱 회귀")

    fig_logistic = px.imshow(
        logistic_cm,
        text_auto=True,
        x=["일반 음식 예측(0)", "고나트륨 예측(1)"],
        y=["실제 일반 음식(0)", "실제 고나트륨(1)"],
        color_continuous_scale="Blues",
        title="로지스틱 회귀 혼동 행렬"
    )

    fig_logistic.update_layout(
        xaxis_title="모델 예측",
        yaxis_title="실제 정답"
    )

    st.plotly_chart(
        fig_logistic,
        use_container_width=True
    )

    st.write(f"TN: {logistic_tn:,}개")
    st.write(f"FP: {logistic_fp:,}개")
    st.write(f"FN: {logistic_fn:,}개")
    st.write(f"TP: {logistic_tp:,}개")


with col2:

    st.markdown("#### 랜덤 포레스트")

    fig_rf = px.imshow(
        rf_cm,
        text_auto=True,
        x=["일반 음식 예측(0)", "고나트륨 예측(1)"],
        y=["실제 일반 음식(0)", "실제 고나트륨(1)"],
        color_continuous_scale="Greens",
        title="랜덤 포레스트 혼동 행렬"
    )

    fig_rf.update_layout(
        xaxis_title="모델 예측",
        yaxis_title="실제 정답"
    )

    st.plotly_chart(
        fig_rf,
        use_container_width=True
    )

    st.write(f"TN: {rf_tn:,}개")
    st.write(f"FP: {rf_fp:,}개")
    st.write(f"FN: {rf_fn:,}개")
    st.write(f"TP: {rf_tp:,}개")


# --------------------------------------------------
# 11. 성능 지표 설명
# --------------------------------------------------

st.divider()

st.subheader("6. 성능 지표 이해하기")

st.markdown("""
| 평가 지표 | 의미 | 쉽게 이해하기 |
|---|---|---|
| 정확도 | 전체 예측 중 정답을 맞힌 비율 | 전체 문제 중 몇 문제를 맞혔을까? |
| 정밀도 | 고나트륨이라고 예측한 음식 중 실제 고나트륨인 비율 | 고나트륨이라고 한 음식이 실제로도 고나트륨일까? |
| 재현율 | 실제 고나트륨 음식 중 모델이 찾아낸 비율 | 실제 고나트륨 음식 중 얼마나 놓치지 않았을까? |
| F1 점수 | 정밀도와 재현율의 조화평균 | 두 능력을 함께 고려하면 어느 정도일까? |
""")

st.caption(
    "평가 지표는 0~1 사이의 값으로 나타나며, "
    "일반적으로 값이 클수록 해당 지표의 성능이 좋습니다."
)


# --------------------------------------------------
# 12. 두 모델 성능 비교표
# --------------------------------------------------

st.divider()

st.subheader("7. 두 모델의 성능 비교")

comparison_df = pd.DataFrame({
    "평가 지표": ["정확도", "정밀도", "재현율", "F1 점수"],
    "로지스틱 회귀": [
        logistic_metrics["정확도"],
        logistic_metrics["정밀도"],
        logistic_metrics["재현율"],
        logistic_metrics["F1 점수"]
    ],
    "랜덤 포레스트": [
        rf_metrics["정확도"],
        rf_metrics["정밀도"],
        rf_metrics["재현율"],
        rf_metrics["F1 점수"]
    ]
})

st.dataframe(
    comparison_df.set_index("평가 지표").style.format("{:.3f}"),
    use_container_width=True
)


# --------------------------------------------------
# 13. 성능 지표 막대그래프
# --------------------------------------------------

comparison_long = comparison_df.melt(
    id_vars="평가 지표",
    var_name="모델",
    value_name="성능 점수"
)

fig_comparison = px.bar(
    comparison_long,
    x="평가 지표",
    y="성능 점수",
    color="모델",
    barmode="group",
    text_auto=".3f",
    title="로지스틱 회귀와 랜덤 포레스트 성능 비교",
    range_y=[0, 1]
)

fig_comparison.update_layout(
    xaxis_title="평가 지표",
    yaxis_title="성능 점수",
    legend_title="분류 모델"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


# --------------------------------------------------
# 14. 고나트륨 음식 탐지 능력 비교
# --------------------------------------------------

st.divider()

st.subheader("8. 고나트륨 음식을 얼마나 놓치지 않았을까?")

recall_comparison = pd.DataFrame({
    "모델": ["로지스틱 회귀", "랜덤 포레스트"],
    "놓친 고나트륨 음식 수(FN)": [
        logistic_fn,
        rf_fn
    ],
    "찾아낸 고나트륨 음식 수(TP)": [
        logistic_tp,
        rf_tp
    ]
})

st.dataframe(
    recall_comparison,
    use_container_width=True,
    hide_index=True
)

fig_fn = px.bar(
    recall_comparison,
    x="모델",
    y="놓친 고나트륨 음식 수(FN)",
    color="모델",
    text="놓친 고나트륨 음식 수(FN)",
    title="모델별 고나트륨 음식 예측 실패 개수"
)

fig_fn.update_layout(
    xaxis_title="분류 모델",
    yaxis_title="놓친 고나트륨 음식 수"
)

st.plotly_chart(
    fig_fn,
    use_container_width=True
)


# --------------------------------------------------
# 15. 최종 모델 평가
# --------------------------------------------------

st.divider()

st.subheader("9. 최종 평가 결과")

st.write(
    "이 프로젝트에서는 정확도만으로 모델을 판단하지 않습니다. "
    "실제 고나트륨 음식을 놓치지 않는 능력인 재현율을 중요하게 고려합니다."
)

logistic_recall = logistic_metrics["재현율"]
rf_recall = rf_metrics["재현율"]

logistic_f1 = logistic_metrics["F1 점수"]
rf_f1 = rf_metrics["F1 점수"]

logistic_accuracy = logistic_metrics["정확도"]
rf_accuracy = rf_metrics["정확도"]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "로지스틱 회귀 재현율",
        f"{logistic_recall:.3f}"
    )
    st.metric(
        "로지스틱 회귀 F1 점수",
        f"{logistic_f1:.3f}"
    )

with col2:
    st.metric(
        "랜덤 포레스트 재현율",
        f"{rf_recall:.3f}"
    )
    st.metric(
        "랜덤 포레스트 F1 점수",
        f"{rf_f1:.3f}"
    )


# 재현율을 우선 비교하고 F1 점수를 보조적으로 고려
if logistic_recall > rf_recall:
    better_model = "로지스틱 회귀"
    reason = (
        "실제 고나트륨 음식을 찾아내는 재현율이 "
        "더 높기 때문입니다."
    )

elif rf_recall > logistic_recall:
    better_model = "랜덤 포레스트"
    reason = (
        "실제 고나트륨 음식을 찾아내는 재현율이 "
        "더 높기 때문입니다."
    )

else:
    if logistic_f1 > rf_f1:
        better_model = "로지스틱 회귀"
        reason = (
            "두 모델의 재현율은 같지만 "
            "F1 점수가 더 높기 때문입니다."
        )

    elif rf_f1 > logistic_f1:
        better_model = "랜덤 포레스트"
        reason = (
            "두 모델의 재현율은 같지만 "
            "F1 점수가 더 높기 때문입니다."
        )

    else:
        if logistic_accuracy > rf_accuracy:
            better_model = "로지스틱 회귀"
            reason = (
                "재현율과 F1 점수가 같고 "
                "정확도가 더 높기 때문입니다."
            )

        elif rf_accuracy > logistic_accuracy:
            better_model = "랜덤 포레스트"
            reason = (
                "재현율과 F1 점수가 같고 "
                "정확도가 더 높기 때문입니다."
            )

        else:
            better_model = "두 모델"
            reason = (
                "재현율, F1 점수, 정확도가 모두 같습니다."
            )


st.success(
    f"현재 평가 기준에서 더 좋은 성능을 보인 모델은 "
    f"**{better_model}**입니다."
)

st.write(reason)

st.caption(
    "※ 이 페이지에서는 고나트륨 음식을 놓치지 않는 능력을 "
    "중요하게 생각하여 재현율을 우선 비교했습니다. "
    "실제 모델의 활용 목적에 따라 좋은 모델의 기준은 달라질 수 있습니다."
)


# --------------------------------------------------
# 16. 학생 분석 공간
# --------------------------------------------------

st.divider()

st.subheader("✏️ 나의 모델 평가 및 해석")

st.write(
    "두 모델의 혼동 행렬과 성능 지표를 비교하고 "
    "어떤 모델이 더 적합하다고 생각하는지 직접 작성해 보세요."
)

st.text_area(
    "1. 두 모델의 정확도와 F1 점수는 어떻게 다른가요?",
    key="evaluation_note_1",
    height=100,
    placeholder="두 모델의 성능 지표를 비교하여 작성해 보세요."
)

st.text_area(
    "2. 고나트륨 음식을 더 적게 놓친 모델은 무엇인가요?",
    key="evaluation_note_2",
    height=100,
    placeholder="혼동 행렬의 FN 값을 비교하여 작성해 보세요."
)

st.text_area(
    "3. 최종적으로 어떤 모델이 더 적합하다고 생각하나요? 이유는 무엇인가요?",
    key="evaluation_note_3",
    height=120,
    placeholder="정확도, 정밀도, 재현율, F1 점수를 종합하여 작성해 보세요."
)

st.caption(
    "※ 작성한 분석 내용은 현재 앱 화면에서 입력하는 공간이며, "
    "별도로 저장되는 기능은 포함하지 않았습니다."
)

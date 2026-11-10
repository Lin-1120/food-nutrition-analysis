
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
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델 | 식품 영양 데이터 분석실",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 고나트륨 식품 분류 모델")
st.caption("음식의 영양성분을 이용하여 고나트륨 식품인지 예측하는 모델을 만들어 봅니다.")

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

    # 영양성분 열을 숫자로 변환
    numeric_columns = [
        "에너지(kcal)",
        "단백질(g)",
        "당류(g)",
        "포화지방산(g)",
        "나트륨(mg)"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    # 나트륨 값이 없는 행은 정답을 만들 수 없으므로 제외
    data = data.dropna(subset=["나트륨(mg)"]).copy()

    # 나트륨 함량의 상위 25% 기준 계산
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

st.subheader("1. 모델에 사용할 입력 변수 선택")

st.write(
    "음식의 어떤 영양성분을 이용하면 고나트륨 식품인지 "
    "잘 예측할 수 있을까요?"
)

feature_options = {
    "에너지(kcal)": "에너지",
    "단백질(g)": "단백질",
    "당류(g)": "당류",
    "포화지방산(g)": "포화지방산"
}

selected_features = st.multiselect(
    "모델 학습에 사용할 영양성분을 선택하세요.",
    options=list(feature_options.keys()),
    default=list(feature_options.keys()),
    format_func=lambda x: feature_options[x]
)

st.info(
    f"현재 선택한 입력 변수: "
    f"{', '.join(feature_options[x] for x in selected_features) if selected_features else '없음'}"
)

st.caption(
    "※ 나트륨(mg)은 고나트륨 여부라는 정답을 만드는 데만 사용하며, "
    "모델의 입력 변수로는 사용하지 않습니다."
)

st.divider()


# --------------------------------------------------
# 4. 데이터 및 정답 확인
# --------------------------------------------------

st.subheader("2. 분류 기준과 데이터 확인")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("분석 대상 음식 수", f"{len(df):,}개")

with col2:
    st.metric("고나트륨 분류 기준", f"{sodium_threshold:,.2f} mg")

with col3:
    high_count = int(df["고나트륨 여부"].sum())
    st.metric("고나트륨 음식 수", f"{high_count:,}개")

st.write(
    f"100g당 나트륨 함량이 **{sodium_threshold:,.2f}mg 이상**이면 "
    "고나트륨 음식(1), 미만이면 일반 음식(0)으로 분류합니다."
)

label_counts = (
    df["고나트륨 여부"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

label_df = pd.DataFrame({
    "분류": ["일반 음식(0)", "고나트륨 음식(1)"],
    "음식 수": [
        int(label_counts.loc[0]),
        int(label_counts.loc[1])
    ]
})

fig_label = px.bar(
    label_df,
    x="분류",
    y="음식 수",
    color="분류",
    text="음식 수",
    title="학습 데이터의 정답 분포"
)

fig_label.update_layout(
    showlegend=False,
    xaxis_title="음식 분류",
    yaxis_title="음식 수"
)

st.plotly_chart(fig_label, use_container_width=True)

st.divider()


# --------------------------------------------------
# 5. 입력 변수가 2개 미만이면 모델 실행 중단
# --------------------------------------------------

if len(selected_features) < 2:
    st.warning(
        "모델을 학습하려면 입력 변수를 2개 이상 선택해야 합니다. "
        "위에서 사용할 영양성분을 2개 이상 선택해 주세요."
    )
    st.stop()


# --------------------------------------------------
# 6. 모델 학습을 위한 데이터 준비
# --------------------------------------------------

st.subheader("3. 학습 데이터와 테스트 데이터 나누기")

# 입력 변수 X
# 나트륨(mg)은 절대로 입력 변수에 포함하지 않음
X = df[selected_features].copy()

# 정답 y
y = df["고나트륨 여부"].copy()

# 학습 데이터 80%, 테스트 데이터 20%
# stratify=y를 사용하여 정답의 클래스 비율 유지
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

except ValueError as e:
    st.error(
        "데이터를 학습용과 테스트용으로 나누지 못했습니다. "
        "각 분류에 충분한 데이터가 있는지 확인해 주세요."
    )
    st.code(str(e))
    st.stop()


# 학습 데이터의 중앙값만 이용하여 결측값 처리
train_medians = X_train.median()

# 학습 데이터에서 전부 결측값인 열이 있는지 확인
if train_medians.isna().any():
    empty_columns = train_medians[
        train_medians.isna()
    ].index.tolist()

    st.error(
        f"다음 입력 변수는 학습 데이터에 유효한 값이 없습니다: "
        f"{empty_columns}"
    )
    st.stop()

# 학습 데이터의 중앙값을 학습 및 테스트 데이터에 동일하게 적용
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# 데이터 분할 결과 표시
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터", f"{len(X):,}개")

with col2:
    st.metric("학습 데이터", f"{len(X_train):,}개")

with col3:
    st.metric("테스트 데이터", f"{len(X_test):,}개")

st.caption(
    "학습 데이터와 테스트 데이터는 8:2로 나누었으며, "
    "층화 추출을 이용하여 고나트륨 음식과 일반 음식의 비율을 유지했습니다."
)

# 클래스 비율 확인
split_ratio_df = pd.DataFrame({
    "데이터": ["학습 데이터", "테스트 데이터"],
    "일반 음식(0)": [
        (y_train == 0).mean() * 100,
        (y_test == 0).mean() * 100
    ],
    "고나트륨 음식(1)": [
        (y_train == 1).mean() * 100,
        (y_test == 1).mean() * 100
    ]
})

st.write("데이터 분할 후 정답 비율")

st.dataframe(
    split_ratio_df.round(2),
    use_container_width=True,
    hide_index=True
)

st.divider()


# --------------------------------------------------
# 7. 로지스틱 회귀 모델 학습
# --------------------------------------------------

st.subheader("4. 로지스틱 회귀 모델")

st.write(
    "로지스틱 회귀는 입력 변수들을 바탕으로 "
    "음식이 고나트륨일 확률을 계산하고 분류하는 모델입니다."
)

# 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 로지스틱 회귀 모델
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# 예측
logistic_pred = logistic_model.predict(X_test_scaled)

# 예측 확률
logistic_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]

# 평가 지표
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(
    y_test, logistic_pred, zero_division=0
)
logistic_recall = recall_score(
    y_test, logistic_pred, zero_division=0
)
logistic_f1 = f1_score(
    y_test, logistic_pred, zero_division=0
)

# 예측 개수
logistic_normal = int((logistic_pred == 0).sum())
logistic_high = int((logistic_pred == 1).sum())

st.success("로지스틱 회귀 모델 학습 완료")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("정확도", f"{logistic_accuracy:.3f}")

with col2:
    st.metric("정밀도", f"{logistic_precision:.3f}")

with col3:
    st.metric("재현율", f"{logistic_recall:.3f}")

with col4:
    st.metric("F1 점수", f"{logistic_f1:.3f}")

col1, col2 = st.columns(2)

with col1:
    st.metric("일반 음식으로 예측", f"{logistic_normal:,}개")

with col2:
    st.metric("고나트륨 음식으로 예측", f"{logistic_high:,}개")

# 혼동 행렬
logistic_cm = confusion_matrix(
    y_test,
    logistic_pred,
    labels=[0, 1]
)

fig_logistic_cm = px.imshow(
    logistic_cm,
    text_auto=True,
    x=["일반 음식 예측", "고나트륨 음식 예측"],
    y=["실제 일반 음식", "실제 고나트륨 음식"],
    color_continuous_scale="Blues",
    title="로지스틱 회귀 혼동 행렬"
)

st.plotly_chart(
    fig_logistic_cm,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# 8. 랜덤 포레스트 모델 학습
# --------------------------------------------------

st.subheader("5. 랜덤 포레스트 모델")

st.write(
    "랜덤 포레스트는 여러 의사결정트리를 만들어 "
    "각각의 예측 결과를 종합하여 분류하는 모델입니다."
)

# 랜덤 포레스트는 표준화가 필수적이지 않으므로
# 결측값을 처리한 원래 입력 데이터를 사용
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

random_forest_model.fit(
    X_train,
    y_train
)

# 예측
rf_pred = random_forest_model.predict(X_test)

# 예측 확률
rf_proba = random_forest_model.predict_proba(X_test)[:, 1]

# 평가 지표
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(
    y_test, rf_pred, zero_division=0
)
rf_recall = recall_score(
    y_test, rf_pred, zero_division=0
)
rf_f1 = f1_score(
    y_test, rf_pred, zero_division=0
)

# 예측 개수
rf_normal = int((rf_pred == 0).sum())
rf_high = int((rf_pred == 1).sum())

st.success("랜덤 포레스트 모델 학습 완료")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("정확도", f"{rf_accuracy:.3f}")

with col2:
    st.metric("정밀도", f"{rf_precision:.3f}")

with col3:
    st.metric("재현율", f"{rf_recall:.3f}")

with col4:
    st.metric("F1 점수", f"{rf_f1:.3f}")

col1, col2 = st.columns(2)

with col1:
    st.metric("일반 음식으로 예측", f"{rf_normal:,}개")

with col2:
    st.metric("고나트륨 음식으로 예측", f"{rf_high:,}개")

# 혼동 행렬
rf_cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=[0, 1]
)

fig_rf_cm = px.imshow(
    rf_cm,
    text_auto=True,
    x=["일반 음식 예측", "고나트륨 음식 예측"],
    y=["실제 일반 음식", "실제 고나트륨 음식"],
    color_continuous_scale="Greens",
    title="랜덤 포레스트 혼동 행렬"
)

st.plotly_chart(
    fig_rf_cm,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# 9. 두 모델의 성능 비교
# --------------------------------------------------

st.subheader("6. 두 모델의 성능 비교")

comparison_df = pd.DataFrame({
    "평가 지표": ["정확도", "정밀도", "재현율", "F1 점수"],
    "로지스틱 회귀": [
        logistic_accuracy,
        logistic_precision,
        logistic_recall,
        logistic_f1
    ],
    "랜덤 포레스트": [
        rf_accuracy,
        rf_precision,
        rf_recall,
        rf_f1
    ]
})

st.dataframe(
    comparison_df.set_index("평가 지표").style.format("{:.3f}"),
    use_container_width=True
)

comparison_long = comparison_df.melt(
    id_vars="평가 지표",
    var_name="모델",
    value_name="점수"
)

fig_comparison = px.bar(
    comparison_long,
    x="평가 지표",
    y="점수",
    color="모델",
    barmode="group",
    text_auto=".3f",
    title="로지스틱 회귀와 랜덤 포레스트 성능 비교",
    range_y=[0, 1]
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)

# F1 점수 기준으로 비교
if logistic_f1 > rf_f1:
    better_model = "로지스틱 회귀"
    better_score = logistic_f1
elif rf_f1 > logistic_f1:
    better_model = "랜덤 포레스트"
    better_score = rf_f1
else:
    better_model = "두 모델"
    better_score = logistic_f1

st.info(
    f"F1 점수를 기준으로 비교하면 **{better_model}**의 "
    f"성능이 더 높습니다. (F1 점수: {better_score:.3f})"
)

st.caption(
    "※ 평가 지표가 높다고 해서 모든 상황에서 더 좋은 모델이라는 뜻은 아닙니다. "
    "특히 고나트륨 음식을 놓치지 않는 것이 중요하다면 재현율도 함께 살펴보세요."
)

st.divider()


# --------------------------------------------------
# 10. 테스트 데이터의 예측 결과 확인
# --------------------------------------------------

st.subheader("7. 테스트 데이터 예측 결과")

result_df = pd.DataFrame({
    "실제 정답": y_test.values,
    "로지스틱 회귀 예측": logistic_pred,
    "로지스틱 회귀 고나트륨 확률(%)": logistic_proba * 100,
    "랜덤 포레스트 예측": rf_pred,
    "랜덤 포레스트 고나트륨 확률(%)": rf_proba * 100
})

# 분류 결과를 한글로 표시
result_df["실제 정답"] = result_df["실제 정답"].map({
    0: "일반 음식",
    1: "고나트륨 음식"
})

result_df["로지스틱 회귀 예측"] = result_df[
    "로지스틱 회귀 예측"
].map({
    0: "일반 음식",
    1: "고나트륨 음식"
})

result_df["랜덤 포레스트 예측"] = result_df[
    "랜덤 포레스트 예측"
].map({
    0: "일반 음식",
    1: "고나트륨 음식"
})

result_df = result_df.round(2)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)

# 예측이 맞은 비율
logistic_correct = int(
    (logistic_pred == y_test.to_numpy()).sum()
)

rf_correct = int(
    (rf_pred == y_test.to_numpy()).sum()
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "로지스틱 회귀 정답 개수",
        f"{logistic_correct:,}개 / {len(y_test):,}개"
    )

with col2:
    st.metric(
        "랜덤 포레스트 정답 개수",
        f"{rf_correct:,}개 / {len(y_test):,}개"
    )


# --------------------------------------------------
# 11. 마무리 안내
# --------------------------------------------------

st.divider()

st.subheader("📌 분석할 때 생각해 볼 점")

st.markdown("""
1. 두 모델 중 정확도가 더 높은 모델은 무엇인가요?
2. 고나트륨 음식을 놓치지 않는 능력인 재현율은 어떤 모델이 더 높은가요?
3. 입력 변수의 선택을 바꾸면 모델의 성능은 어떻게 달라질까요?
4. 영양성분만으로 음식의 나트륨 함량을 정확하게 분류할 수 있을까요?
""")

st.caption(
    "이 페이지는 나트륨 함량의 상위 25%를 고나트륨 음식으로 정의한 "
    "프로젝트 내부 기준을 사용합니다. 공식적인 영양 기준과는 다릅니다."
)

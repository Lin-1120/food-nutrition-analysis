
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기 | 식품 영양 데이터 분석실",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 모델 성능 높이기")
st.caption(
    "클래스 불균형을 고려한 모델 학습과 입력 변수 선택을 통해 "
    "고나트륨 식품 분류 성능을 개선해 봅니다."
)

st.divider()


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/Lin-1120/food-nutrition-analysis/main/%EC%A0%84%EA%B5%AD%ED%86%B5%ED%95%A9%EC%8B%9D%ED%92%88%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84%EC%A0%95%EB%B3%B4_%EC%9D%8C%EC%8B%9D_%ED%91%9C%준데이터.csv"


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

    # 영양성분 열 숫자 변환
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

    # 나트륨 결측값 제거
    data = data.dropna(subset=["나트륨(mg)"]).copy()

    # 나트륨 상위 25% 기준
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
# 3. 데이터와 클래스 분포 확인
# --------------------------------------------------

st.subheader("1. 고나트륨 음식과 일반 음식의 데이터 분포")

label_counts = (
    df["고나트륨 여부"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

normal_count = int(label_counts.loc[0])
high_count = int(label_counts.loc[1])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 음식 수", f"{len(df):,}개")

with col2:
    st.metric("일반 음식 수", f"{normal_count:,}개")

with col3:
    st.metric("고나트륨 음식 수", f"{high_count:,}개")

distribution_df = pd.DataFrame({
    "음식 분류": ["일반 음식(0)", "고나트륨 음식(1)"],
    "음식 수": [normal_count, high_count]
})

fig_distribution = px.bar(
    distribution_df,
    x="음식 분류",
    y="음식 수",
    color="음식 분류",
    text="음식 수",
    title="고나트륨 음식과 일반 음식의 개수"
)

fig_distribution.update_layout(
    showlegend=False,
    xaxis_title="음식 분류",
    yaxis_title="음식 수"
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)

st.info(
    f"현재 데이터에서는 일반 음식이 {normal_count:,}개, "
    f"고나트륨 음식이 {high_count:,}개입니다. "
    "두 집단의 데이터 개수 차이가 크면 모델이 적은 집단을 "
    "상대적으로 잘 찾아내지 못할 수 있습니다."
)

st.write(
    f"**고나트륨 분류 기준:** 100g당 나트륨 "
    f"{sodium_threshold:,.2f}mg 이상"
)

st.divider()


# --------------------------------------------------
# 4. 입력 변수 선택
# --------------------------------------------------

st.subheader("2. 모델에 사용할 입력 변수 선택")

feature_options = {
    "에너지(kcal)": "에너지",
    "단백질(g)": "단백질",
    "당류(g)": "당류",
    "포화지방산(g)": "포화지방산"
}

selected_features = st.multiselect(
    "분류 모델에 사용할 영양성분을 선택하세요.",
    options=list(feature_options.keys()),
    default=list(feature_options.keys()),
    format_func=lambda x: feature_options[x]
)

st.info(
    "선택한 입력 변수: "
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

if len(selected_features) < 2:
    st.warning(
        "모델을 학습하려면 입력 변수를 2개 이상 선택해 주세요."
    )
    st.stop()

st.divider()


# --------------------------------------------------
# 5. 학습 데이터와 테스트 데이터 분할
# --------------------------------------------------

st.subheader("3. 학습 데이터와 테스트 데이터 준비")

X = df[selected_features].copy()
y = df["고나트륨 여부"].copy()

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


# 학습 데이터의 중앙값으로 결측값 처리
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

col1, col2 = st.columns(2)

with col1:
    st.metric("학습 데이터", f"{len(X_train):,}개")

with col2:
    st.metric("테스트 데이터", f"{len(X_test):,}개")

st.caption(
    "두 모델은 random_state=42와 층화 추출을 사용하여 "
    "동일한 학습 데이터와 테스트 데이터를 사용합니다."
)

st.divider()


# --------------------------------------------------
# 6. 기본 랜덤 포레스트 모델
# --------------------------------------------------

st.subheader("4. 기본 랜덤 포레스트 모델")

st.write(
    "먼저 클래스 가중치를 별도로 적용하지 않은 "
    "기본 랜덤 포레스트 모델을 학습합니다."
)

basic_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

basic_model.fit(
    X_train,
    y_train
)

basic_pred = basic_model.predict(X_test)

basic_accuracy = accuracy_score(y_test, basic_pred)
basic_precision = precision_score(
    y_test, basic_pred, zero_division=0
)
basic_recall = recall_score(
    y_test, basic_pred, zero_division=0
)
basic_f1 = f1_score(
    y_test, basic_pred, zero_division=0
)

st.success("기본 랜덤 포레스트 모델 학습 완료")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("정확도", f"{basic_accuracy:.3f}")

with col2:
    st.metric("정밀도", f"{basic_precision:.3f}")

with col3:
    st.metric("재현율", f"{basic_recall:.3f}")

with col4:
    st.metric("F1 점수", f"{basic_f1:.3f}")

st.divider()


# --------------------------------------------------
# 7. class_weight 적용 모델
# --------------------------------------------------

st.subheader("5. class_weight를 적용한 랜덤 포레스트 모델")

st.write(
    "이번에는 class_weight='balanced'를 적용합니다. "
    "데이터 개수가 적은 클래스에 더 큰 가중치를 주어 "
    "모델이 해당 집단도 학습에 반영하도록 합니다."
)

balanced_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

balanced_model.fit(
    X_train,
    y_train
)

balanced_pred = balanced_model.predict(X_test)

balanced_accuracy = accuracy_score(
    y_test, balanced_pred
)
balanced_precision = precision_score(
    y_test, balanced_pred, zero_division=0
)
balanced_recall = recall_score(
    y_test, balanced_pred, zero_division=0
)
balanced_f1 = f1_score(
    y_test, balanced_pred, zero_division=0
)

st.success("class_weight 적용 모델 학습 완료")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("정확도", f"{balanced_accuracy:.3f}")

with col2:
    st.metric("정밀도", f"{balanced_precision:.3f}")

with col3:
    st.metric("재현율", f"{balanced_recall:.3f}")

with col4:
    st.metric("F1 점수", f"{balanced_f1:.3f}")

st.divider()


# --------------------------------------------------
# 8. 두 모델 성능 비교표
# --------------------------------------------------

st.subheader("6. class_weight 적용 전과 후의 성능 비교")

comparison_df = pd.DataFrame({
    "평가 지표": ["정확도", "정밀도", "재현율", "F1 점수"],
    "기본 랜덤 포레스트": [
        basic_accuracy,
        basic_precision,
        basic_recall,
        basic_f1
    ],
    "class_weight 적용": [
        balanced_accuracy,
        balanced_precision,
        balanced_recall,
        balanced_f1
    ]
})

st.dataframe(
    comparison_df.set_index("평가 지표").style.format("{:.3f}"),
    use_container_width=True
)

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
    title="class_weight 적용 전과 후의 성능 비교",
    range_y=[0, 1]
)

fig_comparison.update_layout(
    xaxis_title="평가 지표",
    yaxis_title="성능 점수"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)

st.caption(
    "※ 두 모델의 성능 차이는 class_weight 적용 여부에 따른 "
    "변화를 비교한 것입니다. 다른 조건은 동일하게 유지했습니다."
)

st.divider()


# --------------------------------------------------
# 9. 성능 변화 분석
# --------------------------------------------------

st.subheader("7. class_weight 적용으로 달라진 점")

change_df = pd.DataFrame({
    "평가 지표": ["정확도", "정밀도", "재현율", "F1 점수"],
    "성능 변화": [
        balanced_accuracy - basic_accuracy,
        balanced_precision - basic_precision,
        balanced_recall - basic_recall,
        balanced_f1 - basic_f1
    ]
})

change_df["변화 방향"] = change_df["성능 변화"].apply(
    lambda x: "상승" if x > 0 else ("하락" if x < 0 else "변화 없음")
)

change_df["성능 변화"] = change_df["성능 변화"].apply(
    lambda x: f"{x:+.3f}"
)

st.dataframe(
    change_df,
    use_container_width=True,
    hide_index=True
)

recall_change = balanced_recall - basic_recall

if recall_change > 0:
    st.success(
        f"class_weight 적용 후 재현율이 {recall_change:.3f} 상승했습니다. "
        "실제 고나트륨 음식을 더 많이 찾아냈다는 의미입니다."
    )
elif recall_change < 0:
    st.warning(
        f"class_weight 적용 후 재현율이 {abs(recall_change):.3f} 하락했습니다. "
        "이번 데이터에서는 고나트륨 음식 탐지 능력이 낮아졌습니다."
    )
else:
    st.info(
        "class_weight 적용 전후의 재현율이 동일합니다."
    )

st.divider()


# --------------------------------------------------
# 10. 변수 중요도
# --------------------------------------------------

st.subheader("8. 어떤 영양성분이 예측에 중요할까?")

st.write(
    "랜덤 포레스트 모델이 고나트륨 음식을 분류할 때 "
    "각 입력 변수를 얼마나 중요하게 활용했는지 확인합니다."
)

importance_df = pd.DataFrame({
    "영양성분": [
        feature_options[x] for x in selected_features
    ],
    "변수 중요도": balanced_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "변수 중요도",
    ascending=False
)

fig_importance = px.bar(
    importance_df,
    x="변수 중요도",
    y="영양성분",
    orientation="h",
    text_auto=".3f",
    title="랜덤 포레스트 변수 중요도",
    color="변수 중요도"
)

fig_importance.update_layout(
    xaxis_title="변수 중요도",
    yaxis_title="영양성분",
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

most_important = importance_df.iloc[0]

st.info(
    f"현재 선택한 변수 중 모델의 예측에서 가장 높은 중요도를 보인 "
    f"변수는 **{most_important['영양성분']}**입니다. "
    f"(중요도: {most_important['변수 중요도']:.3f})"
)

st.warning(
    "변수 중요도는 해당 영양성분이 고나트륨 음식의 원인이라는 뜻이 아닙니다. "
    "랜덤 포레스트 모델이 예측 과정에서 해당 변수를 상대적으로 "
    "얼마나 활용했는지를 나타내는 값입니다."
)

st.divider()


# --------------------------------------------------
# 11. 결과 해석
# --------------------------------------------------

st.subheader("9. 최종 분석 결과")

st.markdown("""
다음 내용을 중심으로 결과를 해석해 보세요.

1. 일반 음식과 고나트륨 음식의 데이터 개수에는 어떤 차이가 있나요?
2. class_weight를 적용한 후 재현율은 어떻게 달라졌나요?
3. 정확도와 재현율이 서로 다른 방향으로 변했다면 그 이유는 무엇일까요?
4. 선택한 입력 변수 중 어떤 영양성분의 중요도가 가장 높았나요?
5. 입력 변수를 바꾸면 모델의 성능과 변수 중요도는 어떻게 달라지나요?
""")

st.text_area(
    "나의 분석 및 해석",
    height=160,
    placeholder=(
        "예: class_weight를 적용한 후 재현율이 어떻게 달라졌는지, "
        "어떤 영양성분이 높은 중요도를 보였는지 작성해 보세요."
    )
)

st.caption(
    "※ 입력 변수의 선택에 따라 모델 성능과 변수 중요도는 달라질 수 있습니다. "
    "공정한 비교를 위해서는 동일한 테스트 데이터와 평가 조건을 유지해야 합니다."
)

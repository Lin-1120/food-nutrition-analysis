
import streamlit as st
import pandas as pd
import plotly.express as px


# ==================================================
# 1. 기본 설정
# ==================================================

DATA_URL = (
    "https://raw.githubusercontent.com/Lin-1120/"
    "food-nutrition-analysis/main/"
    "%EC%A0%84%EA%B5%AD%ED%86%B5%ED%95%A9%EC%8B%9D%ED%92%88"
    "%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84%EC%A0%95%EB%B3%B4_"
    "%EC%9D%8C%EC%8B%9D_%ED%91%9C%EC%A4%80%EB%8D%B0%EC%9D%B4"
    "%ED%84%B0.csv"
)

st.set_page_config(
    page_title="데이터 탐색실 | 식품 영양 데이터 분석실",
    page_icon="🔎",
    layout="wide"
)


# ==================================================
# 2. 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_URL,
        encoding="cp949",
        low_memory=False
    )

    data.columns = data.columns.astype(str).str.strip()

    return data


try:
    original_df = load_data()

except Exception as e:

    st.title("🔎 데이터 탐색실")

    st.error("CSV 데이터를 불러오지 못했습니다.")

    st.write(
        "GitHub에 CSV 파일이 제대로 업로드되었는지, "
        "공개 저장소인지, 데이터 주소가 올바른지 확인해 주세요."
    )

    with st.expander("오류 내용 자세히 보기"):
        st.code(str(e))

    st.stop()


# ==================================================
# 3. 필요한 열 확인
# ==================================================

required_columns = [
    "식품명",
    "식품대분류명",
    "영양성분함량기준량",
    "에너지(kcal)",
    "단백질(g)",
    "당류(g)",
    "포화지방산(g)",
    "나트륨(mg)"
]

missing_columns = [
    col for col in required_columns
    if col not in original_df.columns
]

if missing_columns:

    st.title("🔎 데이터 탐색실")

    st.error("데이터에 필요한 열이 없습니다.")

    st.write("찾지 못한 열:")

    for col in missing_columns:
        st.write(f"- {col}")

    with st.expander("실제 CSV 열 이름 확인하기"):
        st.write(original_df.columns.tolist())

    st.stop()


# ==================================================
# 4. 100g 기준 데이터 선택
# ==================================================

reference = (
    original_df["영양성분함량기준량"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = original_df[
    reference == "100g"
].copy()

if df.empty:

    st.title("🔎 데이터 탐색실")

    st.error("100g 기준 데이터가 없습니다.")

    with st.expander("기준량 열의 실제 값 확인하기"):
        st.write(
            original_df["영양성분함량기준량"]
            .value_counts(dropna=False)
        )

    st.stop()


# ==================================================
# 5. 영양성분 숫자형 변환
# ==================================================

numeric_columns = [
    "에너지(kcal)",
    "단백질(g)",
    "당류(g)",
    "포화지방산(g)",
    "나트륨(mg)"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )


# ==================================================
# 6. 고나트륨 음식 분류
# ==================================================

sodium_data = df["나트륨(mg)"].dropna()

if sodium_data.empty:

    st.title("🔎 데이터 탐색실")

    st.error("나트륨 함량 데이터가 없어 분석할 수 없습니다.")

    st.stop()


# 첫 화면과 동일한 상위 25% 기준
sodium_threshold = sodium_data.quantile(0.75)

df["나트륨 분류"] = pd.NA

df.loc[
    df["나트륨(mg)"].notna(),
    "나트륨 분류"
] = (
    df.loc[
        df["나트륨(mg)"].notna(),
        "나트륨(mg)"
    ] >= sodium_threshold
).map({
    True: "고나트륨 음식",
    False: "일반 음식"
})


# ==================================================
# 7. 페이지 제목
# ==================================================

st.title("🔎 데이터 탐색실")

st.markdown(
    """
    ### 음식의 영양성분 데이터를 자세히 살펴보자!

    음식의 종류와 나트륨 함량을 비교하고,
    영양성분 사이의 관계를 그래프로 탐색하는 페이지입니다.
    """
)

st.info(
    f"분석 대상: 100g 기준 음식 | "
    f"고나트륨 기준: 나트륨 {sodium_threshold:,.2f}mg 이상"
)

st.divider()


# ==================================================
# 8. 식품대분류별 음식 개수
# ==================================================

st.header("1. 식품대분류별 음식 개수")

st.write(
    "어떤 종류의 음식이 데이터에 많이 포함되어 있는지 확인해 봅니다."
)

category_count = (
    df["식품대분류명"]
    .fillna("분류 없음")
    .value_counts()
    .reset_index()
)

category_count.columns = [
    "식품대분류명",
    "음식 개수"
]

fig1 = px.bar(
    category_count,
    x="식품대분류명",
    y="음식 개수",
    text="음식 개수",
    title="식품대분류별 음식 데이터 개수",
    labels={
        "식품대분류명": "식품대분류",
        "음식 개수": "음식 개수"
    }
)

fig1.update_traces(
    textposition="outside"
)

fig1.update_layout(
    xaxis_tickangle=-30,
    height=500
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

st.subheader("📝 이 그래프로 알 수 있는 것")

st.text_area(
    "식품대분류별 음식 개수를 비교하고 알게 된 사실을 적어 보세요.",
    placeholder="예: 데이터에 가장 많이 포함된 음식 종류는 무엇인가?",
    key="exploration_answer_1",
    height=100
)

st.divider()


# ==================================================
# 9. 식품대분류별 평균 나트륨 함량
# ==================================================

st.header("2. 식품대분류별 평균 나트륨 함량")

st.write(
    "음식 종류에 따라 평균 나트륨 함량이 어떻게 다른지 비교합니다."
)

category_sodium = (
    df.dropna(subset=["나트륨(mg)"])
    .groupby("식품대분류명", dropna=False)["나트륨(mg)"]
    .mean()
    .reset_index()
)

category_sodium["식품대분류명"] = (
    category_sodium["식품대분류명"]
    .fillna("분류 없음")
)

category_sodium = category_sodium.sort_values(
    "나트륨(mg)",
    ascending=False
)

fig2 = px.bar(
    category_sodium,
    x="식품대분류명",
    y="나트륨(mg)",
    text="나트륨(mg)",
    title="식품대분류별 평균 나트륨 함량",
    labels={
        "식품대분류명": "식품대분류",
        "나트륨(mg)": "평균 나트륨 함량 (mg/100g)"
    }
)

fig2.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig2.update_layout(
    xaxis_tickangle=-30,
    height=500
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.subheader("📝 이 그래프로 알 수 있는 것")

st.text_area(
    "평균 나트륨 함량을 비교하고 알게 된 사실을 적어 보세요.",
    placeholder="예: 평균 나트륨 함량이 가장 높은 식품대분류는 무엇인가?",
    key="exploration_answer_2",
    height=100
)

st.divider()


# ==================================================
# 10. 에너지와 나트륨의 관계
# ==================================================

st.header("3. 에너지와 나트륨의 관계")

st.write(
    """
    음식의 에너지(kcal)와 나트륨(mg) 사이에
    어떤 관계가 있는지 산점도로 확인합니다.

    고나트륨 음식과 일반 음식을 서로 다른 색으로 표시합니다.
    """
)

scatter_df = df.dropna(
    subset=[
        "에너지(kcal)",
        "나트륨(mg)",
        "나트륨 분류"
    ]
).copy()

fig3 = px.scatter(
    scatter_df,
    x="에너지(kcal)",
    y="나트륨(mg)",
    color="나트륨 분류",
    hover_name="식품명",
    hover_data=["식품대분류명"],
    title="에너지와 나트륨 함량의 관계",
    labels={
        "에너지(kcal)": "에너지 (kcal/100g)",
        "나트륨(mg)": "나트륨 (mg/100g)",
        "나트륨 분류": "음식 분류"
    },
    opacity=0.65
)

fig3.update_layout(
    height=600
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.subheader("📝 이 그래프로 알 수 있는 것")

st.text_area(
    "에너지와 나트륨의 관계를 살펴보고 알게 된 사실을 적어 보세요.",
    placeholder="예: 에너지가 높은 음식은 나트륨도 높은 경향이 있는가?",
    key="exploration_answer_3",
    height=100
)

st.divider()


# ==================================================
# 11. 영양성분 상관관계 히트맵
# ==================================================

st.header("4. 영양성분 사이의 상관관계")

st.write(
    """
    단백질, 당류, 포화지방산, 나트륨 사이의
    상관관계를 확인합니다.

    상관계수는 -1에서 1 사이의 값으로 표시됩니다.
    1에 가까울수록 양의 상관관계,
    -1에 가까울수록 음의 상관관계,
    0에 가까울수록 선형 관계가 약하다는 뜻입니다.
    """
)

correlation_columns = [
    "단백질(g)",
    "당류(g)",
    "포화지방산(g)",
    "나트륨(mg)"
]

correlation_df = df[
    correlation_columns
].copy()

correlation_matrix = correlation_df.corr(
    method="pearson"
)

fig4 = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="영양성분 사이의 상관관계 히트맵",
    labels={
        "x": "영양성분",
        "y": "영양성분",
        "color": "상관계수"
    }
)

fig4.update_layout(
    height=550
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

st.subheader("📝 이 그래프로 알 수 있는 것")

st.text_area(
    "상관관계 히트맵을 보고 알게 된 사실을 적어 보세요.",
    placeholder="예: 나트륨과 가장 높은 양의 상관관계를 보이는 영양성분은 무엇인가?",
    key="exploration_answer_4",
    height=100
)

st.divider()


# ==================================================
# 12. 나트륨 함량 상위 10개 음식
# ==================================================

st.header("5. 나트륨 함량이 가장 높은 음식 TOP 10")

st.write(
    "100g 기준 나트륨 함량이 높은 음식 10개를 확인합니다."
)

top10 = (
    df.dropna(subset=["나트륨(mg)"])
    .sort_values(
        "나트륨(mg)",
        ascending=False
    )
    .head(10)
    [
        [
            "식품명",
            "식품대분류명",
            "나트륨(mg)"
        ]
    ]
    .reset_index(drop=True)
)

top10.index = top10.index + 1
top10.index.name = "순위"

st.dataframe(
    top10,
    use_container_width=True
)

fig5 = px.bar(
    top10.reset_index(),
    x="나트륨(mg)",
    y="식품명",
    orientation="h",
    text="나트륨(mg)",
    title="나트륨 함량 상위 10개 음식",
    labels={
        "식품명": "음식 이름",
        "나트륨(mg)": "나트륨 (mg/100g)"
    }
)

fig5.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig5.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=550
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

st.subheader("📝 이 그래프로 알 수 있는 것")

st.text_area(
    "나트륨 함량 상위 10개 음식을 살펴보고 알게 된 사실을 적어 보세요.",
    placeholder="예: 상위 10개 음식에는 어떤 공통점이 있는가?",
    key="exploration_answer_5",
    height=100
)

st.divider()


# ==================================================
# 13. 분석 결과 정리
# ==================================================

st.header("📖 데이터 탐색 결과 정리")

st.write(
    """
    지금까지 살펴본 그래프를 바탕으로
    음식의 영양성분과 나트륨 함량에 대해 알게 된 점을 정리해 보세요.
    """
)

st.text_area(
    "전체 분석 결과를 정리해 보세요.",
    placeholder="데이터 탐색을 통해 알게 된 사실과 새롭게 생긴 질문을 적어 보세요.",
    key="exploration_final_answer",
    height=180
)

st.caption(
    "※ 입력한 분석 내용은 현재 페이지에서 작성할 수 있으며, "
    "별도로 저장되지는 않습니다."
)

st.success("데이터 탐색이 끝났습니다! 다음 페이지에서 분류 모델을 만들어 보세요.")

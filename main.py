
import pandas as pd
import streamlit as st


# ==================================================
# 1. 기본 설정
# ==================================================

# GitHub에 업로드한 CSV 파일의 공개 주소
DATA_URL = (
    "https://raw.githubusercontent.com/Lin-1120/"
    "food-nutrition-analysis/main/"
    "%EC%A0%84%EA%B5%AD%ED%86%B5%ED%95%A9%EC%8B%9D%ED%92%88"
    "%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84%EC%A0%95%EB%B3%B4_"
    "%EC%9D%8C%EC%8B%9D_%ED%91%9C%EC%A4%80%EB%8D%B0%EC%9D%B4"
    "%ED%84%B0.csv"
)

st.set_page_config(
    page_title="식품 영양 데이터 분석실",
    page_icon="🥗",
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

    # 열 이름 앞뒤 공백 제거
    data.columns = data.columns.astype(str).str.strip()

    return data


# ==================================================
# 3. 앱 제목
# ==================================================

st.title("🥗 식품 영양 데이터 분석실")

st.markdown(
    """
    ### 음식의 영양성분으로 나트륨 함량을 분석해 보자!

    전국통합식품영양성분정보 음식 표준데이터를 활용하여
    음식의 영양성분을 탐색하고 나트륨 함량이 높은 음식을
    분류하는 프로젝트입니다.
    """
)

st.divider()


# ==================================================
# 4. 데이터 불러오기 및 오류 처리
# ==================================================

try:
    original_df = load_data()

except Exception as e:
    st.error("CSV 데이터를 불러오지 못했습니다.")

    st.write("다음 사항을 확인해 주세요.")

    st.write("1. GitHub 저장소에 CSV 파일이 업로드되어 있는지 확인해 주세요.")
    st.write("2. CSV 파일이 저장소의 최상위 폴더에 있는지 확인해 주세요.")
    st.write("3. 파일 이름이 코드에 입력된 이름과 같은지 확인해 주세요.")
    st.write("4. GitHub 저장소가 Public 상태인지 확인해 주세요.")
    st.write("5. CSV 파일의 인코딩이 CP949인지 확인해 주세요.")

    with st.expander("오류 내용 자세히 보기"):
        st.code(str(e))

    st.stop()


# ==================================================
# 5. 필수 열 확인
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
    st.error("CSV 파일에서 필요한 열을 찾지 못했습니다.")

    st.write("다음 열이 파일에 없습니다.")

    for col in missing_columns:
        st.write(f"- {col}")

    with st.expander("CSV 파일의 실제 열 이름 확인하기"):
        st.write(original_df.columns.tolist())

    st.stop()


# ==================================================
# 6. 100g 기준 데이터만 선택
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
    st.error("100g 기준에 해당하는 데이터를 찾지 못했습니다.")

    with st.expander("기준량 열의 실제 값 확인하기"):
        st.write(
            original_df["영양성분함량기준량"]
            .value_counts(dropna=False)
        )

    st.stop()


# ==================================================
# 7. 영양성분 숫자형 변환
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
# 8. 나트륨 데이터 확인
# ==================================================

sodium_data = df["나트륨(mg)"].dropna()

if sodium_data.empty:
    st.error(
        "100g 기준 데이터에서 나트륨 함량을 확인할 수 없습니다."
    )
    st.stop()


# ==================================================
# 9. 고나트륨 음식 분류
# ==================================================

# 나트륨 함량의 75백분위수 계산
sodium_threshold = sodium_data.quantile(0.75)

# 나트륨 값이 기준 이상이면 고나트륨 음식
high_sodium_df = df[
    df["나트륨(mg)"] >= sodium_threshold
].copy()

total_food_count = len(df)
column_count = len(df.columns)
high_sodium_count = len(high_sodium_df)
valid_sodium_count = len(sodium_data)

high_sodium_rate = (
    high_sodium_count / valid_sodium_count * 100
)


# ==================================================
# 10. 프로젝트 소개
# ==================================================

st.header("📌 프로젝트 소개")

st.write(
    """
    우리가 먹는 음식에는 에너지, 단백질, 당류, 지방,
    나트륨 등 다양한 영양성분이 들어 있습니다.

    이 프로젝트에서는 전국통합식품영양성분정보 데이터를
    활용하여 음식의 영양성분과 나트륨 함량 사이의 관계를
    살펴보고자 합니다.

    특히 100g 기준 음식만을 분석 대상으로 선택하여
    음식의 영양성분 조합에 따라 나트륨 함량이 높은 음식을
    분류할 수 있는지 알아봅니다.
    """
)

st.info(
    f"현재 데이터의 고나트륨 분류 기준은 "
    f"100g당 나트륨 {sodium_threshold:,.2f}mg 이상입니다."
)

st.caption(
    "※ 고나트륨 기준은 전체 나트륨 함량의 상위 25%를 이용한 "
    "프로젝트 내부의 상대적 분류 기준입니다."
)

st.divider()


# ==================================================
# 11. 데이터 요약 카드
# ==================================================

st.header("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 음식 데이터 수",
    f"{total_food_count:,}개"
)

col2.metric(
    "열 개수",
    f"{column_count:,}개"
)

col3.metric(
    "고나트륨 음식 수",
    f"{high_sodium_count:,}개"
)

col4.metric(
    "고나트륨 음식 비율",
    f"{high_sodium_rate:.2f}%"
)

st.caption(
    "전체 음식 데이터 수는 100g 기준 음식의 행 개수입니다. "
    "고나트륨 음식 비율은 나트륨 값이 있는 음식만을 기준으로 계산합니다."
)

st.divider()


# ==================================================
# 12. 데이터 열 정보 표
# ==================================================

st.header("📋 데이터 열 정보")

st.write(
    """
    아래 표에서 데이터의 열 이름, 값의 종류, 빈 값 개수를
    확인할 수 있습니다.

    '우리말 뜻' 칸은 교재를 참고하여 직접 채워 보세요.
    """
)

column_info = []

for col in df.columns:

    if pd.api.types.is_numeric_dtype(df[col]):
        value_type = "숫자형"
    else:
        value_type = "문자형"

    unique_count = df[col].nunique(dropna=True)
    missing_count = int(df[col].isna().sum())

    column_info.append({
        "열 이름": col,
        "우리말 뜻": "",
        "값의 종류": f"{value_type} (고유값 {unique_count:,}개)",
        "빈 값 개수": missing_count
    })

column_info_df = pd.DataFrame(column_info)

st.data_editor(
    column_info_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            disabled=True
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 뜻을 입력해 보세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            disabled=True
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            disabled=True,
            format="%d"
        )
    },
    key="column_info_editor"
)

st.caption(
    "※ 우리말 뜻 칸에 입력한 내용은 현재 화면에서 수정할 수 있지만 "
    "별도로 저장되지는 않습니다."
)

st.divider()


# ==================================================
# 13. 데이터 처음 다섯 줄
# ==================================================

st.header("🔎 원본 데이터 미리보기")

st.write(
    "100g 기준으로 선택한 데이터의 처음 다섯 줄입니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

st.divider()


# ==================================================
# 14. 데이터 출처
# ==================================================

st.header("📚 데이터 출처")

st.write(
    """
    **데이터 이름:** 전국통합식품영양성분정보 음식 표준데이터

    **분석 대상:** 영양성분함량기준량이 100g인 음식

    **데이터 인코딩:** CP949
    """
)

st.text_input(
    "데이터 제공 기관 및 출처 주소",
    placeholder="데이터 제공 기관과 원본 데이터 주소를 입력하세요."
)

st.caption(
    "정확한 데이터 제공 기관과 원본 데이터 주소를 확인하여 기록해 주세요."
)

st.divider()


# ==================================================
# 15. 분석 기준 자세히 보기
# ==================================================

with st.expander("고나트륨 음식 분류 기준 자세히 보기"):

    st.write(
        f"""
        **분석 대상 음식 수:** {total_food_count:,}개

        **나트륨 값이 있는 음식 수:** {valid_sodium_count:,}개

        **고나트륨 분류 기준:** 100g당 나트륨 {sodium_threshold:,.2f}mg 이상

        **고나트륨 음식 수:** {high_sodium_count:,}개

        **고나트륨 음식 비율:** {high_sodium_rate:.2f}%
        """
    )

    st.warning(
        "이 기준은 데이터 분포의 상위 25%를 이용한 분석용 기준입니다. "
        "공식적인 영양학적 고나트륨 판정 기준은 아닙니다."
    )

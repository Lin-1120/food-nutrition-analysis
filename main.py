import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="식품 영양 데이터 분석실",
    page_icon="🥗",
    layout="wide",
)

# GitHub에 CSV를 올린 뒤 Raw URL을 여기에 입력하세요.
DATA_URL = "여기에_CSV_RAW_URL_입력"

# 분석에 반드시 필요한 열 이름
REQUIRED_COLUMNS = [
    "식품명",
    "식품대분류명",
    "영양성분함량기준량",
    "에너지(kcal)",
    "단백질(g)",
    "당류(g)",
    "포화지방산(g)",
    "나트륨(mg)",
]


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data(url):
    if not url or url == "여기에_CSV_RAW_URL_입력":
        raise ValueError(
            "CSV 공개 URL이 입력되지 않았습니다. "
            "코드 맨 위의 DATA_URL에 GitHub Raw URL을 입력해 주세요."
        )

    data = pd.read_csv(url, encoding="cp949")

    # 열 이름 앞뒤의 불필요한 공백 제거
    data.columns = data.columns.astype(str).str.strip()

    return data


# --------------------------------------------------
# 100g 기준 데이터만 선택
# --------------------------------------------------
def select_100g_data(data):
    reference = data["영양성분함량기준량"].astype(str).str.strip()

    # '100g', '100 g', '100g당' 등의 표현을 확인
    mask = reference.str.contains(
        r"100\s*g",
        case=False,
        regex=True,
        na=False,
    )

    # 기준량이 숫자 100으로 저장된 경우도 처리
    numeric_reference = pd.to_numeric(reference, errors="coerce")
    mask = mask | (numeric_reference == 100)

    return data.loc[mask].copy()


# --------------------------------------------------
# 숫자형 영양성분 변환
# --------------------------------------------------
def convert_numeric_columns(data):
    data = data.copy()

    numeric_columns = [
        "에너지(kcal)",
        "단백질(g)",
        "당류(g)",
        "포화지방산(g)",
        "나트륨(mg)",
    ]

    for column in numeric_columns:
        # 쉼표가 들어간 숫자도 처리
        cleaned = (
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        data[column] = pd.to_numeric(cleaned, errors="coerce")

    return data


# --------------------------------------------------
# 앱 제목
# --------------------------------------------------
st.title("🥗 식품 영양 데이터 분석실")
st.caption(
    "음식의 영양성분 데이터를 살펴보고 "
    "영양성분과 나트륨 함량의 관계를 알아보는 공간입니다."
)

st.divider()


# --------------------------------------------------
# 데이터 불러오기 및 오류 처리
# --------------------------------------------------
try:
    original_df = load_data(DATA_URL)

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")

    st.write(
        "CSV 주소가 올바른지, GitHub 저장소에 파일이 올라가 있는지 "
        "확인해 주세요. CSV 인코딩은 CP949로 설정되어 있어야 합니다."
    )

    with st.expander("오류 내용 자세히 보기"):
        st.code(str(error))

    st.stop()


# --------------------------------------------------
# 필수 열 확인
# --------------------------------------------------
missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in original_df.columns
]

if missing_columns:
    st.error("CSV 파일에서 필요한 열을 찾지 못했습니다.")

    st.write("다음 열이 파일에 없습니다.")

    for column in missing_columns:
        st.write(f"• {column}")

    st.write("CSV 파일의 실제 열 이름과 코드에 적힌 열 이름을 확인해 주세요.")

    with st.expander("CSV에 실제로 들어 있는 열 이름 확인하기"):
        st.write(list(original_df.columns))

    st.stop()


# --------------------------------------------------
# 100g 기준 데이터 선택
# --------------------------------------------------
df = select_100g_data(original_df)

if df.empty:
    st.error("100g 기준에 해당하는 데이터를 찾지 못했습니다.")

    st.write(
        "영양성분함량기준량 열에 100g 또는 100 g과 같은 값이 "
        "들어 있는지 확인해 주세요."
    )

    with st.expander("영양성분함량기준량의 실제 값 확인하기"):
        st.write(
            original_df["영양성분함량기준량"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    st.stop()


# --------------------------------------------------
# 영양성분 숫자형 변환
# --------------------------------------------------
df = convert_numeric_columns(df)

# 나트륨 함량이 없는 행은 고나트륨 기준 계산에서 제외
valid_sodium = df["나트륨(mg)"].dropna()

if valid_sodium.empty:
    st.error(
        "100g 기준 데이터에서 나트륨 함량을 확인할 수 없습니다. "
        "나트륨(mg) 열의 값을 확인해 주세요."
    )
    st.stop()


# --------------------------------------------------
# 고나트륨 기준 계산
# --------------------------------------------------
sodium_threshold = valid_sodium.quantile(0.75)

high_sodium_df = df[
    df["나트륨(mg)"] >= sodium_threshold
].copy()

total_foods = len(df)
column_count = len(df.columns)
high_sodium_count = len(high_sodium_df)

# 나트륨 값이 있는 음식 중 고나트륨 음식의 비율
valid_food_count = len(valid_sodium)

high_sodium_ratio = (
    high_sodium_count / valid_food_count * 100
    if valid_food_count > 0
    else 0
)


# --------------------------------------------------
# 프로젝트 소개
# --------------------------------------------------
st.header("프로젝트 소개")

st.write(
    """
    이 프로젝트는 전국통합식품영양성분정보 음식 표준데이터를 활용하여
    음식에 포함된 영양성분과 나트륨 함량의 관계를 살펴보는 프로젝트입니다.

    모든 분석은 영양성분함량기준량이 100g인 음식만을 대상으로 합니다.

    나트륨 함량의 상위 25%에 해당하는 음식을
    '고나트륨 음식'으로 분류합니다.
    """
)

st.info(
    f"현재 데이터의 고나트륨 분류 기준은 "
    f"100g당 나트륨 {sodium_threshold:,.2f}mg 이상입니다. "
    "이 기준은 현재 데이터의 75백분위수를 이용한 상대적 분류 기준입니다."
)


# --------------------------------------------------
# 핵심 요약 카드
# --------------------------------------------------
st.header("데이터 한눈에 보기")

card1, card2, card3, card4 = st.columns(4)

card1.metric(
    "전체 음식 데이터 수",
    f"{total_foods:,}개",
)

card2.metric(
    "열 개수",
    f"{column_count:,}개",
)

card3.metric(
    "고나트륨 음식 수",
    f"{high_sodium_count:,}개",
)

card4.metric(
    "고나트륨 음식 비율",
    f"{high_sodium_ratio:.2f}%",
)

st.caption(
    "전체 음식 데이터 수는 100g 기준 음식 행의 수입니다. "
    "고나트륨 음식 비율은 나트륨 값이 확인되는 음식만을 기준으로 계산합니다."
)

st.divider()


# --------------------------------------------------
# 열 이름과 데이터 정보
# --------------------------------------------------
st.header("열 이름과 데이터 살펴보기")

st.write(
    "각 열의 이름과 값의 종류, 빈 값 개수를 확인할 수 있습니다. "
    "‘우리말 뜻’ 칸은 교재를 참고하여 직접 작성해 보세요."
)

summary_rows = []

for column in df.columns:
    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        data_type = "숫자형"
    else:
        data_type = "문자형"

    unique_count = series.nunique(dropna=True)
    missing_count = int(series.isna().sum())

    summary_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": "",
            "값의 종류": f"{data_type} · 고유값 {unique_count:,}종",
            "빈 값 개수": missing_count,
        }
    )

summary_df = pd.DataFrame(summary_rows)

edited_summary = st.data_editor(
    summary_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수",
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            disabled=True,
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 각 열의 의미를 직접 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            disabled=True,
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            disabled=True,
            format="%d",
        ),
    },
    key="meaning_editor",
)

st.caption(
    "값의 종류는 숫자형 또는 문자형 구분과 "
    "결측치를 제외한 고유값 개수를 나타냅니다."
)

st.divider()


# --------------------------------------------------
# 원본 데이터 미리보기
# --------------------------------------------------
st.header("데이터 처음 다섯 줄")

st.write(
    "아래 표는 100g 기준으로 선택한 데이터의 처음 다섯 행입니다. "
    "열 순서는 CSV 원본 순서를 유지합니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.divider()


# --------------------------------------------------
# 데이터 출처
# --------------------------------------------------
st.header("데이터 출처")

st.write(
    "사용한 데이터의 출처와 관련 정보를 기록하는 공간입니다."
)

source_text = st.text_area(
    "데이터 출처 기록",
    placeholder="데이터 출처와 제공 기관 등의 정보를 입력하세요.",
    height=100,
)

st.caption(
    "전국통합식품영양성분정보 음식 표준데이터를 사용합니다. "
    "정확한 제공 기관 및 원본 데이터 주소를 확인하여 기록해 주세요."
)

st.divider()


# --------------------------------------------------
# 분석 기준 안내
# --------------------------------------------------
with st.expander("데이터 분석 기준 확인하기"):
    st.write(
        f"""
        **분석 대상:** 영양성분함량기준량이 100g인 음식

        **고나트륨 분류 기준:** 나트륨 함량의 75백분위수 이상

        **현재 계산된 기준:** 100g당 나트륨 {sodium_threshold:,.2f}mg

        **나트륨 값이 있는 음식 수:** {valid_food_count:,}개

        **고나트륨 음식 수:** {high_sodium_count:,}개

        **고나트륨 음식 비율:** {high_sodium_ratio:.2f}%
        """
    )

    st.write(
        "주의: 이 프로젝트에서 사용하는 고나트륨 분류는 "
        "현재 데이터의 분포를 기준으로 만든 분석용 기준입니다. "
        "공식적인 영양학적 판정 기준과는 다릅니다."
    )

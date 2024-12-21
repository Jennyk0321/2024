import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
file_path = 'info_collection.xlsx'
test_data = pd.read_excel(file_path, sheet_name='test')
training_data = pd.read_excel(file_path, sheet_name='training')

# Streamlit 앱
def main():
    st.title("영상 분석 데이터셋 통계 및 시각화")

    st.sidebar.title("설정")
    analysis_type = st.sidebar.selectbox("분석 유형 선택:", [
        "전체 개요",
        "훈련 데이터셋",
        "테스트 데이터셋",
        "통합 데이터셋"
    ])

    # 데이터셋 개요
    if analysis_type == "전체 개요":
        st.header("데이터셋 개요")
        st.write("### 테스트 데이터셋")
        st.write(test_data.head())

        st.write("### 훈련 데이터셋")
        st.write(training_data.head())

        st.write("### 통합 데이터셋")
        combined_data = pd.concat([training_data, test_data], axis=0, ignore_index=True)
        st.write(combined_data.head())

    # 훈련 데이터셋 분석
    elif analysis_type == "훈련 데이터셋":
        st.header("훈련 데이터셋 분석")
        dataset_analysis(training_data)

    # 테스트 데이터셋 분석
    elif analysis_type == "테스트 데이터셋":
        st.header("테스트 데이터셋 분석")
        dataset_analysis(test_data)

    # 통합 데이터셋 분석
    elif analysis_type == "통합 데이터셋":
        st.header("통합 데이터셋 분석")
        combined_data = pd.concat([training_data, test_data], axis=0, ignore_index=True)
        dataset_analysis(combined_data)


def dataset_analysis(data):
    """데이터셋 분석 및 시각화"""
    st.write("### 기본 통계")
    st.write(data.describe())

    st.write("### 결측치 확인")
    st.write(data.isnull().sum())

    st.write("### 변수 분포")
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        st.write(f"#### {column} 분포")
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### 상관관계 히트맵")
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("상관관계 히트맵을 생성하기에 충분한 수치형 열이 없습니다.")

    st.write("### 변수 간 관계 (Pairplot)")
    if len(numeric_columns) > 1:
        pairplot_fig = sns.pairplot(data[numeric_columns])
        st.pyplot(pairplot_fig)
    else:
        st.write("Pairplot을 생성하기에 충분한 수치형 열이 없습니다.")

    st.write("### 범주형 변수 분석")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        st.write(f"#### {column} 값 분포")
        st.bar_chart(data[column].value_counts())


if __name__ == "__main__":
    main()


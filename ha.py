import streamlit as st
import pandas as pd
import os

# matplotlib 및 한글 지원 라이브러리 import
import matplotlib.pyplot as plt
import matplotlib
from koreanize_matplotlib import koreanize_matplotlib

matplotlib.use('Agg')  # Streamlit에서 오류 방지용
koreanize_matplotlib()  # 한글 폰트 설정

# 데이터 로드
file_path = 'info_coll.csv'
data = pd.read_csv(file_path)

# 데이터 분리
test_data = data[data['index'] == 'test']
training_data = data[data['index'] == 'training']

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
        ax.hist(data[column], bins=30, alpha=0.7)
        ax.set_title(f"{column} 분포")
        ax.set_xlabel(column)
        ax.set_ylabel("빈도")
        st.pyplot(fig)

    st.write("### 상관관계 히트맵")
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots()
        cax = ax.matshow(data[numeric_columns].corr(), cmap='coolwarm')
        fig.colorbar(cax)
        ax.set_xticks(range(len(numeric_columns)))
        ax.set_yticks(range(len(numeric_columns)))
        ax.set_xticklabels(numeric_columns, rotation=90)
        ax.set_yticklabels(numeric_columns)
        st.pyplot(fig)
    else:
        st.write("상관관계 히트맵을 생성하기에 충분한 수치형 열이 없습니다.")

    st.write("### 범주형 변수 분석")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        st.write(f"#### {column} 값 분포")
        value_counts = data[column].value_counts()
        fig, ax = plt.subplots()
        ax.bar(value_counts.index, value_counts.values, alpha=0.7)
        ax.set_title(f"{column} 값 분포")
        ax.set_xlabel(column)
        ax.set_ylabel("개수")
        st.pyplot(fig)


if __name__ == "__main__":
    main()

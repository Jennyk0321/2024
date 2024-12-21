import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'info_collection.xlsx'
test_data = pd.read_excel(file_path, sheet_name='test')
training_data = pd.read_excel(file_path, sheet_name='training')

# Streamlit app
def main():
    st.title("Video Analysis Dataset Insights")

    st.sidebar.title("Settings")
    analysis_type = st.sidebar.selectbox("Select Analysis Type:", [
        "Overview",
        "Training Dataset",
        "Test Dataset",
        "Combined Dataset"
    ])

    # Dataset Overview
    if analysis_type == "Overview":
        st.header("Dataset Overview")
        st.write("### Test Dataset")
        st.write(test_data.head())

        st.write("### Training Dataset")
        st.write(training_data.head())

        st.write("### Combined Dataset")
        combined_data = pd.concat([training_data, test_data], axis=0, ignore_index=True)
        st.write(combined_data.head())

    # Training Dataset Analysis
    elif analysis_type == "Training Dataset":
        st.header("Training Dataset Analysis")
        dataset_analysis(training_data)

    # Test Dataset Analysis
    elif analysis_type == "Test Dataset":
        st.header("Test Dataset Analysis")
        dataset_analysis(test_data)

    # Combined Dataset Analysis
    elif analysis_type == "Combined Dataset":
        st.header("Combined Dataset Analysis")
        combined_data = pd.concat([training_data, test_data], axis=0, ignore_index=True)
        dataset_analysis(combined_data)


def dataset_analysis(data):
    """Analyze and visualize the dataset"""
    st.write("### Basic Statistics")
    st.write(data.describe())

    st.write("### Null Values")
    st.write(data.isnull().sum())

    st.write("### Distribution of Variables")
    numeric_columns = data.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        st.write(f"#### {column} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### Correlation Heatmap")
    if len(numeric_columns) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation heatmap.")

    st.write("### Pairplot")
    if len(numeric_columns) > 1:
        pairplot_fig = sns.pairplot(data[numeric_columns])
        st.pyplot(pairplot_fig)
    else:
        st.write("Not enough numeric columns for pairplot.")

    st.write("### Category Counts")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        st.write(f"#### {column} Counts")
        st.bar_chart(data[column].value_counts())


if __name__ == "__main__":
    main()


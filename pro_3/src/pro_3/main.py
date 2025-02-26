import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_data():
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })
    return df

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Iris Dataset Explorer",
        page_icon="🌸",
        layout="wide"
    )

    # Add title and description
    st.title("🌸 Iris Dataset Explorer")
    st.write("""
    Welcome to the Iris Dataset Explorer! This app allows you to visualize and analyze
    the famous Iris dataset. Use the sidebar to customize your visualization.
    """)

    # Load data
    df = load_data()

    # Sidebar
    st.sidebar.header("Visualization Settings")
    
    # Select plot type
    plot_type = st.sidebar.selectbox(
        "Choose Plot Type",
        ["Scatter Plot", "Box Plot", "Violin Plot"]
    )

    # Data summary
    st.header("Dataset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("First few rows of the dataset:")
        st.dataframe(df.head())
    
    with col2:
        st.write("Dataset Statistics:")
        st.dataframe(df.describe())

    # Visualization
    st.header("Data Visualization")

    if plot_type == "Scatter Plot":
        x_axis = st.sidebar.selectbox("X-axis", df.columns[:-2])
        y_axis = st.sidebar.selectbox("Y-axis", df.columns[:-2])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='species', ax=ax)
        plt.title(f'{x_axis} vs {y_axis} by Species')
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        feature = st.sidebar.selectbox("Feature", df.columns[:-2])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='species', y=feature, ax=ax)
        plt.title(f'Distribution of {feature} by Species')
        st.pyplot(fig)

    else:  # Violin Plot
        feature = st.sidebar.selectbox("Feature", df.columns[:-2])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x='species', y=feature, ax=ax)
        plt.title(f'Distribution of {feature} by Species')
        st.pyplot(fig)

    # Show correlation matrix
    st.header("Correlation Matrix")
    corr = df.iloc[:, :-2].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()

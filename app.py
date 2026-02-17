"""
Streamlit Exploratory Data Analysis App - Iris Dataset

This app demonstrates basic EDA techniques using Streamlit.
Load the iris dataset, explore statistics, and visualize relationships.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Configure page title and layout
st.set_page_config(page_title="Iris EDA App", layout="wide")

# Add title and description
st.title("🌸 Iris Dataset - Exploratory Data Analysis")
st.markdown("Explore the iris dataset interactively using Streamlit")

# Load the iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df["species"] = iris_data.target_names[iris_data.target]

# Display dataset overview section
st.header("📊 Dataset Overview")

# Show first rows of the dataset
st.subheader("First 10 rows of the dataset")
st.dataframe(iris_df.head(10))

# Display basic information about the dataset
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", len(iris_df))
with col2:
    st.metric("Total Columns", len(iris_df.columns))
with col3:
    st.metric("Unique Species", iris_df["species"].nunique())

# Display summary statistics section
st.header("📈 Summary Statistics")

# Get numeric columns (all except species)
numeric_columns = iris_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Show summary statistics for all numeric columns
st.subheader("Statistical Summary of Numeric Features")
st.dataframe(iris_df[numeric_columns].describe())

# Allow user to select columns for visualization
st.header("📉 Data Visualization")

# Create two columns for histogram and scatter plot controls
viz_col1, viz_col2 = st.columns(2)

# Histogram section
with viz_col1:
    st.subheader("Histogram")
    selected_histogram_column = st.selectbox(
        "Select a numeric column for histogram:",
        numeric_columns,
        key="histogram_column"
    )
    
    # Create histogram
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(iris_df[selected_histogram_column], bins=20, color="skyblue", edgecolor="black")
    ax_hist.set_xlabel(selected_histogram_column)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"Distribution of {selected_histogram_column}")
    ax_hist.grid(axis="y", alpha=0.3)
    st.pyplot(fig_hist)

# Scatter plot section
with viz_col2:
    st.subheader("Scatter Plot")
    
    # Allow user to select two columns for scatter plot
    col1_scatter = st.selectbox(
        "Select X-axis column:",
        numeric_columns,
        key="scatter_x"
    )
    col2_scatter = st.selectbox(
        "Select Y-axis column:",
        numeric_columns,
        index=1 if len(numeric_columns) > 1 else 0,
        key="scatter_y"
    )
    
    # Create scatter plot with species color coding
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    
    # Plot each species with a different color
    for species in iris_df["species"].unique():
        species_data = iris_df[iris_df["species"] == species]
        ax_scatter.scatter(
            species_data[col1_scatter],
            species_data[col2_scatter],
            label=species,
            alpha=0.7,
            s=100
        )
    
    ax_scatter.set_xlabel(col1_scatter)
    ax_scatter.set_ylabel(col2_scatter)
    ax_scatter.set_title(f"{col1_scatter} vs {col2_scatter}")
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.3)
    st.pyplot(fig_scatter)

# Display species distribution
st.header("🎯 Species Distribution")
species_counts = iris_df["species"].value_counts()
st.bar_chart(species_counts)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("Basic Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Column selection
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        # Histogram
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(y=df[column], ax=ax, color="lightgreen")
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    else:
        # Bar chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax, color="coral")
        ax.set_title(f"Bar Chart of {column}")
        st.pyplot(fig)

        # Pie chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Pie Chart of {column}")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap (numeric columns)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Scatter plot with regression
    st.subheader("Scatter Plot with Regression Line")
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
        y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)

        # Regression line
        X = df[[x_col]].dropna()
        y = df[y_col].dropna()
        if len(X) > 1 and len(y) > 1:
            model = LinearRegression()
            model.fit(X, y[:len(X)])
            y_pred = model.predict(X)
            ax.plot(X, y_pred, color="red")
            st.write(f"Regression: {y_col} = {model.coef_[0]:.2f} × {x_col} + {model.intercept_:.2f}")

        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig)

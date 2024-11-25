import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# App title
st.title("Car Price Analysis")
st.markdown("Analyzing main characteristics that have the most impact on car prices.")

# Section 1: Import Data
st.header("1. Import Data")
path = 'https://raw.githubusercontent.com/ncit17153/Exam1_Utsavkhadka1/refs/heads/main/Exam1_clean_df.csv'
df = pd.read_csv(path)
st.write("### Data Preview:")
st.dataframe(df.head())

# Data types
st.subheader("Data Types")
st.write(df.dtypes)

# Analyze "peak-rpm"
st.subheader("Question 1: Data Type of 'peak-rpm'")
st.write(f"Data type of 'peak-rpm': **{df['peak-rpm'].dtypes}**")

# Correlation examples
st.header("2. Analyzing Individual Feature Patterns Using Visualizations")
st.write("### Correlation Example: Horsepower and Price")
st.write(df[['horsepower', 'price']].corr())

# Scatterplots with regression lines
st.subheader("Scatterplots")
col1, col2 = st.columns(2)

with col1:
    st.write("### Engine Size vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x="engine-size", y="price", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Highway MPG vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x="highway-mpg", y="price", data=df, ax=ax)
    st.pyplot(fig)

st.write("### Peak RPM vs Price")
fig, ax = plt.subplots()
sns.regplot(x="peak-rpm", y="price", data=df, ax=ax)
st.pyplot(fig)

# Correlation matrix for selected features
st.write("### Correlation: Bore, Stroke, Compression Ratio, Horsepower")
st.write(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Boxplots for categorical variables
st.header("3. Categorical Variables")
st.write("### Body Style vs Price")
fig, ax = plt.subplots()
sns.boxplot(x="body-style", y="price", data=df, palette="Set2", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.write("### Engine Location vs Price")
fig, ax = plt.subplots()
sns.boxplot(x="engine-location", y="price", data=df, palette="Set3", ax=ax)
st.pyplot(fig)

st.write("### Drive Wheels vs Price")
fig, ax = plt.subplots()
sns.boxplot(x="drive-wheels", y="price", data=df, palette="coolwarm", ax=ax)
st.pyplot(fig)

# Descriptive statistics
st.header("4. Descriptive Statistics")
st.write("### Continuous Variables:")
st.write(df.describe())

st.write("### Categorical Variables:")
st.write(df.describe(include=['object']))

# Value counts
st.write("### Value Counts: Drive Wheels")
st.write(df['drive-wheels'].value_counts().to_frame())

# Grouping and Pivot
st.header("5. Grouping and Pivot")
st.write("### Grouped Data: Drive Wheels and Body Style vs Price")
df_group_one = df[['drive-wheels', 'body-style', 'price']].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
grouped_pivot = df_group_one.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
st.write(grouped_pivot)

st.write("### Heatmap: Drive Wheels and Body Style vs Price")
fig, ax = plt.subplots()
sns.heatmap(grouped_pivot, annot=True, fmt=".1f", cmap="RdBu", ax=ax)
st.pyplot(fig)

# Correlation and Causation
st.header("6. Correlation and Causation")
st.write("### Pearson Correlation and P-value")

features = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight']
for feature in features:
    pearson_coef, p_value = stats.pearsonr(df[feature], df['price'])
    st.write(f"**{feature.capitalize()} vs Price:**")
    st.write(f"- Pearson Correlation Coefficient: {pearson_coef:.3f}")
    st.write(f"- P-value: {p_value:.3e}")

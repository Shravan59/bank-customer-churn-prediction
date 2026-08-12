import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Dataset Analysis", layout="wide")

st.title("📊 Dataset Analysis")

DATA_PATH = Path("data/processed/cleaned_data.csv")

if not DATA_PATH.exists():
    st.error("cleaned_data.csv not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.success("Dataset Loaded Successfully")

st.subheader("Shape")
st.write(df.shape)

st.subheader("Columns")
st.write(df.columns.tolist())

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

st.subheader("Data Types")
st.dataframe(df.dtypes.astype(str))

st.subheader("Statistics")
st.dataframe(df.describe(include="all"), use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Streamlit Application")

st.write("This app is running on a deployed server.")

# Sidebar
st.sidebar.header("Controls")
rows = st.sidebar.slider("Select number of rows", 10, 100, 20)

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    "A": np.random.randn(rows),
    "B": np.random.randn(rows),
    "C": np.random.randn(rows)
})

st.subheader("Sample Dataset")
st.dataframe(data)

st.subheader("Line Chart")
st.line_chart(data)

if st.button("Show Statistics"):
    st.write(data.describe())

st.success("Application deployed successfully!")
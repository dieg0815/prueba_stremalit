import streamlit as st
import pandas as pd

data = pd.read_csv("Cobertura_mov.csv")
st.write(data)
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def grafico_hist(df, x, hue):
    fig, ax = plt.subplots()
    sns.histplot(df, x=x, hue=hue, kde=True, ax=ax)
    st.pyplot(fig)
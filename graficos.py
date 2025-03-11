import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd

def grafico_hist(df, x, hue):
    fig, ax = plt.subplots()
    sns.histplot(df, x=x, hue=hue, kde=True, ax=ax)
    st.pyplot(fig)

    
def grafico_proporcao_diabeticos_por_idade(df):
    import matplotlib.ticker as mtick
    
    df = df.copy()
    df["FaixaIdade"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 70, 80], right=False)
    proporcoes = df.groupby("FaixaIdade")["Outcome"].mean().reset_index()
    proporcoes.columns = ["FaixaIdade", "ProporcaoDiabeticos"]

    fig, ax = plt.subplots()
    sns.barplot(data=proporcoes, x="FaixaIdade", y="ProporcaoDiabeticos", palette="viridis", ax=ax)
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
    ax.set_ylabel("Proporção de Diabéticos")
    ax.set_xlabel("Faixa Etária")
    st.pyplot(fig)
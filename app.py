import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from graficos import grafico_hist, grafico_proporcao_diabeticos_por_idade

# Carregar o modelo e o scaler
modelo = joblib.load("./modelo/modelo_diabetes.pkl")
scaler = joblib.load("./modelo/scaler.pkl")

# Carregar dataset para visualização
path = "diabetes.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
df = pd.read_csv(path, names=columns)

# Tratar valores inconsistentes
def tratar_valores_inconsistentes(df):
    
    df["Glucose"] = df["Glucose"].replace(0, np.nan)
    df["BloodPressure"] = df["BloodPressure"].replace(0, np.nan)
    df["SkinThickness"] = df["SkinThickness"].replace(0, np.nan)
    df["Insulin"] = df["Insulin"].replace(0, np.nan)
    df["BMI"] = df["BMI"].replace(0, np.nan)
    
    return df

df = tratar_valores_inconsistentes(df)

# Criar menu lateral
st.sidebar.title("Menu de Navegação")
pagina = st.sidebar.radio("Escolha uma página:", ["📊 Dashboard", "🔍 Fazer Previsão"])

# --------------------- Página 1: Dashboard ---------------------
if pagina == "📊 Dashboard":
    st.title("📊 Dashboard de Análise de Diabetes")
    
    st.subheader("Distribuição da Glicose")
    grafico_hist(df, "Glucose", "Outcome")

    st.subheader("Distribuição do IMC")
    grafico_hist(df, "BMI", "Outcome")

    st.subheader("Proporção de Diabéticos por Faixa Etária")
    grafico_proporcao_diabeticos_por_idade(df)

# --------------------- Página 2: Fazer Previsão ---------------------
elif pagina == "🔍 Fazer Previsão":
    st.title("🔍 Previsão de Diabetes")

    st.write("Insira os dados clínicos do paciente para prever se ele tem diabetes.")

    valores = [
        st.number_input("Número de Gravidez", min_value=0, max_value=20, value=1),
        st.number_input("Glicose", min_value=0, max_value=300, value=120),
        st.number_input("Pressão Sanguínea", min_value=0, max_value=200, value=70),
        st.number_input("Espessura da Pele", min_value=0, max_value=100, value=20),
        st.number_input("Insulina", min_value=0, max_value=900, value=80),
        st.number_input("IMC", min_value=0.0, max_value=60.0, value=30.5),
        st.number_input("Histórico Diabetes", min_value=0.0, max_value=2.5, value=0.5),
        st.number_input("Idade", min_value=1, max_value=120, value=40),
    ]

    if st.button("🔍 Fazer Previsão"):
        dados_array = np.array([valores])
        dados_normalizados = scaler.transform(dados_array)
        previsao = modelo.predict(dados_normalizados)
        resultado = "✅ Não Diabético" if previsao[0] == 0 else "⚠️ Diabético"
        st.subheader(f"Resultado: {resultado}")


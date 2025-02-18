import streamlit as st
import pandas as pd
import numpy as np
import joblib

modelo_path = 'modelo_diabetes.pkl'
try:
    modelo = joblib.load(modelo_path)
    st.success("Modelo carregado com sucesso!")
except FileNotFoundError:
    st.error("Erro: Arquivo do modelo não encontrado. Verifique se 'modelo_diabetes.pkl' está na pasta correta.")
    st.stop()

st.title('Previsão de Presença de Diabetes')
st.markdown("Este aplicativo usa Machine Learning para prever a presença de diabetes com base em dados clínicos.")

feature_names = ['race', 'gender', 'age', 'admission_type_id', 'time_in_hospital', 'num_lab_procedures',
                 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
                 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses']

st.sidebar.subheader("Insira os dados do paciente:")
input_data = {}
for col in feature_names:
    input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

if st.sidebar.button("Realizar Predição"):
    input_df = pd.DataFrame([input_data])
    
    try:
        predicao = modelo.predict(input_df)[0]
        classificacao = "Diabetes presente" if predicao == 1 else "Sem diabetes"

        st.write("### Resultado da Predição:")
        st.write(f"A predição do modelo indica: **{classificacao}**")
    except Exception as e:
        st.error(f"Erro ao realizar a predição: {e}")

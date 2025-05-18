# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:12:09 2025

@author: manab
"""

import pandas as pd
import joblib

# Cargar modelos entrenados
modelo_aprobacion = joblib.load('modelo_xgboost_aprobacion.pkl')
modelo_riesgo = joblib.load('modelo_riesgo_impago.pkl')
scaler = joblib.load('scaler_entrenado.pkl')
# Cargar columnas esperadas
columnas_entrenamiento = joblib.load("columnas_entrenamiento.pkl")

# Ejemplo de cliente (puedes modificar estos valores)
cliente = pd.DataFrame([{
    'ScoreEvaluado': 650,
    'CantidadExcepciones': 5,
    'IdPolitica':18,
    'tipoProducto':1,
    'producto':3,
    'montoFinanciar': 10000,
    'plazo': 60,
    'mensualidad': 500,
    'tasa': 8.5,
    'region':1,
    'nivelEndeudamiento': 65,
    'capacidadDescuento': 0,
    'edad':35,
    'totalIngresos': 1800,
    'totalOtrosIngresos': 0,
    'sexo':1, 
    'idOcupacion':120
}])

# Escalar datos
cliente_scaled = scaler.transform(cliente)

# Predecir aprobación del préstamo
proba_aprobacion = modelo_aprobacion.predict_proba(cliente_scaled)[0]
pred_aprobacion = modelo_aprobacion.predict(cliente_scaled)[0]

# Predecir riesgo de impago
proba_riesgo = modelo_riesgo.predict_proba(cliente_scaled)[0]
pred_riesgo = modelo_riesgo.predict(cliente_scaled)[0]

# Mostrar resultados
estado_dict = {0: 'Aprobado', 1: 'Rechazado', 2: 'Revisión'}
riesgo_dict = {0: 'Bajo Riesgo', 1: 'Alto Riesgo'}

print("--- Resultado del Análisis del Cliente ---")
print(f"Probabilidad de aprobación: {proba_aprobacion[pred_aprobacion]*100:.2f}%")
print(f"Evaluacion por cantidad de excepciones y Score: {estado_dict[pred_aprobacion]}")
print(f"Probabilidad de incumplimiento de pago: {proba_riesgo[1]*100:.2f}% -> Riesgo: {riesgo_dict[pred_riesgo]}")
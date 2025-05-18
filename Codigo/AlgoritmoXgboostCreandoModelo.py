# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:09:15 2025

@author: manab
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar los datos
df = pd.read_csv("df_final.csv")

# Convertir IdDecision a numérico (si es necesario)
decision_mapping = {'AP': 0, 'RE': 1, 'RV': 2}
df['IdDecision'] = df['IdDecision'].map(decision_mapping)

# Eliminar columnas que no sean predictoras útiles o que contengan texto irrelevante
drop_cols = ['Identificador', 'IdAplicacion', 'IdSolicitud','FechaEvaluacion']
df = df.drop(columns=drop_cols)

# Eliminar la columna objetivo si es necesario
X_column = df.drop(columns=["IdDecision", "RiesgoImpago"], errors='ignore')

# Guardar las columnas que se usaron para entrenar
columnas_entrenamiento = X_column.columns.tolist()

joblib.dump(columnas_entrenamiento, "columnas_entrenamiento.pkl")

# Definir X y y
X = df.drop(columns=["IdDecision"])
y = df["IdDecision"]

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar modelo
modelo_aprobacion = XGBClassifier(eval_metric="mlogloss")
modelo_aprobacion.fit(X_train, y_train)

# Guardar modelo y scaler
joblib.dump(modelo_aprobacion, "modelo_xgboost_aprobacion.pkl")
joblib.dump(scaler, "scaler_entrenado.pkl")

# Crear la variable objetivo de riesgo
df["RiesgoImpago"] = (df["nivelEndeudamiento"] > 50).astype(int)

X_riesgo = df.drop(columns=["IdDecision", "RiesgoImpago"])
y_riesgo = df["RiesgoImpago"]

# Escalar con el mismo scaler
X_riesgo_scaled = scaler.transform(X_riesgo)

# Dividir y entrenar
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_riesgo_scaled, y_riesgo, test_size=0.3, random_state=42)

modelo_riesgo = XGBClassifier(eval_metric="logloss")
modelo_riesgo.fit(X_train_r, y_train_r)

# Guardar modelo
joblib.dump(modelo_riesgo, "modelo_riesgo_impago.pkl")

print("fin")

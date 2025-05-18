# -*- coding: utf-8 -*-
"""
Created on Sat May 10 23:02:43 2025

@author: manab
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Cargar dataset
df = pd.read_csv('df_final.csv')

# Convertir IdDecision a numérico (si es necesario)
decision_mapping = {'AP': 0, 'RE': 1, 'RV': 2}
df['IdDecision'] = df['IdDecision'].map(decision_mapping)

# Eliminar columnas que no sean predictoras útiles o que contengan texto irrelevante
drop_cols = ['Identificador', 'IdAplicacion', 'IdSolicitud','FechaEvaluacion']
df = df.drop(columns=drop_cols)

# Separar características (X) y variable objetivo (y)
X = df.drop('IdDecision', axis=1)  # Reemplaza con el nombre exacto de tu columna target
y = df['IdDecision']

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones
y_pred = rf_model.predict(X_test)

# Evaluación
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Matriz de confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", report)

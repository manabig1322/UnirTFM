# -*- coding: utf-8 -*-
"""
Created on Sat May 10 19:39:20 2025

@author: manab
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Cargar datos
df = pd.read_csv("df_final.csv", sep=',')

# Convertir IdDecision a numérico (si es necesario)
decision_mapping = {'AP': 0, 'RE': 1, 'RV': 2}
df['IdDecision'] = df['IdDecision'].map(decision_mapping)

# Eliminar columnas que no sean predictoras útiles o que contengan texto irrelevante
drop_cols = ['Identificador', 'IdAplicacion', 'IdSolicitud','FechaEvaluacion']
df = df.drop(columns=drop_cols)

# Eliminar filas con valores nulos
df = df.dropna()

# Separar variables predictoras y objetivo
X = df.drop(columns=['IdDecision'])
y = df['IdDecision']

# Escalar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Entrenar modelo de regresión logística multiclase
modelo = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Evaluación
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

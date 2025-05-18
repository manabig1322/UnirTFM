# -*- coding: utf-8 -*-
"""
Created on Sun May 11 00:02:42 2025

@author: manab
"""

# Instalar librerías necesarias si aún no las tienes
# !pip install pandas scikit-learn xgboost matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("df_final.csv")

# Convertir IdDecision a numérico (si es necesario)
decision_mapping = {'AP': 0, 'RE': 1, 'RV': 2}
df['IdDecision'] = df['IdDecision'].map(decision_mapping)

# Eliminar columnas que no sean predictoras útiles o que contengan texto irrelevante
drop_cols = ['Identificador', 'IdAplicacion', 'IdSolicitud','FechaEvaluacion']
df = df.drop(columns=drop_cols)

# Verificamos columnas y valores únicos del target
print("Valores únicos de 'IdDecision':", df['IdDecision'].unique())

# Separar features (X) y target (y)
X = df.drop('IdDecision', axis=1)
y = df['IdDecision']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inicializar y entrenar el modelo XGBoost
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)

# Predicciones
y_pred = xgb_model.predict(X_test)

# Matriz de confusión
conf_mat = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", conf_mat)

# Visualizar matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title("Matriz de Confusión - XGBoost")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

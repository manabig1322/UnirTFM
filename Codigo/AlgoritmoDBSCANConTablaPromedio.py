# -*- coding: utf-8 -*-
"""
Created on Sun May 11 01:33:40 2025

@author: manab
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv("df_final.csv")

# Seleccionar las columnas numéricas relevantes (ajusta según tu análisis)
features = ['ScoreEvaluado', 'CantidadExcepciones', 'montoFinanciar', 'plazo', 'mensualidad', 'tasa',
            'totalIngresos', 'nivelEndeudamiento', 'capacidadDescuento', 'totalOtrosIngresos']

# Eliminar nulos si existen
df_features = df[features].dropna()

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

# Añadir clústeres al DataFrame
df_features['Cluster'] = clusters

# Tabla resumen
summary = df_features.groupby('Cluster').mean().round(2)
summary['CantidadClientes'] = df_features['Cluster'].value_counts().sort_index()
print("Resumen de clusters:\n")
print(summary)

# Guardar la tabla resumen
summary.to_csv("resumen_clusters_dbscan.csv")

# Visualización en 2D con PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df_features['PCA1'] = components[:, 0]
df_features['PCA2'] = components[:, 1]

# Graficar clústeres
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_features, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=60)
plt.title('Visualización de clústeres con DBSCAN (PCA)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend(title='Clúster')
plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sun May 11 00:27:30 2025

@author: manab
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('df_final.csv')

# Variables relevantes para clusterización
variables_cluster = ['totalIngresos', 'nivelEndeudamiento', 'capacidadDescuento', 'edad', 'montoFinanciar']

# Preprocesamiento
X = df[variables_cluster].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar número óptimo de clusters usando el método del codo
inertia = []
K = range(1, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Graficar el codo
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Inercia')
plt.title('Método del codo para determinar el número óptimo de clusters')
plt.show()

# Elegimos K = 3 como ejemplo (puedes ajustar)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Asignar al dataframe original
df_clusters = X.copy()
df_clusters['Cluster'] = clusters

# Visualización con seaborn
sns.pairplot(df_clusters, hue='Cluster', palette='tab10')
plt.suptitle('Segmentación de clientes usando K-Means', y=1.02)
plt.show()

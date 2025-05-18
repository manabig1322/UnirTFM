# -*- coding: utf-8 -*-
"""
Created on Sun May 11 00:59:25 2025

@author: manab
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar los datos desde el CSV
df = pd.read_csv("df_final.csv")

# Seleccionar las variables numéricas para el clustering
features = ['edad', 'totalIngresos', 'nivelEndeudamiento', 'capacidadDescuento', 'montoFinanciar']

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Calcular los promedios por cluster
cluster_summary = df.groupby('cluster')[features].mean().reset_index()

# Mostrar la tabla
print(cluster_summary)

# Visualización: gráfico de barras
plt.figure(figsize=(12, 6))
cluster_summary_melted = cluster_summary.melt(id_vars='cluster', var_name='Feature', value_name='Average')
sns.barplot(data=cluster_summary_melted, x='Feature', y='Average', hue='cluster')
plt.title('Promedio de características por clúster')
plt.ylabel('Promedio')
plt.xlabel('Característica')
plt.xticks(rotation=45)
plt.legend(title='Clúster')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
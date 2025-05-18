# -*- coding: utf-8 -*-
"""
Created on Sun May 11 01:23:13 2025

@author: manab
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('df_final.csv')

# Selección de características numéricas relevantes
features = ['ScoreEvaluado', 'CantidadExcepciones', 'montoFinanciar', 'plazo', 'mensualidad', 'tasa',
            'totalIngresos', 'nivelEndeudamiento', 'capacidadDescuento', 'totalOtrosIngresos']
df_clean = df[features].dropna()

# Escalado de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Aplicar DBSCAN
db = DBSCAN(eps=1.5, min_samples=10)
labels = db.fit_predict(X_scaled)

# Añadir etiquetas al DataFrame
df_clean['Cluster'] = labels

# Visualización con reducción de dimensión (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1', legend='full')
plt.title('Clusters encontrados por DBSCAN (proyección PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Conteo de puntos por cluster
print("Distribución de clusters:", df_clean['Cluster'].value_counts())

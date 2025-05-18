# -*- coding: utf-8 -*-
"""
Created on Sat May 10 17:47:31 2025

@author: Manabi Garcia
"""
# Preparación de Datos para Modelado en Python

import pandas as pd
import json

# 1. Cargar los datos desde un archivo CSV (reemplaza con la ruta real del archivo)
archivo = 'DataSet.csv'  # Nombre referencial
df = pd.read_csv(archivo, sep=';', encoding='utf-8')

# 2. Verificar estructura inicial del DataFrame
print("Columnas disponibles:", df.columns)
print("Primeras filas:\n", df.head())

# 3. Convertir la columna DataJSON (que contiene información estructurada) a columnas normales
# Crear un DataFrame temporal con los datos JSON anidados
json_expandido = df['DataJSON'].apply(json.loads).apply(pd.Series)

# Extraer campos relevantes del JSON
clientes_df = json_expandido['clientes'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)

df_expandido = pd.concat([
    df[['Identificador', 'IdAplicacion', 'IdSolicitud', 'IdDecision', 'ScoreEvaluado', 'CantidadExcepciones', 'FechaEvaluacion', 'IdPolitica']],
    json_expandido[[
        'tipoProducto', 'producto', 'montoFinanciar', 'plazo', 'mensualidad',
        'tasa', 'region', 'nivelEndeudamiento', 'capacidadDescuento']
    ],
    clientes_df[[
        'edad', 'totalIngresos', 'totalOtrosIngresos', 'sexo', 'idOcupacion'
    ]]
], axis=1)

# 4. Conversión de tipos de datos
from datetime import datetime

df_expandido['FechaEvaluacion'] = pd.to_datetime(df_expandido['FechaEvaluacion'])
df_expandido['edad'] = pd.to_numeric(df_expandido['edad'], errors='coerce')
df_expandido['totalIngresos'] = pd.to_numeric(df_expandido['totalIngresos'], errors='coerce')
df_expandido['montoFinanciar'] = pd.to_numeric(df_expandido['montoFinanciar'], errors='coerce')
df_expandido['nivelEndeudamiento'] = pd.to_numeric(df_expandido['nivelEndeudamiento'], errors='coerce')
df_expandido['capacidadDescuento'] = pd.to_numeric(df_expandido['capacidadDescuento'], errors='coerce')

# 5. Eliminar valores nulos críticos o imputar si es necesario
df_final = df_expandido.dropna(subset=['edad', 'totalIngresos', 'montoFinanciar'])

# 6. Convertir variables categóricas
from sklearn.preprocessing import LabelEncoder

label_cols = ['sexo', 'idOcupacion', 'producto', 'tipoProducto']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders[col] = le

# 7. Guardar dataset preparado
prepared_file = 'df_final.csv'
df_final.to_csv(prepared_file, index=False)
print(f"Dataset preparado y guardado como {prepared_file}")


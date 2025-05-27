import os
import pandas as pd
from sklearn.impute import SimpleImputer

# Ruta principal donde están las carpetas tipo NF031, PF038, etc.
base_path = 'parquets/no_wind/sujetos'

# Lista para guardar todos los DataFrames
dfs = []

# Recorrer cada subcarpeta
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if folder_name.startswith('NF') or folder_name.startswith('PF'):
        if os.path.isdir(folder_path):  # asegurarse que sea carpeta
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.parquet'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_parquet(file_path)
                    dfs.append(df)

# Concatenar todos los DataFrames en uno solo
final_df = pd.concat(dfs, ignore_index=True)
columns_to_keep = ['ts', 'subject_id', 'week', 'date','cppall', 'zcrall', 'normpeakall', 
                   'spectralTiltall', 'LHratioall', 'H1H2all', 'periodicity', 'level', 'freq', 'dBcms2', 'cppall_2048',
                   'acflow', 'mfdr', 'oq', 'naq', 'h1h2', 'voicedRMS']
final_df = final_df[columns_to_keep]

features = ['cppall', 'zcrall', 'normpeakall', 'spectralTiltall', 'LHratioall', 'H1H2all', 'periodicity', 'level', 'freq', 'dBcms2', 'cppall_2048',
            'acflow', 'mfdr', 'oq', 'naq', 'h1h2']

# (Opcional) Guardarlo en un nuevo archivo parquet
final_df.to_parquet('parquets/no_wind/all_data.parquet')

print(f"Se concatenaron {len(dfs)} archivos. Dimensión final: {final_df.shape}")
print(final_df.head())  # Muestra las primeras filas del DataFrame final
print(final_df[columns_to_keep].isna().sum())
X = final_df[features]
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_imputed_df = pd.DataFrame(X_imputed, columns=features)
final_df[features] = X_imputed_df[features]

final_df.to_parquet("parquets/no_wind/all_data_imputed.parquet")


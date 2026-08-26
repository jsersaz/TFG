import numpy as np
import os
import pandas as pd


def load_obesity_levels(data_dir=None):
    """
    Carga el dataset "Estimation of Obesity Levels".
    Nombre de archivo: 'obesity_levels.csv'.
    Predecir: NObeyesdad (nivel de obesidad).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'obesity_levels')
    
    # Leer el archivo
    filepath = os.path.join(data_dir, 'obesity_levels.csv')
    # filepath = f'{data_dir}obesity_levels.csv'
    
    # Cargar el archivo
    df = pd.read_csv(filepath)
    
    # Separar características y objetivo
    X_raw = df.drop(columns=['NObeyesdad'])
    y_raw = df['NObeyesdad'].values
    
    # Mapear variables binarias 'yes'/'no' a 1/0
    binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_cols:
        X_raw[col] = X_raw[col].map({'yes': 1, 'no': 0})
    
    # Codificar variables categóricas (One-Hot Encoding)
    categorical_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=np.uint8)
    X = X_encoded.values.astype(np.float32)
    
    # Mapeo ordinal de niveles de obesidad a valores numéricos (0 a 6)
    obesity_mapping = {
        'Insufficient_Weight': 0,
        'Normal_Weight': 1,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III': 6
    }
    y = np.array([obesity_mapping[label] for label in y_raw], dtype=np.float32)
    
    # print(f"Obesity dataset (regresión) cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Objetivo (nivel de obesidad): min={y.min():.0f}, max={y.max():.0f}, media={y.mean():.2f}")
    # print("Mapeo: 0=Insufficient_Weight, 1=Normal_Weight, 2=Overweight_I, 3=Overweight_II, 4=Obesity_I, 5=Obesity_II, 6=Obesity_III")
    return X, y


if __name__ == "__main__":
    X, y = load_obesity_levels(None)
    print("Dimensiones X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 valores de y:", y[:5])

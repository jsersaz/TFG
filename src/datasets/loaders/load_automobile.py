import numpy as np
import os
import pandas as pd


def load_automobile(data_dir=None):
    """
    Carga el dataset "Automobile".
    Nombre de archivo: 'imports-85.data'.
    Predecir: price (precio del coche).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'automobile')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'imports-85.data')
    # filepath = f'{data_dir}imports-85.data'
    
    # Asignar nombres a las columnas
    columns = [
        'symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
        'num_of_doors', 'body_style', 'drive_wheels', 'engine_location',
        'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type',
        'num_of_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',
        'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price'
    ]

    # Leer el archivo
    df = pd.read_csv(filepath, header=None, names=columns, na_values='?')

    # Eliminar filas con precio faltante
    df = df.dropna(subset=['price'])

    # # Separar características y objetivo
    X_raw = df.drop(columns=['price'])
    y = df['price'].values.astype(np.float32)

    # Imputar valores NaN en características numéricas con la mediana de cada columna
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_raw[col].isnull().any():
            median_val = X_raw[col].median()
            X_raw[col] = X_raw[col].fillna(median_val)

    # Imputar valores NaN en características categóricas con la moda de cada columna
    categorical_cols = X_raw.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X_raw[col].isnull().any():
            mode_val = X_raw[col].mode()[0]
            X_raw[col] = X_raw[col].fillna(mode_val)

    # Codificar variables categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=np.uint8)
    X = X_encoded.values.astype(np.float32)

    # print(f"Automobile dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Precios: min={y.min():.0f}, max={y.max():.0f}, media={y.mean():.2f}")

    return X, y


if __name__ == "__main__":
    X, y = load_automobile(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeros 5 precios:", y[:5])

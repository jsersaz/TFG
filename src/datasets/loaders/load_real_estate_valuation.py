import numpy as np
import os
import pandas as pd


def load_real_estate_valuation(data_dir=None):
    """
    Carga el dataset "Real Estate Valuation".
    Nombre de archivo: 'real_estate_valuation.csv'.
    Predecir: Y house price of unit area (10000 New Taiwan Dollar/Ping).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'real_estate_valuation')
    
    # Leer el archivo
    filepath = os.path.join(data_dir, 'real_estate_valuation.csv')
    # filepath = f'{data_dir}real_estate_valuation.csv'
    
    # Cargar el archivo
    df = pd.read_csv(filepath)

    # Separar características y objetivo
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    # print(f"Real Estate dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Precio unitario: min={y.min():.2f}, max={y.max():.2f}, media={y.mean():.2f}")

    return X, y


if __name__ == "__main__":
    X, y = load_real_estate_valuation(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeros 5 precios:", y[:5])

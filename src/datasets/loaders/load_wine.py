import numpy as np
import os
import pandas as pd


# ----------------------------
# LOAD CLASSIFICATION (WINE)
# ----------------------------
def load_wine_classification(data_dir=None):
    """
    Carga el dataset "Wine" para clasificación.
    Nombre de archivo: 'wine.data'.
    Clases: 3 clases de vino (1, 2, 3).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'wine')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'wine.data')
    # filepath = f'{data_dir}wine.data'
    
    # Leer el archivo
    df = pd.read_csv(filepath, header=None)

    # Separar objetivo y características
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values.astype(np.float32)

    return X, y


# ------------------------------------
# LOAD CLASSIFICATION (WINE QUALITY)
# ------------------------------------
def load_wine_quality_classification(data_dir=None):
    """
    Carga el dataset "Wine Quality" para clasificación.
    Nombre de archivo: 'winequality-white.csv'.
    Clases: calidad de 0 a 10, pero se reducirá a 3 clases.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :return: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'wine_quality')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'winequality-white.csv')
    # filepath = f'{data_dir}winequality-white.csv'
    
    # Leer el archivo
    df = pd.read_csv(filepath, sep=";")

    # Separar objetivo y características
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values.astype(np.float32)
    
    # Reducción a 3 clases: 0 (calidad < 6), 1 (calidad = 6), 2 (calidad > 6)
    y = np.zeros_like(y_raw)
    y[y_raw < 6] = 0
    y[y_raw == 6] = 1
    y[y_raw > 6] = 2

    return X, y


# --------------------------------
# LOAD REGRESSION (WINE QUALITY)
# --------------------------------
def load_wine_quality_regression(data_dir=None):
    """
    Carga el dataset "Wine Quality" para regresión.
    Nombre de archivo: 'winequality-white.csv'.
    Clases: calidad de 0 a 10.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :return: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'wine_quality')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'winequality-white.csv')
    # filepath = f'{data_dir}winequality-white.csv'
    
    # Leer el archivo
    df = pd.read_csv(filepath, sep=";")

    # Separar objetivo y características
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    # print(f"Wine Quality ({wine_type}) cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Calidad: min={y.min():.0f}, max={y.max():.0f}, media={y.mean():.2f}")

    return X, y

    
# if __name__ == "__main__":
#     X, y = load_wine_cv()

#     print("X:", X.shape)
#     print("y:", y.shape)

if __name__ == "__main__":
    X, y = load_wine_quality_regression(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 calidades:", y[:5])

import numpy as np
import os
import pandas as pd


def load_gamma(data_dir=None):
    """
    Carga el dataset "MAGIC Gamma Telescope".
    Nombre de archivo: 'magic04.data'.
    Clases: gamma=0, hadron=1.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'magic_gamma_telescope')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'magic04.data')
    # filepath = f'{data_dir}magic04.data'
    
    # Asignar nombres a las columnas
    columns = [
        'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
        'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
    ]
    
    # Leer el archivo
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # Separar características y objetivo
    X = df.drop(columns=['class']).values.astype(np.float32)
    # X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df['class'].values
    
    # Codificar clases: g=0 (gamma), h=1 (hadron)
    y = np.where(y_raw == 'h', 1, 0).astype(np.float32)
    
    # print(f"MAGIC Gamma Telescope dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Distribución de clases: gamma (0) = {np.sum(y==0)}, hadron (1) = {np.sum(y==1)}")
    return X, y


if __name__ == "__main__":
    X, y = load_gamma(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

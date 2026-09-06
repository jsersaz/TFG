import numpy as np
import os
import pandas as pd


def load_mnist(data_dir=None, max_samples=10000):
    """
    Carga el dataset "MNIST".
    Nombre de archivo: 'mnist.csv'.
    Clases: dígitos del 0 al 9 (10 clases).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    :param max_samples: Número máximo de muestras a cargar.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'mnist')
    
    # Cargar y leer el archivo
    filepath = os.path.join(data_dir, 'mnist.csv')
    df = pd.read_csv(filepath, header=None)
    
    # Separar objetivo y características
    X = df.iloc[:max_samples, 1:].values.astype(np.float32)
    y = df.iloc[:max_samples, 0].values.astype(np.uint8)

    return X, y

import numpy as np
import os
import pandas as pd


def load_har(data_dir=None):
    """
    Carga el dataset "HAR (Human Activity Recognition)".
    
    Estructura de directorios:
    data_dir/
    ├── train/
    │   ├── X_train.txt
    │   └── y_train.txt
    └── test/
        ├── X_test.txt
        └── y_test.txt
        
    Clases: caminar=1, subir=2, bajar=3, sentarse=4, pararse=5, acostarse=6.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'har')
    
    # Construir rutas de los archivos
    x_train_path = os.path.join(data_dir, 'train', 'X_train.txt')
    y_train_path = os.path.join(data_dir, 'train', 'y_train.txt')
    x_test_path = os.path.join(data_dir, 'test', 'X_test.txt')
    y_test_path = os.path.join(data_dir, 'test', 'y_test.txt')
    
    # Leer los archivos
    X_train = pd.read_csv(x_train_path, sep=r'\s+', header=None).values
    y_train = pd.read_csv(y_train_path, sep=r'\s+', header=None).values.ravel()
    X_test = pd.read_csv(x_test_path, sep=r'\s+', header=None).values
    y_test = pd.read_csv(y_test_path, sep=r'\s+', header=None).values.ravel()
    
    # Separar objetivo y características, unificando train y test
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    return X, y


if __name__ == "__main__":
    X, y = load_har(None)
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

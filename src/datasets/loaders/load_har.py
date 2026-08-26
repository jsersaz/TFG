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
    
    # # Verificar que los archivos existan
    # for path in [x_train_path, y_train_path, x_test_path, y_test_path]:
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    # Leer los archivos
    X_train = pd.read_csv(x_train_path, sep=r'\s+', header=None).values
    y_train = pd.read_csv(y_train_path, sep=r'\s+', header=None).values.ravel()
    X_test = pd.read_csv(x_test_path, sep=r'\s+', header=None).values
    y_test = pd.read_csv(y_test_path, sep=r'\s+', header=None).values.ravel()
    
    # Separar objetivo y características, unificando train y test
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    # print(f"Dataset HAR cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Etiquetas únicas: {np.unique(y)} (1: Caminar, 2: Subir, 3: Bajar, 4: Sentarse, 5: Pararse, 6: Acostarse)")
    
    return X, y


if __name__ == "__main__":
    X, y = load_har(None)
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

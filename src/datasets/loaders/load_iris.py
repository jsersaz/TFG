import numpy as np
import os
import pandas as pd


def load_iris(data_dir=None):
    """
    Carga el dataset "Iris".
    Nombre de archivo: 'iris.data'.
    Clases: Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'iris')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'iris.data')
    # filepath = f'{data_dir}iris.data'
    
    # Asignar nombres a las columnas
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    # Leer el archivo
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # Separar características y objetivo
    X = df.drop(columns=['class']).values.astype(np.float32)
    # X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df['class'].values
    
    # Codificar clases a enteros (Setosa=0, Versicolour=1, Virginica=2)
    class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = np.array([class_mapping[label] for label in y_raw], dtype=np.float32)
    
    # print(f"Iris dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Clases: {list(class_mapping.keys())} -> {list(class_mapping.values())}")
    return X, y


if __name__ == "__main__":
    X, y = load_iris(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

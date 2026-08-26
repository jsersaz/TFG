import numpy as np
import os
import pandas as pd
from scipy.io import arff


def load_diabetes(data_dir=None):
    """
    Carga el dataset "Pima Indians Diabetes".
    Nombre de archivo: 'diabetes.arff'.
    Clases: tested_negative=0, tested_positive=1.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'diabetes')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'diabetes.arff')
    # filepath = f'{data_dir}diabetes.arff'
    
    # Leer el archivo
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Convertir las variables de bytes a float (excepto la clase)
    for col in df.columns[:-1]:
        df[col] = df[col].astype(np.float32)
    
    # La clase original está en bytes: b'tested_negative', b'tested_positive'
    y_raw = df['class'].values
    y = np.where(y_raw == b'tested_positive', 1, 0).astype(np.float32)
    
    # Asignar características a X
    X = df.iloc[:, :-1].values.astype(np.float32)
    
    # print(f"Diabetes dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Distribución: tested_negative={np.sum(y==0)}, tested_positive={np.sum(y==1)}")
    return X, y


if __name__ == "__main__":
    X, y = load_diabetes(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

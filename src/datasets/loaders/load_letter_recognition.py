import numpy as np
import os
import pandas as pd


def load_letter_recognition(data_dir=None):
    """
    Carga el dataset "Letter Recognition".
    Nombre de archivo: 'letter-recognition.data'.
    Clases: letras A-Z codificadas de 0 a 25.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'letter_recognition')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'letter-recognition.data')
    # filepath = f'{data_dir}letter-recognition.data'
    
    # Asignar nombres a las columnas
    columns = [
        'lettr', 'x-box', 'y-box', 'width', 'high', 'onpix',
        'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr',
        'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'
    ]
    
    # Leer el archivo
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # Separar objetivo y características
    X = df.drop(columns=['lettr']).values.astype(np.float32)
    y_raw = df['lettr'].values
    
    # Codificar clases: A-Z -> 0-25
    class_mapping = {chr(ord('A') + i): i for i in range(26)}
    y = np.array([class_mapping[ch] for ch in y_raw], dtype=np.float32)
    
    # print(f"Letter Recognition dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Distribución de clases (A-Z): {np.bincount(y.astype(int))}")
    return X, y


if __name__ == "__main__":
    X, y = load_letter_recognition(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas (letras codificadas):", y[:5])

import numpy as np
import os
import pandas as pd


def load_car(data_dir=None):
    """
    Carga el dataset "Car Evaluation".
    Nombre de archivo: 'car.data'.
    Clases: unacc=0, acc=1, good=2, v-good=3.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'car_evaluation')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'car.data')
    # filepath = f'{data_dir}car.data'
    
    # Asignar nombres a las columnas
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    
    # Leer el archivo
    df = pd.read_csv(filepath, header=None, names=columns)

    # Separar características y objetivo
    X_raw = df.drop(columns=['class'])
    y_raw = df['class'].values
    
    # Codificar variables categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X_raw, drop_first=False, dtype=np.uint8)
    X = X_encoded.values.astype(np.float32)
    
    # Codificar clases: unacc=0, acc=1, good=2, v-good=3
    class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    y = np.array([class_mapping[label] for label in y_raw], dtype=np.float32)
    
    # print(f"Car Evaluation dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Distribución de clases: unacc={np.sum(y==0)}, acc={np.sum(y==1)}, good={np.sum(y==2)}, v-good={np.sum(y==3)}")
    return X, y


if __name__ == "__main__":
    X, y = load_car(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

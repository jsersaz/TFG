import numpy as np
import os
import pandas as pd


def load_abalone(data_dir=None):
    """
    Carga el dataset "Abalone".
    Nombre de archivo: 'abalone.data'.
    Predecir: Rings (edad en años = Rings + 1.5).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'abalone')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'abalone.data')
    # filepath = f'{data_dir}abalone.data'
    
    # Asignar nombres a las columnas
    columns = ['Sex', 'Length', 'Diameter', 'Height', 
                    'Whole_weight', 'Shucked_weight', 'Viscera_weight', 
                    'Shell_weight', 'Rings']
    
    # Leer el archivo
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # Separar características y objetivo
    X_raw = df.drop(columns=['Rings'])
    y = df['Rings'].values.astype(np.float32)
    
    # Codificar variables categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X_raw, columns=['Sex'], drop_first=False, dtype=np.uint8)
    X = X_encoded.values.astype(np.float32)
    
    # print(f"Abalone dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Rings: min={y.min():.0f}, max={y.max():.0f}, media={y.mean():.2f}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_abalone(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeros 5 anillos:", y[:5])

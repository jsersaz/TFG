import numpy as np
import os
import pandas as pd


def load_bike_sharing(aggregation='hour', data_dir=None):
    """
    Carga el dataset "Bike Sharing".
    Nombre de archivos: 'hour.csv' (datos horarios), 'day.csv' (datos diarios).
    Predecir: 'cnt' (número total de alquileres).
    
    :param aggregation: 'hour' para datos horarios, 'day' para datos diarios.
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'bike_sharing')
    
    # Cargar el archivo
    if aggregation == 'hour':
        filename = 'hour.csv'
    elif aggregation == 'day':
        filename = 'day.csv'
    else:
        raise ValueError("aggregation debe ser 'hour' o 'day'")
    filepath = os.path.join(data_dir, filename)
    # filepath = f'{data_dir}{filename}'

    # Leer el archivo
    df = pd.read_csv(filepath)

    # Separar características y objetivo
    X_raw = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
        # Eliminar columnas que no deben ser características:
            # - 'instant' (índice)
            # - 'dteday' (fecha)
            # - 'casual' y 'registered' (son componentes de cnt, causarían leakage)
    y = df['cnt'].values.astype(np.float32)

    # Convertir a arrays numpy
    X = X_raw.values.astype(np.float32)

    # print(f"Bike Sharing ({aggregation}ly) cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Alquileres cnt: min={y.min():.0f}, max={y.max():.0f}, media={y.mean():.2f}")

    return X, y


if __name__ == "__main__":
    X_hour, y_hour = load_bike_sharing('hour', None)
    print("Dimensiones hourly:", X_hour.shape)
    X_day, y_day = load_bike_sharing('day', None)
    print("Dimensiones daily:", X_day.shape)

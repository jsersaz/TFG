import numpy as np
import os
import pandas as pd


def load_energy_data(data_dir=None,
                     add_time_features=True,
                     target_lags=3,
                     log_target=True):
    """
    Carga el dataset "Appliances Energy Prediction".
    Extracción de características temporales y retardos de la variable objetivo.
	Nombre de archivo: 'energydata.csv'.
    Predecir: Appliances (consumo energético en Wh).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    :param add_time_features: Si es True, añade características de fecha (hora, día de la semana, mes, fin de semana).
    :param target_lags: Número de retardos de la variable objetivo a incluir como características.
    :param log_target: Si es True, aplica transformación logarítmica al objetivo.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'appliances_energy_prediction')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'energydata.csv')
    # filepath = f'{data_dir}energydata.csv'
    
    # Leer el archivo
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Extraer características de fecha
    if add_time_features:
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek  # lunes=0, domingo=6
        df['month'] = df['date'].dt.month
        df['weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Eliminar la columna original de fecha
    df = df.drop(columns=['date'])
    
    # Separar características y objetivo
    X_raw = df.drop(columns=['Appliances'])
    y = df['Appliances'].values.astype(np.float32)
    
    # Añadir retardos de la variable objetivo
    if target_lags > 0:
        for lag in range(1, target_lags + 1):
            X_raw[f'Appliances_lag{lag}'] = df['Appliances'].shift(lag)
        # Eliminar filas con NaN
        X_raw = X_raw.iloc[target_lags:]
        y = y[target_lags:]
    
    # Eliminar filas con NaN residuales
    X_raw = X_raw.dropna()
    y = y[:len(X_raw)]
    
    # Transformación logarítmica de la variable objetivo
    if log_target:
        y = np.log1p(y)	# log(1 + Appliances)
    
    # Convertir a arrays numpy
    X = X_raw.values.astype(np.float32)
    
    # print(f"Appliances Energy cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # if log_target:
    #     print(f"log(Appliances+1): min={y.min():.4f}, max={y.max():.4f}, media={y.mean():.4f}")
    # else:
    #     print(f"Appliances (Wh): min={y.min():.2f}, max={y.max():.2f}, media={y.mean():.2f}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_energy_data(None,
                                   add_time_features=True,
                                   target_lags=3,
                                   log_target=True)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeros 5 valores de y (log):", y[:5])

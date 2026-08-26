import numpy as np
import os
import pandas as pd


def load_air_quality(data_dir=None,
                     target='CO(GT)',
                     target_lags=6,
                     weather_lags=3,
                     sensor_lags=3,
                     log_target=True,
                     fill_missing=True):
    """
    Carga el dataset "Air Quality".
    Extracción de características temporales, retardos del objetivo y variables ambientales/sensores.
    Nombre de archivo: 'AirQualityUCI.csv'.
    Predecir: CO(GT).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    :param target: Nombre de la columna objetivo a predecir.
    :param target_lags: Número de retardos del objetivo a incluir como características.
    :param weather_lags: Número de retardos de variables ambientales (T, RH, AH) a incluir como características.
    :param sensor_lags: Número de retardos de sensores (PT08.S1, PT08.S2, PT08.S3, PT08.S4, PT08.S5) a incluir como características.
    :param log_target: Si es True, aplica transformación logarítmica al objetivo.
    :param fill_missing: Si es True, imputa valores faltantes con la mediana; si es False, elimina filas con NaN.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'air_quality')
    
    # Cargar el archivo
    filepath = os.path.join(data_dir, 'AirQualityUCI.csv')
    # file_path = f'{data_dir}AirQualityUCI.csv'
    
    # Leer el archivo
    df = pd.read_csv(filepath, sep=';', decimal=',', na_values=['-200', '-200.0'],
                     encoding='utf-8', engine='python')
    
    # Eliminar columnas vacías
    df = df.dropna(axis=1, how='all')
    
    # Combinar Date y Time en una sola columna datetime (formato de hora -> "18.00.00")
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    format='%d/%m/%Y %H.%M.%S',
                                    errors='coerce')
    
    # Eliminar filas con fecha inválida
    initial_len = len(df)
    df = df.dropna(subset=['datetime'])
    # print(f"Eliminadas {initial_len - len(df)} filas por fecha inválida")
    
    # Extraer características temporales
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)
    # Cíclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Eliminar datetime original
    df = df.drop(columns=['datetime', 'Date', 'Time'])
    
    # Separar objetivo
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no encontrada. Columnas disponibles: {list(df.columns)}")
    y = df[target].values.astype(np.float32)
    
    # Imputar valores NaN con la mediana de cada columna
    if fill_missing:
        # Objetivo
        if np.isnan(y).any():
            median_y = np.nanmedian(y)
            y = np.nan_to_num(y, nan=median_y)
            print(f"Imputados {np.isnan(y).sum()} NaN en target con mediana {median_y:.4f}")
        # Características
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                # df[col].fillna(median_val, inplace=True)
                df[col] = df[col].fillna(median_val)
        print("Valores faltantes imputados con mediana en todas las columnas.")
    else:
        # Eliminar filas con NaN en objetivo
        valid = ~np.isnan(y)
        df = df[valid]
        y = y[valid]
        # Eliminar filas con NaN en características
        df = df.dropna()
        y = y[:len(df)]
    
    # Separar características
    X_raw = df.drop(columns=[target])
    
    # Añadir retardos del objetivo
    if target_lags > 0:
        for lag in range(1, target_lags + 1):
            X_raw[f'{target}_lag{lag}'] = df[target].shift(lag)
    
    # Añadir retardos de variables ambientales (temperatura, humedad)
    if weather_lags > 0:
        weather_vars = ['T', 'RH', 'AH']
        for var in weather_vars:
            if var in X_raw.columns:
                for lag in range(1, weather_lags + 1):
                    X_raw[f'{var}_lag{lag}'] = df[var].shift(lag)
    
    # Añadir retardos de sensores (PT08.S1, PT08.S2, PT08.S3, PT08.S4, PT08.S5)
    if sensor_lags > 0:
        sensor_vars = [col for col in df.columns if col.startswith('PT08.S')]
        for var in sensor_vars:
            if var in X_raw.columns:
                for lag in range(1, sensor_lags + 1):
                    X_raw[f'{var}_lag{lag}'] = df[var].shift(lag)
    
    # Eliminar filas con NaN generados por los retardos
    max_lag = max(target_lags, weather_lags, sensor_lags)
    if max_lag > 0:
        X_raw = X_raw.iloc[max_lag:]
        y = y[max_lag:]
    
    # Eliminar filas con NaN residuales
    X_raw = X_raw.dropna()
    y = y[:len(X_raw)]
    
    # Transformación logarítmica del objetivo
    if log_target:
        y = np.log1p(y)	# log(1 + CO(GT))
    
    # Convertir a arrays numpy
    X = X_raw.values.astype(np.float32)
    
    # print(f"\nAir Quality cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Target '{target}': min={y.min():.4f}, max={y.max():.4f}, media={y.mean():.4f}")
    # if log_target:
    #     print("(Target transformado con log(1+x))")
    
    return X, y


if __name__ == "__main__":
    X, y = load_air_quality(None,
                            target='CO(GT)',
                            target_lags=6,
                            weather_lags=3,
                            sensor_lags=3,
                            log_target=True,
                            fill_missing=True)
    print("Dimensiones X:", X.shape)

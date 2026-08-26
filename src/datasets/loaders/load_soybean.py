import numpy as np
import os
import pandas as pd


def load_soybean(data_dir=None):
    """
    Carga el dataset "Soybean".
    Nombre de archivo: 'soybean.data' y 'soybean.test'.
    Clases: 19 clases de enfermedades de la soja (codificadas como enteros 0-18).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'soybean')

    # Cargar y leer el archivo data
    filepath = os.path.join(data_dir, 'soybean.data')
    # filepath = f'{data_dir}soybean.data'
    df_data = pd.read_csv(filepath, header=None, na_values='?')
    
    # Cargar y leer el archivo test
    filepath_test = os.path.join(data_dir, 'soybean.test')
    # filepath_test = f'{data_dir}soybean.test'
    df_test = pd.read_csv(filepath_test, header=None, na_values='?')
    
    # Combinar archivos data y test
    df = pd.concat([df_data, df_test], axis=0, ignore_index=True)
    # print(f"Combinados soybean.data ({len(df_data)} filas) y soybean.test ({len(df_test)} filas)")
    
    # # Leer archivo de test si existe
    # filepath_test = f'{data_dir}soybean.test'
    # try:
    #     df_test = pd.read_csv(filepath_test, header=None, na_values='?')
    #     df = pd.concat([df_data, df_test], axis=0, ignore_index=True)
    #     # print(f"Combinados soybean.data ({len(df_data)} filas) y soybean.test ({len(df_test)} filas)")
    # except FileNotFoundError:
    #     df = df_data
    #     # print("Solo se encontró soybean.data")
    
    # Separar objetivo y características
    X_raw = df.iloc[:, 1:].copy()
    y_raw = df.iloc[:, 0].values
    
    # Imputar valores NaN con la moda de cada columna
    for col in X_raw.columns:
        if X_raw[col].isnull().any():
            mode_val = X_raw[col].mode()[0]
            X_raw[col] = X_raw[col].fillna(mode_val)
            # print(f"Imputados NaN en columna {col} con moda {mode_val}")
    X = X_raw.values.astype(np.float32)
    
    # Codificar clases: asignar un entero a cada nombre único
    sorted_classes = sorted(set(y_raw))
    class_mapping = {name: idx for idx, name in enumerate(sorted_classes)}
    y = np.array([class_mapping[name] for name in y_raw], dtype=np.float32)
    
    # print(f"Soybean dataset cargado: {X.shape[0]} muestras, {X.shape[1]} atributos")
    # print(f"Clases únicas: {len(sorted_classes)} (mapeo: {class_mapping})")
    return X, y


if __name__ == "__main__":
    X, y = load_soybean(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

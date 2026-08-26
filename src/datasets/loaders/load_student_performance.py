import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_student_performance(subject='math', data_dir=None):
    """
    Carga el dataset "Student Performance".
    Nombre de archivos: 'student-mat.csv' (matemáticas), 'student-por.csv' (portugués).
    Predecir: G3 (nota final).
    
    :param subject: 'math' para matemáticas, 'por' para portugués.
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'student_performance')
    
    # Leer y cargar el archivo
    if subject == 'math':
        filename = 'student-mat.csv'
        df = pd.read_csv(os.path.join(data_dir, filename), sep=';')
    elif subject == 'por':
        filename = 'student-por.csv'
        df = pd.read_csv(os.path.join(data_dir, filename), sep=';')
    elif subject == 'both':
        df_math = pd.read_csv(os.path.join(data_dir, 'student-mat.csv'), sep=';')
        df_por = pd.read_csv(os.path.join(data_dir, 'student-por.csv'), sep=';')
        # df_math = pd.read_csv(f'{data_dir}student-mat.csv', sep=';')
        # df_por = pd.read_csv(f'{data_dir}student-por.csv', sep=';')
        df = pd.concat([df_math, df_por], axis=0, ignore_index=True)
    else:
        raise ValueError("subject debe ser 'math', 'por' o 'both'")
    # if subject != 'both':
    #     df = pd.read_csv(f'{data_dir}{filename}', sep=';')
    
    # Separar características y objetivo
    X_raw = df.drop(columns=['G3'])
    y = df['G3'].values.astype(np.float32)
    
    # Identificar columnas categóricas y convertirlas a numéricas
    categorical_cols = X_raw.select_dtypes(include=['object']).columns
    
    # Codificar variables categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=np.uint8)
    X = X_encoded.values.astype(np.float32)
    
    # # Label Encoding
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     X_df[col] = le.fit_transform(X_df[col])
    
    # print(f"Student Performance ({subject}) cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Notas G3: min={y.min():.0f}, max={y.max():.0f}, media={y.mean():.2f}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_student_performance('math')
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 valores de y:", y[:5])

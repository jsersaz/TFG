import numpy as np
import os
import pandas as pd


def load_student_dropout(data_dir=None):
    """
    Carga el dataset "Predict Students' Dropout and Academic Success".
    Nombre de archivo: 'data.csv'.
    Clases: Dropout=0, Enrolled=1, Graduate=2.
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'students_dropout_and_academic_success')
    
    # Cargar y leer el archivo
    filepath = os.path.join(data_dir, 'data.csv')
    # filepath = f'{data_dir}data.csv'
    df = pd.read_csv(filepath, sep=';')
    
    # Separar objetivo y características
    X_raw = df.drop(columns=['Target'])
    y_raw = df['Target'].values
    
    # Codificar variables categóricas (One-Hot Encoding)
    categorical_cols = X_raw.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=np.uint8)
    X = X_encoded.values.astype(np.float32)
    
    # Codificar clases: Dropout -> 0, Enrolled -> 1, Graduate -> 2
    class_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    y = np.array([class_mapping[label] for label in y_raw], dtype=np.float32)
    
    # print(f"Student Dropout dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Distribución de clases: Dropout={np.sum(y==0)}, Enrolled={np.sum(y==1)}, Graduate={np.sum(y==2)}")
    return X, y


if __name__ == "__main__":
    X, y = load_student_dropout(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

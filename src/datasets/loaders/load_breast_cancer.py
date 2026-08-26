import pandas as pd
import numpy as np


def load_breast_cancer(data_dir='./data/breast_cancer_wisconsin/wdbc.data'):
    """
    Carga el dataset Wisconsin Diagnostic Breast Cancer (WDBC).
    Se espera el archivo 'wdbc.data' (o similar) con formato:
    ID, diagnosis, 30 características numéricas.
    Devuelve X (30 atributos) e y (0=benigno, 1=maligno).
    """
    # Nombres de columnas basados en documentación
    # (excepto ID y diagnosis, luego se asignan)
    # Crearemos una lista dinámica: 'id', 'diagnosis', y luego 30 nombres
    feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                     'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    
    # Para mean, se, worst
    col_names = ['id', 'diagnosis']
    for suffix in ['_mean', '_se', '_worst']:
        for f in feature_names:
            col_names.append(f + suffix)
    
    # Leer archivo
    filepath = data_dir
    df = pd.read_csv(filepath, header=None)
    # Asignar nombres
    df.columns = col_names
    
    # Eliminar ID
    df = df.drop(columns=['id'])
    
    # Separar características y objetivo
    X = df.drop(columns=['diagnosis']).values.astype(np.float32)
    y_raw = df['diagnosis'].values
    
    # Codificar: M -> 1 (maligno), B -> 0 (benigno)
    y = np.where(y_raw == 'M', 1, 0).astype(np.float32)
    
    # print(f"WDBC dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"Distribución: benigno (0) = {np.sum(y==0)}, maligno (1) = {np.sum(y==1)}")
    return X, y

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta completa al archivo (ajústala según tu estructura)
    X, y = load_breast_cancer('./data/breast_cancer_wisconsin/wdbc.data')
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# def estandarizar(X):
#     scaler = StandardScaler()

#     X_scaled = scaler.fit_transform(X)
#     return pd.DataFrame(
#         X_scaled,
#         columns=X.columns,
#         index=X.index
#     )

# def preprocesamiento(X):
#     X_clean = X.replace("?", np.nan)
#     X_numeric = X_clean.apply(pd.to_numeric, errors="coerce")
#     return X_numeric

# def load_breast_cancer(filepath="./data/breast_cancer_wisconsin/breast-cancer-wisconsin.data"):
#     df_cancer = pd.read_csv(filepath, header=None)

#     df_cancer.columns = [
#         "Sample_code_number",
#         "Clump_Thickness",
#         "Uniformity_Cell_Size",
#         "Uniformity_Cell_Shape",
#         "Marginal_Adhesion",
#         "Single_Epithelial_Cell_Size",
#         "Bare_Nuclei",
#         "Bland_Chromatin",
#         "Normal_Nucleoli",
#         "Mitoses",
#         "Class"
#     ]
    
#     X_cancer = df_cancer.drop(columns=["Class", "Sample_code_number"])
#     y_cancer = df_cancer["Class"]
    
#     X_cancer_pre = preprocesamiento(X_cancer)
#     mask = ~X_cancer_pre.isna().any(axis=1)
#     X_pre = X_cancer_pre[mask]
#     y = y_cancer[mask]
    
#     X = estandarizar(X_pre)
    
#     return X, y
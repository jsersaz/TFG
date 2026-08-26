import pandas as pd
import numpy as np


def load_mushroom(data_dir='./data/mushroom/'):
    """
    Carga el dataset Mushroom para clasificación (edible/poisonous).
    Los valores faltantes (atributo stalk-root) se imputan con la moda.
    Las variables categóricas se convierten a numéricas mediante One-Hot Encoding.
    """
    # Nombres de las 22 columnas según la documentación
    column_names = [
        'class',                     # edible=e, poisonous=p
        'cap-shape',                 # b,c,x,f,k,s
        'cap-surface',               # f,g,y,s
        'cap-color',                 # n,b,c,g,r,p,u,e,w,y
        'bruises',                   # t, f
        'odor',                      # a,l,c,y,f,m,n,p,s
        'gill-attachment',           # a,d,f,n
        'gill-spacing',              # c,w,d
        'gill-size',                 # b,n
        'gill-color',                # k,n,b,h,g,r,o,p,u,e,w,y
        'stalk-shape',               # e,t
        'stalk-root',                # b,c,u,e,z,r,?  (missing)
        'stalk-surface-above-ring',  # f,y,k,s
        'stalk-surface-below-ring',  # f,y,k,s
        'stalk-color-above-ring',    # n,b,c,g,o,p,e,w,y
        'stalk-color-below-ring',    # n,b,c,g,o,p,e,w,y
        'veil-type',                 # p,u
        'veil-color',                # n,o,w,y
        'ring-number',               # n,o,t
        'ring-type',                 # c,e,f,l,n,p,s,z
        'spore-print-color',         # k,n,b,h,r,o,u,w,y
        'population',                # a,c,n,s,v,y
        'habitat'                    # g,l,m,p,u,w,d
    ]
    
    # Leer el archivo (sin cabecera, valores '?' como NaN)
    filepath = f'{data_dir}agaricus-lepiota.data'
    df = pd.read_csv(filepath, header=None, names=column_names, na_values='?')
    
    # Separar variable objetivo y características
    y_raw = df['class'].values
    X_df = df.drop(columns=['class'])
    
    # Imputar valores faltantes (solo en stalk-root) con la moda de esa columna
    for col in X_df.columns:
        if X_df[col].isnull().any():
            mode_val = X_df[col].mode()[0]
            X_df[col] = X_df[col].fillna(mode_val)
            print(f"Imputados valores faltantes en '{col}' con la moda '{mode_val}'")
    
    # Aplicar One-Hot Encoding a todas las columnas categóricas (son todas)
    X_encoded = pd.get_dummies(X_df, drop_first=False, dtype=np.uint8)
    
    # Codificar la variable objetivo: e → 0 (comestible), p → 1 (venenoso)
    y = np.where(y_raw == 'p', 1, 0).astype(np.float32)
    
    # Convertir a arrays numpy
    X = X_encoded.values.astype(np.float32)
    
    print(f"Mushroom dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: comestible (0) = {np.sum(y==0)}, venenoso (1) = {np.sum(y==1)}")
    return X, y

# Ejemplo de uso
if __name__ == "__main__":
    X, y = load_mushroom('./data/mushroom/')
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeras 5 etiquetas:", y[:5])

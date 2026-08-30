import numpy as np
import os
import pandas as pd

from src.datasets.datasets_classification import get_classification_datasets
from src.models.models_classification import get_classification_models
from src.core.tinyml_pipeline import (
    print_results_table_classification,
    run_classification
)
from src.utils.utils import get_project_root


# -----------------------------------------------------
# EJECUCIÓN PRINCIPAL DE LOS MODELOS DE CLASIFICACIÓN
# -----------------------------------------------------
def run_models():
    """
    Itera sobre todos los datasets de clasificación y modelos definidos.
    Ejecuta la validación cruzada repetida y recoge los resultados.
    Devuelve una lista con los resultados agregados de todos los datasets.
    
    :returns: Lista con los resultados de la evaluación.
    """
    # Obtener el diccionario de datasets: nombre -> función de carga
    datasets = get_classification_datasets()
    # models = get_classification_models()

    all_results = []    # Acumulará los resultados de todos los datasets

    # Recorrer cada dataset
    for dataset_name, load_fn in datasets.items():

        print("\n########################################")
        print(f"DATASET: {dataset_name}")
        print("########################################")

        # Cargar los datos
        X, y = load_fn()
        
        # Convertir etiquetas a 0‑index si empiezan en 1
        if y.min() == 1 and y.max() == len(np.unique(y)):
            y = y - 1
            print(f"Etiquetas convertidas a 0‑index: {np.unique(y)}")

        # Obtener información básica del dataset
        input_dim = X.shape[1]			# número de características
        num_classes = len(np.unique(y))	# número de clases
        
        # Construir el diccionario de modelos
        # Los modelos de sklearn ya están preconfigurados; los TinyML necesitan input_dim y num_classes
        models = get_classification_models(input_dim, num_classes)

		# Lista para guardar los resultados del dataset
        dataset_results = []

		# Iterar sobre cada modelo
        for model_name, model_builder in models.items():

            print("\n==========================")
            print(f"Running {model_name}")
            print("==========================")

			# Ejecutar la prueba de clasificación con validación cruzada repetida
            res = run_classification(
                model_name,
                model_builder,
                X,
                y
            )

            res["Dataset"] = dataset_name

            dataset_results.append(res)
            all_results.append(res)

		# Mostrar la tabla de resultados para el dataset
        print_results_table_classification(dataset_results)

    return all_results


# ---------------------------------------------------
# MOSTRAR RESULTADOS GLOBALES (TABLA PIVOTE CON F1)
# ---------------------------------------------------
def print_global_results(all_results):
    """
    Convierte la lista de resultados en un DataFrame y muestra una tabla pivote
    con los valores medios de F1 por dataset y modelo.
    
    :param all_results: Lista de diccionarios con los resultados de todos los datasets.
    """
    df_all = pd.DataFrame(all_results)

    print("\n\n=== GLOBAL RESULTS (F1) ===\n")

	# Crear tabla pivote: filas = datasets, columnas = modelos, valores = F1_mean
    tabla_f1 = df_all.pivot(
        index="Dataset",
        columns="Model",
        values="F1_mean"
    )

    print(tabla_f1.round(5))


# ------------------------------------
# GUARDAR RESULTADOS GLOBALES EN CSV
# ------------------------------------
def save_global_results_to_csv(all_results, filename="tflite_distill_t2_static_classification_results.csv"):
    """
    Guarda los resultados globales en archivos CSV dentro de la carpeta 'results'.
    - Un archivo con todos los resultados (DataFrame completo).
    - Archivos separados para las tablas pivote de cada métrica.
    
    :param all_results: Lista de diccionarios con los resultados de todos los datasets.
    :param filename: Nombre del archivo CSV para guardar los resultados completos.
    """
    # Convertir lista de resultados a DataFrame
    df_all = pd.DataFrame(all_results)
    # Redondear todas las columnas numéricas a 5 decimales
    df_all = df_all.round(5)
    
    # Obtener la raíz del proyecto y la carpeta 'results'
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "results2")
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar el DataFrame
    full_path = os.path.join(results_dir, filename)
    df_all.to_csv(full_path, index=False)
    print(f"\nResultados completos guardados en {full_path}")
    
    # Tablas pivote para cada métrica
    # Diccionario: nombre_archivo -> nombre_columna en el DataFrame
    metrics = {
        "accuracy": "Accuracy_mean",
        "f1": "F1_mean",
        "precision": "Precision_mean",
        "recall": "Recall_mean",
        "size": "Size_KB",
        "train": "Train_time_s",
        "inference": "Inference_time_s"
    }
    
    for name, col in metrics.items():
        # Crear tabla pivote: filas = datasets, columnas = modelos, valores = col
        tabla = df_all.pivot(index="Dataset", columns="Model", values=col)
        file_name = f"tflite_distill_t2_static_classification_{name}.csv"
        file_path = os.path.join(results_dir, file_name)
        tabla.to_csv(file_path)
        print(f"Tabla pivote de {col} guardada en {file_path}")

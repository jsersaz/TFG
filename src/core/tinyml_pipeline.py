import numpy as np
import os
import pickle
import time
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score
)


# -----------------------------------
# MEDICIÓN DEL TIEMPO DE INFERENCIA
# -----------------------------------
def measure_inference_time(model, X, n_runs=100):
    """
    Mide el tiempo de inferencia del modelo sobre una muestra X.
    Realiza una ejecución de calentamiento (warmup) y luego n_runs mediciones.
    Devuelve la mediana de los tiempos (más robusta que la media).
    
    :param model: Modelo a evaluar (debe tener método predict).
    :param X: Datos de entrada para la inferencia (numpy array o DataFrame).
    :param n_runs: Número de ejecuciones para medir el tiempo.
    
    :returns: Mediana del tiempo de inferencia en segundos.
    """
    model.predict(X)  # warmup: primera ejecución (ignorar tiempo)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()	# temporizador de alta resolución
        model.predict(X)
        times.append(time.perf_counter() - start)

    return np.median(times)	# mediana para evitar outliers


# ------------------------
# TAMAÑO DEL MODELO (KB)
# ------------------------
def measure_model_size(model):
    """
    Calcula el tamaño del modelo en KB.
    Para modelos con atributo '_model_path' (ej. ONNX, TorchScript, TFLite) se obtiene el tamaño del archivo.
    Para modelos scikit-learn estándar se serializa con pickle y se mide su longitud.
    
    :param model: Modelo a evaluar (puede ser scikit-learn, ONNX, TorchScript, TFLite).
    
    :returns: Tamaño del modelo en KB.
    """
    if hasattr(model, "_model_path") and model._model_path is not None:
        # Modelo guardado en archivo (ONNX, TorchScript, TFLite)
        return os.path.getsize(model._model_path) / 1024  # KB
    else:
        # Modelo scikit-learn en memoria
        size = len(pickle.dumps(model))
        return size / 1024  # KB


# ---------------------------
# MÉTRICAS DE CLASIFICACIÓN
# ---------------------------
def evaluate_classification(model, X_test, y_test):
    """
    Evalúa un clasificador en el conjunto de test.
    Calcula accuracy, F1 macro, precision macro y recall macro.
    
    :param model: Clasificador a evaluar (debe tener método predict).
    :param X_test: Datos de test (numpy array o DataFrame).
    :param y_test: Etiquetas verdaderas del test (numpy array o Series).
    
    :returns: accuracy, precision, recall, F1.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    return acc, f1, precision, recall


# -----------------------
# MÉTRICAS DE REGRESIÓN
# -----------------------
def evaluate_regression(model, X_test, y_test):
    """
    Evalúa un regresor en el conjunto de test.
    Calcula MAE, RMSE y R².
    
    :param model: Regresor a evaluar (debe tener método predict).
    :param X_test: Datos de test (numpy array o DataFrame).
    :param y_test: Etiquetas verdaderas del test (numpy array o Series).
    
    :returns: MAE, RMSE, R².
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2


# --------------------------------------
# CLASIFICACIÓN CON VALIDACIÓN CRUZADA
# --------------------------------------
def run_classification(
    name,
    model_builder,
    X,
    y,
    n_splits=5,
    n_repeats=3,
    random_state=42
):
    """
    Ejecuta validación cruzada repetida y estratificada para clasificación (RepeatedStratifiedKFold).
    Para cada fold:
    	- Entrena el modelo.
		- Mide tiempo de entrenamiento, inferencia, tamaño del modelo y métricas (accuracy, F1, precision, recall).
	Acumula resultados y devuelve medias y desviaciones.
 
    :param name: Nombre del modelo (string).
    :param model_builder: Función que devuelve una nueva instancia del modelo.
    :param X: Datos de entrada (numpy array o DataFrame).
    :param y: Etiquetas verdaderas (numpy array o Series).
    :param n_splits: Número de folds para KFold.
    :param n_repeats: Número de repeticiones del KFold.
    :param random_state: Semilla para reproducibilidad.
    
    :returns: Diccionario con métricas medias y desviaciones.
    """
    # Convertir DataFrames a arrays numpy si es necesario
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    # Asegurar que y es un array 1D
    y = np.ravel(y)

	# Validación cruzada repetida y estratificada (mantiene proporción de clases)
	# Se repite n_repeats veces el proceso de n_splits folds, barajando antes de cada repetición.
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

	# Listas para almacenar resultados de cada fold
    acc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    size_list = []
    time_list = []
    train_time_list = []
    conversion_time_list = []
    size_kb = None
    
    # Calcular número total de folds
    total_folds = n_splits * n_repeats
    current_fold = 0
    
    # Iterar sobre cada partición generada por el validador cruzado
    for train_idx, test_idx in rskf.split(X, y):
        # Incrementar contador de fold y calcular repetición actual y fold dentro de la repetición
        current_fold += 1
        repeat = (current_fold - 1) // n_splits + 1
        fold_in_repeat = (current_fold - 1) % n_splits + 1
        
        # Informar del progreso con timestamp
        print(f"  [Fold {current_fold}/{total_folds} (Rep. {repeat}/{n_repeats}, Fold {fold_in_repeat}/{n_splits})] - {time.strftime('%H:%M:%S')}")

		# Dividir datos en train y test para este fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

		# Construir una nueva instancia del modelo
        model = clone(model_builder)
        # model = model_builder

		# Entrenamiento
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        end_train = time.perf_counter()

        total_train_time = end_train - start_train

		# Usar atributos propios de tiempo de sklearn y conversión (wrappers)
        if hasattr(model, 'train_time_') and hasattr(model, 'conversion_time_'):
            train_time = model.train_time_
            conversion_time = model.conversion_time_
        # Usar el tiempo total calculado anteriormente
        else:
            train_time = total_train_time
            conversion_time = 0.0

		# Evaluación de métricas
        acc, f1, precision, recall = evaluate_classification(model, X_test, y_test)

		# Calcular tiempo de inferencia
        inference_time = measure_inference_time(model, X_test[:1])

		# Calcular tamaño del modelo (solo se mide una vez, ya que es constante)
        if size_kb is None:
            size_kb = measure_model_size(model)
            size_list.append(size_kb)

		# Almacenar resultados del fold
        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        time_list.append(inference_time)
        train_time_list.append(train_time)
        conversion_time_list.append(conversion_time)
        
        print(f"    -> OK. F1: {f1:.3f}")

	# Resultados finales: media y desviación típica de cada métrica
    results = {
        "Model": name,

        "Accuracy_mean": np.mean(acc_list),
        "Accuracy_std": np.std(acc_list),

        "F1_mean": np.mean(f1_list),
        "F1_std": np.std(f1_list),

        "Precision_mean": np.mean(precision_list),
        "Precision_std": np.std(precision_list),

        "Recall_mean": np.mean(recall_list),
        "Recall_std": np.std(recall_list),

        # "Size_KB": size_kb,
        "Size_KB": np.mean(size_list),
        "Inference_time_s": np.mean(time_list),
        "Train_time_s": np.mean(train_time_list),
        "Conversion_time_s": np.mean(conversion_time_list),
    }

    return results


# -----------------------------------
#  REGRESIÓN CON VALIDACIÓN CRUZADA
# -----------------------------------
def run_regression(
    name,
    model_builder,
    X,
    y,
    n_splits=5,
    n_repeats=3,
    random_state=42
):
    """
    Ejecuta validación cruzada repetida para regresión (RepeatedKFold: sin estratificar porque la variable objetivo es continua).
	Para cada fold:
    	- Entrena el modelo.
		- Mide tiempo de entrenamiento, inferencia, tamaño del modelo y métricas (MAE, RMSE, R²).
	Acumula resultados y devuelve medias y desviaciones.
 
    :param name: Nombre del modelo (string).
    :param model_builder: Función que devuelve una nueva instancia del modelo.
    :param X: Datos de entrada (numpy array o DataFrame).
    :param y: Etiquetas verdaderas (numpy array o Series).
    :param n_splits: Número de folds para KFold.
    :param n_repeats: Número de repeticiones del KFold.
    :param random_state: Semilla para reproducibilidad.
    
    :returns: Diccionario con métricas medias y desviaciones.
    """
    # Convertir DataFrames a arrays numpy si es necesario
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    # Asegurar que y es un array 1D
    y = np.ravel(y)

	# Validación cruzada repetida
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

	# Listas para almacenar resultados de cada fold
    mae_list = []
    rmse_list = []
    r2_list = []
    size_list = []
    time_list = []
    train_time_list = []
    conversion_time_list = []
    size_kb = None

	# Calcular número total de folds
    total_folds = n_splits * n_repeats
    current_fold = 0

	# Iterar sobre cada partición generada por el validador cruzado
    for train_idx, test_idx in rkf.split(X):
        # Incrementar contador de fold y calcular repetición actual y fold dentro de la repetición
        current_fold += 1
        repeat = (current_fold - 1) // n_splits + 1
        fold_in_repeat = (current_fold - 1) % n_splits + 1

		# Informar del progreso con timestamp
        print(f"  [Fold {current_fold}/{total_folds} (Rep. {repeat}/{n_repeats}, Fold {fold_in_repeat}/{n_splits})] - {time.strftime('%H:%M:%S')}")

		# Dividir datos en train y test para este fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

		# Construir una nueva instancia del modelo (cada fold parte de cero)
        model = clone(model_builder)
        # model = model_builder
        
        # Entrenamiento
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        end_train = time.perf_counter()
        
        total_train_time = end_train - start_train

		# Usar atributos propios de tiempo de sklearn y conversión (wrappers)
        if hasattr(model, 'train_time_') and hasattr(model, 'conversion_time_'):
            train_time = model.train_time_
            conversion_time = model.conversion_time_
        # Usar el tiempo total calculado anteriormente
        else:
            train_time = total_train_time
            conversion_time = 0.0

		# Evaluación del modelo
        mae, rmse, r2 = evaluate_regression(model, X_test, y_test)
        
		# Calcular tiempo de inferencia
        inference_time = measure_inference_time(model, X_test[:1])
        
		# Calcular tamaño del modelo (solo se mide una vez, ya que es constante)
        if size_kb is None:
            size_kb = measure_model_size(model)
            size_list.append(size_kb)

		# Almacenar resultados del fold
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        time_list.append(inference_time)
        train_time_list.append(train_time)
        conversion_time_list.append(conversion_time)

        print(f"    -> OK. R²: {r2:.3f}")

	# Resultados finales: media y desviación típica de cada métrica
    results = {
        "Model": name,
        
        "MAE_mean": np.mean(mae_list),
        "MAE_std": np.std(mae_list),
        
        "RMSE_mean": np.mean(rmse_list),
        "RMSE_std": np.std(rmse_list),
        
        "R2_mean": np.mean(r2_list),
        "R2_std": np.std(r2_list),
        
        # "Size_KB": size_kb,
        "Size_KB": np.mean(size_list),
        
        "Train_time_s": np.mean(train_time_list),
        "Conversion_time_s": np.mean(conversion_time_list),
        "Inference_time_s": np.mean(time_list),
    }

    return results


# ---------------------------------------------
# MOSTRAR RESULTADOS EN TABLA (CLASIFICACIÓN)
# ---------------------------------------------
def print_results_table_classification(results_list):
    """
    Muestra una tabla con los resultados de clasificación.
    Formatea cada métrica como media ± desviación típica.
    
    :param results_list: Lista de diccionarios con los resultados de cada modelo.
    """
    # Verificar si algún modelo reporta tiempo de conversión (modelos comprimidos)
    show_conversion = any(r.get("Conversion_time_s", 0) > 0 for r in results_list)

    # Definir la cabecera de la tabla según corresponda
    if show_conversion:
        header = "{:<15} {:<18} {:<18} {:<18} {:<18} {:<12} {:<15} {:<15} {:<15}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1",
            "Size(KB)", "Train time(s)", "Conversion time(s)", "Infer time(s)"
        )
    else:
        header = "{:<15} {:<18} {:<18} {:<18} {:<18} {:<12} {:<15} {:<15}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1",
            "Size(KB)", "Train time(s)", "Infer time(s)"
        )
    print(header)
    print("-" * len(header))

	# Iterar sobre cada resultado (cada modelo/dataset)
    for r in results_list:
        # Formatear cada métrica como "media ± desviación" con precisión fija
        acc = f"{r['Accuracy_mean']:.3f} ± {r['Accuracy_std']:.3f}"
        f1 = f"{r['F1_mean']:.3f} ± {r['F1_std']:.3f}"
        prec = f"{r['Precision_mean']:.3f} ± {r['Precision_std']:.3f}"
        rec = f"{r['Recall_mean']:.3f} ± {r['Recall_std']:.3f}"

		# Imprimir fila
        if show_conversion:
            print("{:<15} {:<18} {:<18} {:<18} {:<18} {:<12.3f} {:<15.3e} {:<15.3e} {:<15.3e}".format(
                r["Model"], acc, prec, rec, f1,
                r["Size_KB"],
                r["Train_time_s"],
                r["Conversion_time_s"],
                # r.get("Conversion_time_s", 0.0),
                r["Inference_time_s"]
            ))
        else:
            print("{:<15} {:<18} {:<18} {:<18} {:<18} {:<12.3f} {:<15.3e} {:<15.3e}".format(
                r["Model"], acc, prec, rec, f1,
                r["Size_KB"],
                r["Train_time_s"],
                r["Inference_time_s"]
            ))


# -----------------------------------------
# MOSTRAR RESULTADOS EN TABLA (REGRESIÓN)
# -----------------------------------------
def print_results_table_regression(results_list):
    """
    Muestra una tabla con los resultados de regresión.
    Formatea cada métrica como media ± desviación típica.
    
    :param results_list: Lista de diccionarios con los resultados de cada modelo.
    """
    # Verificar si algún modelo reporta tiempo de conversión (modelos comprimidos)
    show_conversion = any(r.get("Conversion_time_s", 0) > 0 for r in results_list)

    print("\n=== Regression Results ===\n")

    # Definir la cabecera de la tabla según corresponda
    if show_conversion:
        header = "{:<15} {:<18} {:<18} {:<18} {:<12} {:<15} {:<15} {:<15}".format(
            "Model", "MAE", "RMSE", "R2",
            "Size(KB)", "Train time(s)", "Conversion time(s)", "Infer time(s)"
        )
    else:
        header = "{:<15} {:<18} {:<18} {:<18} {:<12} {:<15} {:<15}".format(
            "Model", "MAE", "RMSE", "R2",
            "Size(KB)", "Train time(s)", "Infer time(s)"
        )
    print(header)
    print("-" * len(header))

	# Iterar sobre cada resultado (cada modelo/dataset)
    for r in results_list:
        mae = f"{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}"
        rmse = f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f}"
        r2 = f"{r['R2_mean']:.3f} ± {r['R2_std']:.3f}"

		# Imprimir fila
        if show_conversion:
            print("{:<15} {:<18} {:<18} {:<18} {:<12.3f} {:<15.3e} {:<15.3e} {:<15.3e}".format(
                r["Model"], mae, rmse, r2,
                r["Size_KB"],
                r["Train_time_s"],
                r["Conversion_time_s"],
                # r.get("Conversion_time_s", 0.0),
                r["Inference_time_s"]
            ))
        else:
            print("{:<15} {:<18} {:<18} {:<18} {:<12.3f} {:<15.3e} {:<15.3e}".format(
                r["Model"], mae, rmse, r2,
                r["Size_KB"],
                r["Train_time_s"],
                r["Inference_time_s"]
            ))

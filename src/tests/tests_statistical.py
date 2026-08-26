import argparse
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy.stats import f
from statds.no_parametrics import friedman, nemenyi


# ===========================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO (ETL)
# ===========================================
def load_and_merge_csvs(filepaths, task, filter_keyword=None):
    """
    Carga una lista de archivos CSV y los fusiona utilizando la columna 'Dataset' como índice.
    Parsea los nombres de los archivos para extraer el framework y la configuración aplicada 
    (cuantización, destilación) y renombra las columnas de los modelos para hacerlas autoexplicativas.
    
    :param filepaths: Lista de rutas absolutas o relativas a los ficheros CSV a cargar.
    :param task: Tarea específica para la que se están cargando los datos ('classification' o 'regression').
    :param filter_keyword: Cadena de texto para filtrar y quedarse solo con las columnas que la contengan (ej. "MLP").
                           
    :returns: DataFrame unificado con todos los modelos y configuraciones como columnas.
    """
    # Diccionario de traducción para formatear el nombre de los frameworks en los gráficos
    framework_names = {
        "sklearn": "scikit-learn",
        "onnx": "ONNX",
        "pytorch": "PyTorch",
        "tflite": "TFLite"
    }
    
    # Diccionario para formatear los nombres técnicos de las configuraciones a nombres legibles
    config_names = {
        "dynamic": "Dinámica",
        "static": "Estática",
        "distill_t1": "Dest. T1",
        "distill_t2": "Dest. T2",
        "distill_t1_dynamic": "Dest. T1 + Din.",
        "distill_t2_dynamic": "Dest. T2 + Din.",
        "distill_t1_static": "Dest. T1 + Est.",
        "distill_t2_static": "Dest. T2 + Est."
    }

    df_merged = pd.DataFrame()
    split_key = f"_{task}_"
    
    for path in filepaths:
        if not os.path.exists(path):
            print(f"Advertencia: No se ha encontrado {path}")
            continue
            
        # 1. Extracción de metadatos desde el nombre del archivo (framework y configuración)
        basename = os.path.basename(path)
        framework = "Unknown"
        config = ""
        
        # Cortar el nombre del fichero según la tarea
        # Ejemplo: de 'pytorch_distill_t1_dynamic_classification_f1.csv'
				# extraemos left_part = 'pytorch_distill_t1_dynamic'
        if split_key in basename:
            left_part = basename.split(split_key)[0]
            parts = left_part.split("_", 1)		# Corta solo en el primer guion bajo
            framework = parts[0].lower()
            if len(parts) > 1:
                config = parts[1].lower()		# Ej: 'distill_t1_dynamic'
        
        # 2. Carga del CSV estableciendo 'Dataset' como índice para facilitar cruces
        df_temp = pd.read_csv(path).set_index('Dataset')
        
        # 3. Construcción del sufijo identificativo para las columnas
        fw_pretty = framework_names.get(framework, framework.capitalize())
        if config:
            conf_pretty = config_names.get(config, config.replace('_', ' ').title())
            suffix = f"{fw_pretty} | {conf_pretty}"
        else:
            suffix = fw_pretty
            
        # 4. Aplicación del sufijo a los nombres de las columnas
        # Ej: "MLP_Clas" -> "MLP_Clas (PyTorch | Dest. T1 + Din.)"
        df_temp.columns = [f"{col} ({suffix})" for col in df_temp.columns]
        
        # 5. Unión al DataFrame principal
        if df_merged.empty:
            df_merged = df_temp
        else:
            # Join por índice ('Dataset'). 'rsuffix' evita colisiones temporales de nombres
            df_merged = df_merged.join(df_temp, how='outer', rsuffix='_dup')
            # Eliminar cualquier columna duplicada accidentalmente
            df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_dup')]

    # 6. Filtrado opcional de columnas por palabra clave (caso de MLP)
    if filter_keyword:
        col_mask = df_merged.columns.str.contains(filter_keyword, case=False, na=False)
        df_merged = df_merged.loc[:, col_mask]
        
    return df_merged

def determine_minimization(metric_name):
    """
    Evalúa semánticamente el nombre de la métrica para decidir si el test estadístico 
    debe interpretarla buscando el valor mínimo (ej. tiempos, tamaños) o el máximo (ej. accuracy, F1).
    
    :param metric_name: Nombre de la métrica evaluada.
    
    :returns: True si la métrica requiere ser minimizada (el menor es mejor), False en caso contrario.
    """
    metric_lower = metric_name.lower()
    # Lista de palabras clave que implican minimización (tiempo, peso)
    min_metrics = ['mae', 'rmse', 'train', 'inference', 'size']
    return any(m in metric_lower for m in min_metrics)


# ==================================
# FUNCIONES DE ANÁLISIS ESTADÍSTICO
# ==================================
def run_statistical_analysis(df_results, metric, task, output_dir=None, framework="sklearn"):
    """
    Orquesta la batería completa de tests estadísticos no paramétricos.
    Ejecuta Friedman para rangos, Iman-Davenport como corrección de significancia, 
    y Nemenyi como análisis post-hoc (si aplica). Genera visualizaciones automáticas.
    
    :param df_results: DataFrame con los modelos en columnas, datasets en filas y rendimiento en valores.
    :param metric: Métrica evaluada (para formateo de textos y lógica de minimización).
    :param task: Tarea específica para la que se están cargando los datos ('classification' o 'regression').
    :param output_dir: Directorio de salida para guardar los gráficos. Si es None, los muestra en pantalla.
    :param framework: Identificador de la fase o grupo que se está analizando para nombrar archivos.
    """
    # Mapeo para mejorar la presentación visual en títulos de gráficos
    display_names = {
        "sklearn": "scikit-learn",
        "onnx": "ONNX",
        "pytorch": "PyTorch",
        "tflite": "TensorFlow Lite",
        "mlp": "MLP"
    }
    framework_display = display_names.get(framework.lower(), framework.capitalize())
    framework_prefix = framework.lower()

    print(f"\n" + "="*50)
    print(f" ANALIZANDO MÉTRICA: {metric.upper()} | TAREA: {task.upper()} ({framework_display})")
    print("="*50)
    print(df_results.round(4))
    
    # Determinar dirección de optimalidad para asignar los rangos matemáticos correctamente
    is_minimize = determine_minimization(metric)
    print(f"\n--> Configuración de Friedman: minimize={is_minimize}")
    
    # Resetear índice para asegurar la compatibilidad con statds
    results_table_sd = df_results.reset_index()
    
    # --------------------
    # 1. TEST DE FRIEDMAN
    # --------------------
    rankings, statistic, p_value, critical_value, hypothesis = friedman(
        results_table_sd, alpha=0.05, minimize=is_minimize
    )
    
    print(f"\n=== Test de Friedman ({metric}) ===")
    print(f"Hipótesis: {hypothesis}")
    print(f"Estadístico: {statistic:.4f}")
    print(f"Valor crítico: {critical_value:.4f}")
    print(f"p-valor: {p_value:.6f}")
    
    average_rankings = pd.Series(rankings, index=df_results.columns)
    print(f"\n=== Rangos medios ({metric}) ===")
    print(average_rankings.round(4).sort_values())

    # Generación de gráfico: barras de rangos medios
    plt.figure(figsize=(12, 7))
    sns.barplot(x=average_rankings.index, y=average_rankings.values, hue=average_rankings.index, palette='viridis', legend=False)
    plt.ylabel("Rango medio (Friedman)")
    plt.xlabel("Modelos")
    plt.title(f"Comparación de modelos - {metric.capitalize()} ({framework_display})")
    plt.ylim(1, len(average_rankings) + 0.5)
    plt.xticks(rotation=60, ha='right') 
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{framework_prefix}_average_ranks_{metric.lower()}.png"
        plt.savefig(os.path.join(output_dir, filename))
    else:
        plt.show()

    # --------------------------
    # 2. TEST DE IMAN-DAVENPORT
    # --------------------------
    N = df_results.shape[0]		# Número de datasets (sujetos)
    k = df_results.shape[1]		# Número de modelos (tratamientos)

	# Cálculo matemático del estadístico corregido
    F_id = ((N - 1) * statistic) / (N * (k - 1) - statistic)
    # Cálculo del p-valor usando la distribución F
    p_id = 1 - f.cdf(F_id, k - 1, (k - 1) * (N - 1))

    print(f"\n=== Test de Iman-Davenport ({metric}) ===")
    print(f"F estadístico: {F_id:.4f}")
    print(f"p-valor: {p_id:.6f}")

    if p_id < 0.05:
        print("--> Existen diferencias significativas entre los modelos (Rechazamos H0).")
    else:
        print("--> No se detectan diferencias significativas (Mantenemos H0).")

    # ----------------------------
    # 3. TEST POST-HOC DE NEMENYI
    # ----------------------------
    # Solo se procede con el análisis si Friedman indicó diferencias globales
    if p_value < 0.05:
        print(f"\n=== Test de Nemenyi ({metric}) ===")
        rank_dict = average_rankings.to_dict()
        
        # statds dibuja automáticamente el diagrama de distancia crítica
        rank_values, critical_distance_nemenyi, fig = nemenyi(
            rank_dict, N, alpha=0.05, verbose=True
        )
        
        if fig:
            fig.suptitle(f'Test de Nemenyi - {metric.capitalize()} ({framework_display})', y=1.05)
            
            if output_dir:
                filename = f"{framework_prefix}_nemenyi_{metric.lower()}.png"
                fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            else:
                plt.show()
    else:
        print("\nNo se realiza el test de Nemenyi porque Friedman no ha detectado diferencias significativas.")


# ==============================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================
if __name__ == "__main__":
    # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Ejecuta tests estadísticos (Friedman, Iman-Davenport, Nemenyi) sobre resultados.")
    parser.add_argument(
        '-t', '--task',
        type=str,
        required=True,
        choices=['classification', 'regression'],
        help="Tarea a analizar: 'classification' o 'regression'")
    parser.add_argument(
        '-m', '--metric',
        type=str,
        required=True,
        help="Métrica a evaluar ('accuracy', 'precision', 'recall', 'f1', 'mae', 'rmse', 'r2', 'train', 'inference', 'size')")
    
    args = parser.parse_args()
    TASK = args.task.lower()
    METRIC = args.metric.lower()
    
    # Validación de métricas por tarea
    valid_metrics = {
        "classification": ["accuracy", "precision", "recall", "f1", "train", "inference", "size"],
        "regression": ["mae", "rmse", "r2", "train", "inference", "size"]
    }
    
    if METRIC not in valid_metrics[TASK]:
        print(f"¡Error! La métrica '{METRIC}' no es válida para la tarea '{TASK}'.")
        print(f"Opciones válidas para {TASK}: {', '.join(valid_metrics[TASK])}")
        exit(1)
    
    # Configuración de rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.abspath(os.path.join(script_dir, "../../results2"))
    OUTPUT_GRAPHS_DIR = os.path.join(RESULTS_DIR, f"graphs/tests/{TASK}", METRIC)
    
    print(f"Iniciando análisis para TAREA: {TASK.upper()} | MÉTRICA: {METRIC.upper()}")
    print(f"Buscando archivos en: {RESULTS_DIR}")
    
    # --------------------------------
    # FASE 1: Modelos de scikit-learn
    # --------------------------------
    print("\n\n" + "*"*60)
    print(f" FASE 1: COMPARATIVA SCIKIT-LEARN ({TASK.capitalize()}) ")
    print("*"*60)
    
    # Buscar todos los archivos que empiecen por 'sklearn_' y termine en '_{TASK}_{METRIC}.csv'
    sklearn_files = glob.glob(os.path.join(RESULTS_DIR, f"sklearn_{TASK}_{METRIC}.csv"))
    
    if sklearn_files:
        df_sklearn = load_and_merge_csvs(sklearn_files, task=TASK)
        run_statistical_analysis(df_sklearn, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="sklearn")
    else:
        print(f"Para probar la Fase 1, asegúrate de tener el archivo base de sklearn.")
        
    # ---------------------------------------
    # FASE 2: Modelos de scikit-learn + ONNX
    # ---------------------------------------
    print("\n\n" + "*"*60)
    print(f" FASE 2: COMPARATIVA ONNX ({TASK.capitalize()}) ")
    print("*"*60)
    
    # Buscar todos los archivos que empiecen por 'onnx_' y termine en '_{TASK}_{METRIC}.csv'
    onnx_files = glob.glob(os.path.join(RESULTS_DIR, f"onnx_*{TASK}_{METRIC}.csv"))
    
    if onnx_files:
        df_onnx = load_and_merge_csvs(onnx_files, task=TASK)
        run_statistical_analysis(df_onnx, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="onnx")
    else:
        print(f"Para probar la Fase 2, asegúrate de tener archivos ONNX en {RESULTS_DIR}")

    # -------------------------------------------------------------
    # FASE 3: Modelos MLP (scikit-learn, ONNX, PyTorch, TensorFlow)
    # -------------------------------------------------------------
    print("\n\n" + "*"*60)
    print(f" FASE 3: COMPARATIVA ESPECÍFICA TODOS LOS MLPs ({TASK.capitalize()}) ")
    print("*"*60)
    
    # Buscar todos los archivos que termine en '_{TASK}_{METRIC}.csv'
    all_files_to_merge = glob.glob(os.path.join(RESULTS_DIR, f"*_{TASK}_{METRIC}.csv"))
    
    if all_files_to_merge:
        # Extraemos el subconjunto de columnas referidas a MLP
        df_all_mlps = load_and_merge_csvs(all_files_to_merge, task=TASK, filter_keyword="MLP")
        
        if not df_all_mlps.empty:
            run_statistical_analysis(df_all_mlps, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="mlp")
        else:
            print("Se han encontrado archivos, pero ninguna columna contenía el modelo 'MLP'.")
    else:
        print("No se han encontrado datos para la Fase 3. Verifique las rutas de los archivos.")

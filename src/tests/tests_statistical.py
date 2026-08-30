import argparse
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy.stats import f
from statds.no_parametrics import friedman, nemenyi


# ===========================================
# CATÁLOGOS Y MAPEOS COMPARTIDOS
# ===========================================
# Diccionario de traducción para formatear el nombre de los frameworks en los gráficos.
# Se usa tanto en load_and_merge_csvs (formato ancho) como en la Fase 2b (formato long).
FRAMEWORK_NAMES = {
    "sklearn": "scikit-learn",
    "onnx": "ONNX",
    "pytorch": "PyTorch",
    "tflite": "TFLite"
}

# Diccionario para formatear los nombres técnicos de las configuraciones a nombres legibles.
CONFIG_NAMES = {
    "static": "Quant.",
    "distill": "KD",
    "distill_static": "KD + Quant.",
    "distill_t1": "KD T1",
    "distill_t2": "KD T2",
    "distill_t1_static": "KD T1 + Quant.",
    "distill_t2_static": "KD T2 + Quant."
}

# Catálogo de algoritmos base por tarea, usado para identificar a qué algoritmo
# pertenece una columna independientemente del framework/sufijo que la acompañe
# (necesario para la Fase 2b, que compara sklearn y ONNX algoritmo a algoritmo).
CLASSIFICATION_ALGOS = ["DecisionTree", "KNN", "LinearSVC", "MLP", "RandomForest"]
REGRESSION_ALGOS = ["DecisionTree", "KNN", "LinearRegression", "MLP", "RandomForest"]


# ===========================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO (ETL)
# ===========================================
def parse_filename_metadata(basename, task):
    """
    Extrae el framework y la configuración (cuantización/destilación) codificados
    en el nombre de un fichero de resultados.

    Ejemplo: 'onnx_static_classification_f1.csv' (task='classification')
             -> framework='onnx', config='static'

    :param basename: Nombre del fichero (sin ruta).
    :param task: 'classification' o 'regression'.
    :returns: Tupla (framework, config). config es '' si el fichero no tiene
              configuración explícita (caso de sklearn).
    """
    framework = "Unknown"
    config = ""
    
    # Pivote para partir el nombre del archivo
    split_key = f"_{task}_"

    if split_key in basename:
        left_part = basename.split(split_key)[0]
        # Corta solo en el primer guion bajo para separar framework de la configuración
        # Ej: 'onnx_static' -> ['onnx', 'static']
        parts = left_part.split("_", 1)
        framework = parts[0].lower()
        if len(parts) > 1:
            config = parts[1].lower()

    return framework, config


def extract_algorithm(col_name, task):
    """
    Identifica el algoritmo base (DecisionTree, KNN, LinearSVC/LinearRegression, MLP,
    RandomForest) a partir del nombre "crudo" de una columna del CSV, ignorando los
    sufijos de framework (_Clas, _Reg, _ONNX, _PyTorch, _TFLite, etc.) que la acompañen.

    Empareja por PREFIJO porque es el único fragmento estable observado en todos los
    frameworks (p.ej. 'KNN_ONN_ClasX' conserva el prefijo 'KNN' aunque el resto del
    nombre esté corrompido).

    :param col_name: Nombre de columna tal como aparece en el CSV original.
    :param task: 'classification' o 'regression', para elegir el catálogo de algoritmos.
    :returns: Nombre canónico del algoritmo, o el nombre original si no se reconoce
              ninguno (se emite un aviso por consola para poder revisarlo).
    """
    # Se ordena por longitud descendente para evitar que coincidencias parciales fallen
    # (por ejemplo, evitar que "Random" haga match antes que "RandomForest").
    candidates = REGRESSION_ALGOS if task == "regression" else CLASSIFICATION_ALGOS
    for algo in sorted(candidates, key=len, reverse=True):
        if col_name.lower().startswith(algo.lower()):
            return algo

    print(f"  [Aviso] No se ha podido identificar el algoritmo de la columna '{col_name}'; se mantiene tal cual.")
    return col_name


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
    df_merged = pd.DataFrame()

    for path in filepaths:
        if not os.path.exists(path):
            print(f"Advertencia: No se ha encontrado {path}")
            continue

        # 1. Extracción de metadatos desde el nombre del archivo (framework y configuración)
        basename = os.path.basename(path)
        framework, config = parse_filename_metadata(basename, task)

        # 2. Carga del CSV estableciendo 'Dataset' como índice para facilitar cruces
        df_temp = pd.read_csv(path).set_index('Dataset')
        
        # 3. Construcción del sufijo identificativo para las columnas
        fw_pretty = FRAMEWORK_NAMES.get(framework, framework.capitalize())
        if config:
            conf_pretty = CONFIG_NAMES.get(config, config.replace('_', ' ').title())
            suffix = f"{fw_pretty} | {conf_pretty}"
        else:
            suffix = fw_pretty
            
        # 4. Aplicación del sufijo a los nombres de las columnas
        # Ej: "MLP_Clas" -> "MLP_Clas (PyTorch | KD T1 + Quant.)"
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


def build_long_format_by_algorithm(file_label_pairs, task):
    """
    Construye una tabla en formato 'long' con filas = Dataset x Algoritmo y columnas =
    las etiquetas indicadas en file_label_pairs, para poder comparar directamente
    frameworks/configuraciones que en su forma 'ancha' natural NO son comparables
    columna a columna (porque cada columna es un algoritmo distinto).

    Caso de uso (Fase 2b): comparar scikit-learn contra ONNX de forma directa,
    controlando el algoritmo -- DecisionTree_sklearn se compara contra
    DecisionTree_ONNX, nunca contra KNN_ONNX -- en vez de mezclar los 5 algoritmos de
    cada framework en un único test heterogéneo de 10-15 columnas.

    Solo se conservan las combinaciones (Dataset, Algoritmo) presentes en TODAS las
    etiquetas (filas completas): Friedman exige el mismo conjunto de sujetos en todos
    los tratamientos.

    :param file_label_pairs: Lista de tuplas (ruta_csv, etiqueta_de_columna). El orden
                              de la lista determina el orden de las columnas resultantes.
    :param task: 'classification' o 'regression'.
    :returns: DataFrame indexado por 'Dataset' (identificador combinado
              'Dataset | Algoritmo'), con una columna por etiqueta. Vacío si no hay
              datos suficientes.
    """
    long_frames = []
    label_order = []

    for path, label in file_label_pairs:
        if not os.path.exists(path):
            print(f"Advertencia: No se ha encontrado {path}")
            continue

        df_temp = pd.read_csv(path)
        
        # Transformamos la tabla ancha a formato largo (unificando métricas bajo 'Value')
        df_melted = df_temp.melt(id_vars='Dataset', var_name='RawColumn', value_name='Value')
        
        # Extraemos el algoritmo base para estandarizar las filas
        df_melted['Algoritmo'] = df_melted['RawColumn'].apply(lambda c: extract_algorithm(c, task))
        df_melted['Label'] = label

        long_frames.append(df_melted[['Dataset', 'Algoritmo', 'Label', 'Value']])
        if label not in label_order:
            label_order.append(label)

    if not long_frames:
        return pd.DataFrame()

    # Concatenamos todos los dataframes largos en uno solo
    df_long = pd.concat(long_frames, ignore_index=True)
    
    # Si (Dataset, Algoritmo, Label) apareciera duplicado, promediamos para no romper el pivot
    # Esto actúa como medida de seguridad para evitar errores de índices duplicados.
    df_long = df_long.groupby(['Dataset', 'Algoritmo', 'Label'], as_index=False)['Value'].mean()

    # Pivotamos de vuelta para que cada 'Label' (configuración) sea una columna, y las filas sean la 
    # combinación de Dataset + Algoritmo
    df_pivot = df_long.pivot(index=['Dataset', 'Algoritmo'], columns='Label', values='Value')

    # Descartamos las filas (Dataset x Algoritmo) que no tengan datos en TODAS las configuraciones, ya que
    # Friedman requiere medidas repetidas idénticas para todos los sujetos.
    n_before = df_pivot.shape[0]
    df_pivot = df_pivot.dropna()
    n_after = df_pivot.shape[0]
    if n_after < n_before:
        print(f"  [Info] Se han descartado {n_before - n_after} combinaciones Dataset x Algoritmo "
              f"sin datos en todas las columnas.")

    if df_pivot.empty:
        return df_pivot

    # Reordenamos las columnas según el orden original de llegada (por defecto pivot ordena alfabéticamente)
    df_pivot = df_pivot[[c for c in label_order if c in df_pivot.columns]]

    # Combinamos el índice jerárquico ('Dataset', 'Algoritmo') en un único string.
    # Esto permite reutilizar la función run_statistical_analysis sin alterar su lógica interna.
    df_pivot.index = df_pivot.index.map(lambda idx: f"{idx[0]} | {idx[1]}")
    df_pivot.index.name = 'Dataset'

    return df_pivot


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
        "mlp": "MLP",
        "sklearn_vs_onnx": "scikit-learn vs ONNX",
        "phase3_quant": "Cuantización",
        "phase4_distill": "Destilación de conocimiento",
        "phase5_pytorch": "Interacción PT (Quant + KD)",
        "phase6_tensorflow": "Interacción TF (Quant + KD)"
    }
    framework_display = display_names.get(framework.lower(), framework.capitalize())
    framework_prefix = framework.lower()

    print(f"\n" + "="*50)
    print(f" ANALIZANDO MÉTRICA: {metric.upper()} | TAREA: {task.upper()} ({framework_display})")
    print("="*50)

    # Comprobación de seguridad: con la Fase 2b es posible que, si faltan ficheros o
    # no hay combinaciones Dataset x Algoritmo completas, no queden datos suficientes.
    if df_results.empty or df_results.shape[1] < 2:
        print(f"--> Datos insuficientes para '{framework_display}' (se necesitan al menos 2 columnas). Se omite.")
        return
    if df_results.shape[0] < 2:
        print(f"--> Datos insuficientes para '{framework_display}' (se necesitan al menos 2 sujetos/filas). Se omite.")
        return

    print(df_results.round(4))
    
    # Determinar dirección de optimalidad para asignar los rangos matemáticos correctamente.
    # Si es True (ej: RMSE), el valor más bajo recibe el mejor rango (1). Si es False (ej: F1),
    # el valor más alto recibe rango 1.
    is_minimize = determine_minimization(metric)
    print(f"\n--> Configuración de Friedman: minimize={is_minimize}")
    
    # Resetear índice para asegurar la compatibilidad con el formato esperado por statds
    results_table_sd = df_results.reset_index()
    
    # --------------------
    # 1. TEST DE FRIEDMAN
    # --------------------
    # alpha=0.05 establece el nivel de significancia (95% confianza)
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

    # Generación de gráfico: barras de rangos medios de Friedman
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
    plt.close()

    # --------------------------
    # 2. TEST DE IMAN-DAVENPORT
    # --------------------------
    # Este test es una corrección (menos conservadora) del test de Friedman.
    N = df_results.shape[0]		# Número de datasets (sujetos)
    k = df_results.shape[1]		# Número de modelos (tratamientos)

	# Cálculo matemático del estadístico corregido de Iman-Davenport
    F_id = ((N - 1) * statistic) / (N * (k - 1) - statistic)
    
    # Cálculo del p-valor calculando la probabilidad en la cola derecha de la distribución F
    # Grados de libertad: k-1 y (k-1)*(N-1)
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
    # Solo se procede con el análisis si Friedman indicó diferencias globales.
    # Nemenyi nos dirá *entre qué pares específicos* existen estas diferencias.
    if p_value < 0.05:
        print(f"\n=== Test de Nemenyi ({metric}) ===")
        rank_dict = average_rankings.to_dict()
        
        # statds dibuja automáticamente el diagrama de distancia crítica
        # Modelos conectados por la misma línea gruesa NO tienen diferencias significativas entre sí.
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
    
    # Validación de métricas: evita ejecutar el script si se pide una métrica de clasificación para regresión, etc.
    valid_metrics = {
        "classification": ["accuracy", "precision", "recall", "f1", "train", "inference", "size"],
        "regression": ["mae", "rmse", "r2", "train", "inference", "size"]
    }
    
    if METRIC not in valid_metrics[TASK]:
        print(f"¡Error! La métrica '{METRIC}' no es válida para la tarea '{TASK}'.")
        print(f"Opciones válidas para {TASK}: {', '.join(valid_metrics[TASK])}")
        exit(1)
    
    # Gestión de rutas de archivos relativas y directorios de salida
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.abspath(os.path.join(script_dir, "../../results2"))
    OUTPUT_GRAPHS_DIR = os.path.join(RESULTS_DIR, f"graphs/tests3/{TASK}", METRIC)
    
    print(f"Iniciando análisis para TAREA: {TASK.upper()} | MÉTRICA: {METRIC.upper()}")
    print(f"Buscando archivos en: {RESULTS_DIR}")
    
    # --------------------------------
    # FASE 1: Modelos de scikit-learn
    # --------------------------------
    # Objetivo: Establecer la base de referencia (baseline) y evaluar si existen
    # diferencias significativas entre los algoritmos originales.
    print("\n\n" + "*"*60)
    print(f" FASE 1: COMPARATIVA SCIKIT-LEARN ({TASK.capitalize()}) ")
    print("*"*60)
    
    # Buscar todos los archivos que empiecen por 'sklearn_' y termine en '_{TASK}_{METRIC}.csv'
    sklearn_file_path = os.path.join(RESULTS_DIR, f"sklearn_{TASK}_{METRIC}.csv")
    sklearn_files = [sklearn_file_path] if os.path.exists(sklearn_file_path) else []
    
    if sklearn_files:
        df_sklearn = load_and_merge_csvs(sklearn_files, task=TASK)
        run_statistical_analysis(df_sklearn, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="sklearn")
    else:
        print(f"Para probar la Fase 1, asegúrese de tener el archivo base de sklearn.")
        
    # ---------------------
    # FASE 2: Modelos ONNX
    # ---------------------
    # Objetivo: Evaluar si el comportamiento intra-framework cambia una vez que los modelos
    # se han exportado y transformado a ONNX.
    print("\n\n" + "*"*60)
    print(f" FASE 2: COMPARATIVA ONNX ({TASK.capitalize()}) ")
    print("*"*60)
    
    # Buscar todos los archivos que empiecen por 'onnx_' y termine en '_{TASK}_{METRIC}.csv'
    onnx_files = glob.glob(os.path.join(RESULTS_DIR, f"onnx_*{TASK}_{METRIC}.csv"))
    
    if onnx_files:
        df_onnx = load_and_merge_csvs(onnx_files, task=TASK)
        run_statistical_analysis(df_onnx, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="onnx")
    else:
        print(f"Para probar la Fase 2, asegúrese de tener archivos ONNX en {RESULTS_DIR}")

    # -----------------------------------------------------------
    # FASE 2b: SCIKIT-LEARN vs ONNX, comparación DIRECTA por algoritmo
    # -----------------------------------------------------------
    # Objetivo: A diferencia de las Fases 1 y 2 (que comparan los algoritmos dentro de cada framework),
    # aquí comparamos cada algoritmo de sklearn contra su propia versión en ONNX. Usamos el formato
    # 'long' (Dataset x Algoritmo) para evitar comparar peras con manzanas (ej: DecisionTree sklearn VS KNN ONNX).
    print("\n\n" + "*"*60)
    print(f" FASE 2b: SCIKIT-LEARN vs ONNX, POR ALGORITMO ({TASK.capitalize()}) ")
    print("*"*60)

    sklearn_vs_onnx_pairs = []

    if sklearn_files:
        sklearn_vs_onnx_pairs.append((sklearn_files[0], "scikit-learn"))

    for onnx_path in sorted(onnx_files):
        onnx_basename = os.path.basename(onnx_path)
        onnx_framework, onnx_config = parse_filename_metadata(onnx_basename, TASK)
        onnx_fw_pretty = FRAMEWORK_NAMES.get(onnx_framework, onnx_framework.capitalize())
        
        # Etiquetar de forma bonita si aplica configuración extra (ej: Quant.)
        if onnx_config:
            onnx_conf_pretty = CONFIG_NAMES.get(onnx_config, onnx_config.replace('_', ' ').title())
            onnx_label = f"{onnx_fw_pretty} | {onnx_conf_pretty}"
        else:
            onnx_label = onnx_fw_pretty
        sklearn_vs_onnx_pairs.append((onnx_path, onnx_label))

    if len(sklearn_vs_onnx_pairs) >= 2:
        df_sklearn_vs_onnx = build_long_format_by_algorithm(sklearn_vs_onnx_pairs, task=TASK)
        run_statistical_analysis(df_sklearn_vs_onnx, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR,
                                  framework="sklearn_vs_onnx")
    else:
        print("Para la Fase 2b hacen falta el fichero de sklearn y al menos un fichero de ONNX.")
        
    # -----------------------------------------------------------
    # FASE 3: Frameworks de despliegue (SOLO CUANTIZACIÓN)
    # -----------------------------------------------------------
    # Objetivo: Evaluar y comparar exclusivamente el impacto de la cuantización en los diferentes
    # frameworks frente al baseline de sklearn.
    print("\n\n" + "*"*60)
    print(f" FASE 3: FRAMEWORKS DE DESPLIEGUE (SOLO CUANTIZACIÓN) ({TASK.capitalize()}) ")
    print("*"*60)
    pairs_phase3 = []
    onnx_static_path = os.path.join(RESULTS_DIR, f"onnx_static_{TASK}_{METRIC}.csv")
    pt_static_path = os.path.join(RESULTS_DIR, f"pytorch_static_{TASK}_{METRIC}.csv")
    tf_static_path = os.path.join(RESULTS_DIR, f"tflite_static_{TASK}_{METRIC}.csv")

    # Anexar sólo los archivos que existan físicamente en el directorio
    if os.path.exists(sklearn_file_path): pairs_phase3.append((sklearn_file_path, "scikit-learn (Ref)"))
    if os.path.exists(onnx_static_path): pairs_phase3.append((onnx_static_path, "ONNX Quant."))
    if os.path.exists(pt_static_path): pairs_phase3.append((pt_static_path, "PT Quant."))
    if os.path.exists(tf_static_path): pairs_phase3.append((tf_static_path, "TF Quant."))

    if len(pairs_phase3) >= 2:
        df_p3 = build_long_format_by_algorithm(pairs_phase3, task=TASK)
        run_statistical_analysis(df_p3, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="phase3_quant")
    else:
        print("Para probar la Fase 3, asegúrese de tener el fichero de sklearn y al menos un fichero de cuantización estática.")
        
    # -----------------------------------------------------------
    # FASE 4: Frameworks de despliegue (SOLO DESTILACIÓN)
    # -----------------------------------------------------------
    # Objetivo: Analizar cómo afecta la destilación de conocimiento aislada.
    # Se manejan diferentes sufijos según la temperatura (T1/T2)
    print("\n\n" + "*"*60)
    print(f" FASE 4: FRAMEWORKS DE DESPLIEGUE (SOLO DESTILACIÓN) ({TASK.capitalize()}) ")
    print("*"*60)
    pairs_phase4 = []
    if os.path.exists(sklearn_file_path): pairs_phase4.append((sklearn_file_path, "scikit-learn (Ref)"))
    
    # Definición de las configuraciones de KD según la tarea
    if TASK == 'classification':
        configs_to_check = [
            ("pytorch_distill_t1", "PT KD T1"), ("pytorch_distill_t2", "PT KD T2"),
            ("tflite_distill_t1", "TF KD T1"), ("tflite_distill_t2", "TF KD T2")
        ]
    else:
        configs_to_check = [
            ("pytorch_distill", "PT KD"), ("tflite_distill", "TF KD"),
            ("pytorch_distill_t1", "PT KD"), ("tflite_distill_t1", "TF KD")
        ]
        
    for prefix, label in configs_to_check:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{TASK}_{METRIC}.csv")
        # Se comprueba también que la etiqueta no se haya añadido ya
        if os.path.exists(path) and label not in [l for _, l in pairs_phase4]:
            pairs_phase4.append((path, label))
            
    if len(pairs_phase4) >= 2:
        df_p4 = build_long_format_by_algorithm(pairs_phase4, task=TASK)
        run_statistical_analysis(df_p4, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="phase4_distill")
    else:
        print("Para probar la Fase 4, asegúrese de tener el fichero de sklearn y al menos un fichero de destilación.")
        
    # -----------------------------------------------------------
    # FASE 5: INTERACCIÓN CUANTIZACIÓN + DESTILACIÓN (PYTORCH)
    # -----------------------------------------------------------
    # Objetivo: Evaluar la interacción entre técnicas (KD y cuantización juntas) operando
    # exclusivamente dentro del ecosistema de PyTorch.
    print("\n\n" + "*"*60)
    print(f" FASE 5: INTERACCIÓN CUANTIZACIÓN + DESTILACIÓN (PYTORCH) ({TASK.capitalize()}) ")
    print("*"*60)
    pairs_phase5 = []
    if os.path.exists(sklearn_file_path): pairs_phase5.append((sklearn_file_path, "scikit-learn (Base)"))
    
    pt_files = glob.glob(os.path.join(RESULTS_DIR, f"pytorch_*_{TASK}_{METRIC}.csv"))
    for path in sorted(pt_files):
        _, conf = parse_filename_metadata(os.path.basename(path), TASK)
        label = f"PT {CONFIG_NAMES.get(conf, conf)}"
        pairs_phase5.append((path, label))
        
    if len(pairs_phase5) >= 2:
        df_p5 = build_long_format_by_algorithm(pairs_phase5, task=TASK)
        run_statistical_analysis(df_p5, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="phase5_pytorch")
    else:
        print("Para probar la Fase 5, asegúrese de tener el fichero de sklearn y ficheros de resultados de PyTorch.")

    # -----------------------------------------------------------
    # FASE 6: INTERACCIÓN CUANTIZACIÓN + DESTILACIÓN (TENSORFLOW)
    # -----------------------------------------------------------
    # Objetivo: Lo mismo que la Fase 5, pero enfocado en TensorFlow Lite.
    print("\n\n" + "*"*60)
    print(f" FASE 6: INTERACCIÓN CUANTIZACIÓN + DESTILACIÓN (TENSORFLOW) ({TASK.capitalize()}) ")
    print("*"*60)
    pairs_phase6 = []
    if os.path.exists(sklearn_file_path): pairs_phase6.append((sklearn_file_path, "scikit-learn (Base)"))
    
    tf_files = glob.glob(os.path.join(RESULTS_DIR, f"tflite_*_{TASK}_{METRIC}.csv"))
    for path in sorted(tf_files):
        _, conf = parse_filename_metadata(os.path.basename(path), TASK)
        label = f"TF {CONFIG_NAMES.get(conf, conf)}"
        pairs_phase6.append((path, label))
        
    if len(pairs_phase6) >= 2:
        df_p6 = build_long_format_by_algorithm(pairs_phase6, task=TASK)
        run_statistical_analysis(df_p6, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="phase6_tensorflow")
    else:
        print("Para probar la Fase 6, asegúrese de tener el fichero de sklearn y ficheros de resultados de TensorFlow Lite.")

    # -------------------------------------------------------------
    # FASE 7: Modelos MLP (scikit-learn, ONNX, PyTorch, TensorFlow)
    # -------------------------------------------------------------
    # Objetivo: Aislamiento del MLP.
    print("\n\n" + "*"*60)
    print(f" FASE 7: COMPARATIVA DE MLP ({TASK.capitalize()}) ")
    print("*"*60)
    
    # Buscar todos los archivos que termine en '_{TASK}_{METRIC}.csv'
    all_files_to_merge = glob.glob(os.path.join(RESULTS_DIR, f"*_{TASK}_{METRIC}.csv"))
    
    if all_files_to_merge:
        # Extraemos dinámicamente el subconjunto de columnas referidas a MLP
        df_all_mlps = load_and_merge_csvs(all_files_to_merge, task=TASK, filter_keyword="MLP")
        
        if not df_all_mlps.empty:
            run_statistical_analysis(df_all_mlps, metric=METRIC, task=TASK, output_dir=OUTPUT_GRAPHS_DIR, framework="mlp")
        else:
            print("Se han encontrado archivos, pero ninguna columna contenía el modelo 'MLP'.")
    else:
        print("No se han encontrado datos para la Fase 7. Verifique las rutas de los archivos.")

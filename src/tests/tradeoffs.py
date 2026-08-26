import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# ===================
# METADATOS Y MAPEOS
# ===================
# Diccionario con la información de la topología de cada dataset
DATASETS_INFO = {
    "classification": {
        "Car Evaluation": {"instances": 1728, "features": 6},
        "Human Activity Recognition Using Smartphones Dataset (HAR)": {"instances": 10299, "features": 561},
        "Iris Plants Database": {"instances": 150, "features": 4},
        "Large Soybean Database": {"instances": 307, "features": 35},
        "Letter Image Recognition Data": {"instances": 20000, "features": 16},
        "MAGIC gamma telescope data 2004": {"instances": 19020, "features": 10},
        "MNIST": {"instances": 10000, "features": 784},
        "Pima Indians Diabetes Database": {"instances": 768, "features": 8},
        "Predict Students' Dropout and Academic Success": {"instances": 4424, "features": 36},
        "Wine recognition": {"instances": 178, "features": 13}
    },
    "regression": {
        "1985 Auto Imports Database": {"instances": 205, "features": 26},
        "Abalone data": {"instances": 4177, "features": 8},
        "Air Quality": {"instances": 9358, "features": 15},
        "Appliances Energy Prediction": {"instances": 19735, "features": 28},
        "Bike Sharing": {"instances": 17379, "features": 13},
        "Communities and Crime": {"instances": 1994, "features": 128},
        "Obesity Levels": {"instances": 2111, "features": 16},
        "Real Estate Valuation": {"instances": 414, "features": 6},
        "Student Performance": {"instances": 395, "features": 32},
        "Wine Quality": {"instances": 4898, "features": 11}
    }
}

# Mapa de traducción para alinear los nombres abreviados de los CSV con los nombres reales de DATASETS_INFO
CSV_TO_DATASET_MAP = {
    "CAR": "Car Evaluation",
    "DIABETES": "Pima Indians Diabetes Database",
    "GAMMA": "MAGIC gamma telescope data 2004",
    "HAR": "Human Activity Recognition Using Smartphones Dataset (HAR)",
    "IRIS": "Iris Plants Database",
    "LETTER RECOGNITION": "Letter Image Recognition Data",
    "MNIST": "MNIST",
    "SOYBEAN": "Large Soybean Database",
    "STUDENT DROPOUT": "Predict Students' Dropout and Academic Success",
    "WINE": "Wine recognition",

    "ABALONE": "1985 Auto Imports Database",
    "AIR QUALITY": "Air Quality",
    "APPLIANCES ENERGY PREDICTION": "Appliances Energy Prediction",
    "AUTOMOBILE": "1985 Auto Imports Database",
    "BIKE SHARING HOURLY": "Bike Sharing",
    "COMMUNITIES AND CRIME": "Communities and Crime",
    "OBESITY LEVELS": "Obesity Levels",
    "REAL ESTATE VALUATION": "Real Estate Valuation",
    "STUDENT PERFORMANCE": "Student Performance",
    "WINE QUALITY": "Wine Quality"
}


# =========================================
# FUNCIONES DE EXTRACCIÓN Y TRANSFORMACIÓN
# =========================================
def extract_model_details(col_name):
    """
    Infiere el framework utilizado analizando el nombre del modelo/columna.
    
    :param col_name: Nombre del modelo extraído del CSV.
    
    :returns: Nombre del framework (ONNX, TFLite, PyTorch o scikit-learn).
    """
    col_lower = col_name.lower()
    if 'onnx' in col_lower: 
        return 'ONNX'
    elif 'tflite' in col_lower: 
        return 'TFLite'
    elif 'pytorch' in col_lower: 
        return 'PyTorch'
    else: 
        return 'scikit-learn'

def load_and_melt_metric(results_dir, task, metric_keyword):
    """
    Busca, carga y transforma todos los CSV correspondientes a una métrica y tarea.
    Pasa los datos de un formato ancho (modelos por columnas) a formato largo (melted).
    
    :param results_dir: Ruta de la carpeta raíz que contiene los CSV.
    :param task: Tarea a evaluar ('classification' o 'regression').
    :param metric_keyword: Métrica a extraer (ej. 'r2', 'inference', 'size').
        
    :returns: DataFrame unificado y en formato largo sin duplicados.
    """
    # Buscar todos los archivos que coincidan con el patrón deseado
    files = glob.glob(os.path.join(results_dir, f"*_{task}_{metric_keyword}.csv"))
    if not files: 
        return pd.DataFrame()

    df_list = []
    for f in files:
        df_temp = pd.read_csv(f)
        # Transformar a formato tabular largo: Dataset | Model | Value
        df_melted = df_temp.melt(id_vars='Dataset', var_name='Model', value_name=metric_keyword.upper())

        # Extraer el nombre de la configuración desde el nombre del archivo y añadirlo al nombre del modelo
        filename = os.path.basename(f)
        config_name = filename.replace(f"_{task}_{metric_keyword}.csv", "")
        df_melted['Model'] = df_melted['Model'] + " (" + config_name + ")"

        df_list.append(df_melted)
        
    # Concatenar todos los DataFrames y limpiar duplicados
    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['Dataset', 'Model'])
    return df_final

def get_pareto_front(df, x_col, y_col, maximize_x=False, maximize_y=True):
    """
    Calcula el frente de Pareto matemático para encontrar los modelos que ofrecen 
    el mejor compromiso entre dos métricas (menor tamaño y mayor rendimiento).
    
    :param df: DataFrame con los datos a evaluar.
    :param x_col: Nombre de la columna en el eje X (ej. Tamaño).
    :param y_col: Nombre de la columna en el eje Y (ej. Rendimiento).
    :param maximize_x: True si se busca un valor mayor en X, False para buscar menor.
    :param maximize_y: True si se busca un valor mayor en Y, False para buscar menor.
        
    :returns: Subconjunto de datos que forman el frente óptimo.
    """
    # Ordenar por el eje X de forma ascendente (buscando minimizar por defecto)
    sorted_df = df.sort_values(by=x_col, ascending=not maximize_x)
    pareto_front = []
    # Inicializar el mejor valor Y con infinito o menos infinito
    best_y = -np.inf if maximize_y else np.inf

    for _, row in sorted_df.iterrows():
        # Comprobar si el modelo actual mejora al mejor visto hasta ahora
        is_better = row[y_col] > best_y if maximize_y else row[y_col] < best_y
        if is_better:
            pareto_front.append(row)
            best_y = row[y_col]

    return pd.DataFrame(pareto_front)


# ===========================
# FUNCIONES DE VISUALIZACIÓN
# ===========================
def plot_pareto_bubble(df_avg, metric_perf, metric_time, metric_size, output_dir, task):
    """
    Genera un gráfico estático de burbujas mostrando el trade-off global de todos los modelos.
    El tamaño de la burbuja representa la latencia.
    
    :param df_avg: DataFrame con la media agrupada de los modelos.
    :param metric_perf: Nombre de la métrica de rendimiento.
    :param metric_time: Nombre de la métrica de latencia.
    :param metric_size: Nombre de la métrica de tamaño.
    :param output_dir: Directorio de guardado.
    :param task: Tipo de tarea ('classification' o 'regression').
    """
    plt.figure(figsize=(14, 8))
    col_perf = metric_perf.upper()
    col_time = metric_time.upper()
    col_size = metric_size.upper()

    # Scatter plot base
    sns.scatterplot(
        data=df_avg, x=col_size, y=col_perf, size=col_time,
        hue='Framework', sizes=(20, 800), alpha=0.6, palette='Set1', edgecolor="black"
    )

    # Añadir la línea del frente de Pareto
    pareto_df = get_pareto_front(df_avg, x_col=col_size, y_col=col_perf)
    plt.plot(pareto_df[col_size], pareto_df[col_perf],
             color='red', linestyle='--', linewidth=2, label='Frente de Pareto', zorder=1)

    # Ajuste dinámico de escala solo para el eje X (tamaño)
    plt.xscale('log')
    if df_avg[col_time].max() - df_avg[col_time].min() > 1000:
            plt.yscale('log')
    
    # Forzar límite inferior del eje Y a 0 para omitir outliers negativos en regresón
    if task == 'regression':
        plt.ylim(bottom=0)    

    plt.title("Trade-off global: Tamaño vs Rendimiento vs Latencia", fontsize=16)
    plt.xlabel("Tamaño medio del modelo (KB)", fontsize=12)
    plt.ylabel(f"Rendimiento medio ({col_perf})", fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_tradeoff_pareto.png"), dpi=300)
    plt.close()

def plot_pareto_zoomed(df_avg, metric_perf, metric_time, metric_size, output_dir, task):
    """
    Genera un gráfico estático que hace zoom sobre la zona del frente de Pareto.
    Filtra los modelos que están lejos del óptimo para mejorar la legibilidad.
    
    :param df_avg: DataFrame con la media agrupada.
    :param metric_perf: Nombre de la métrica de rendimiento.
    :param metric_time: Nombre de la métrica de latencia.
    :param metric_size: Nombre de la métrica de tamaño.
    :param output_dir: Directorio de guardado.
    :param task: Tipo de tarea ('classification' o 'regression').
    """
    plt.figure(figsize=(12, 8))
    col_perf = metric_perf.upper()
    col_time = metric_time.upper()
    col_size = metric_size.upper()

    pareto_df = get_pareto_front(df_avg, x_col=col_size, y_col=col_perf)
    
    # Configurar límites y datos según el tipo de tarea
    if task == 'regression':
        # Para regresión abarcamos todo el frente de Pareto en X, ajustando el zoom en Y
        min_pareto_y = pareto_df[col_perf].min()
        max_pareto_y = pareto_df[col_perf].max()
        min_pareto_x = pareto_df[col_size].min()
        max_pareto_x = pareto_df[col_size].max()
        
        # Filtrar para incluir todo lo que caiga en este rango visual
        zoom_df = df_avg[(df_avg[col_perf] >= min_pareto_y * 0.95) & (df_avg[col_size] <= max_pareto_x * 1.5)]
        
        x_lim = (min_pareto_x * 0.5, max_pareto_x * 2.0)
        y_lim = (min_pareto_y * 0.95, max_pareto_y * 1.05)
        # Mantener escala logarítmica suele ser mejor para la gran dispersión en regresión
        x_scale = 'log' 
    else:
        # Lógica original focalizada en la cima del rendimiento para clasificación
        max_f1 = df_avg[col_perf].max()
        max_pareto_size = pareto_df[col_size].max()
        
        zoom_df = df_avg[(df_avg[col_perf] >= max_f1 - 0.0444) & (df_avg[col_size] <= max_pareto_size * 5)]
        
        min_y_zoom = zoom_df[col_perf].min() * 0.98
        
        x_lim = (zoom_df[col_size].min() * 0.5, zoom_df[col_size].max() * 1.5)
        y_lim = (min_y_zoom, max_f1 * 1.02)
        x_scale = 'linear'

    # Dibujar el scatter plot
    sns.scatterplot(
        data=zoom_df, x=col_size, y=col_perf, size=col_time,
        hue='Framework', sizes=(50, 900), alpha=0.7, palette='Set1', edgecolor="black"
    )

    # Dibujar la línea del frente de Pareto
    plt.plot(pareto_df[col_size], pareto_df[col_perf],
             color='red', linestyle='--', linewidth=2, label='Frente de Pareto', zorder=1)

    # Imprimir etiquetas iterando DIRECTAMENTE sobre pareto_df para no perder ninguna
    for i, row in pareto_df.iterrows():
        plt.text(row[col_size] * 1.1, row[col_perf], row['Model'],
                 fontsize=9, ha='left', va='center',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.2'))

    # Aplicar formato a los ejes
    plt.xscale(x_scale)
    plt.xlim(x_lim)
    
    # Asegurar que el límite inferior en regresión nunca sea menor que 0
    if task == 'regression':
        y_lim = (max(0, y_lim[0]), y_lim[1])
        
    plt.ylim(y_lim)

    plt.title("Frente de Pareto - ZOOM", fontsize=16)
    plt.xlabel("Tamaño medio del modelo (KB)", fontsize=12)
    plt.ylabel(f"Rendimiento medio ({col_perf})", fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_tradeoff_pareto_ZOOM.png"), dpi=300)
    plt.close()

def plot_pareto_plotly(df_avg, metric_perf, metric_time, metric_size, output_dir, task):
    """
    Genera un archivo HTML interactivo con Plotly para visualizar el trade-off global.
    Permite hacer hover (pasar el ratón) para ver los datos exactos de cada modelo.
    
    :param df_avg: DataFrame con la media agrupada.
    :param metric_perf: Nombre de la métrica de rendimiento.
    :param metric_time: Nombre de la métrica de latencia.
    :param metric_size: Nombre de la métrica de tamaño.
    :param output_dir: Directorio de guardado.
    :param task: Tipo de tarea ('classification' o 'regression').
    """
    col_perf = metric_perf.upper()
    col_time = metric_time.upper()
    col_size = metric_size.upper()

    fig = px.scatter(
        df_avg, x=col_size, y=col_perf, size=col_time, color='Framework',
        hover_name='Model',
        hover_data={col_size: ':.2f', col_perf: ':.4f', col_time: ':.4f'},
        log_x=True,
        title="Explorador Interactivo: Trade-off de Modelos (Pase el ratón para detalles)",
        size_max=40, template="plotly_white"
    )

    pareto_df = get_pareto_front(df_avg, x_col=col_size, y_col=col_perf)

    # Añadir traza manual para la línea del frente de Pareto
    fig.add_trace(
        go.Scatter(
            x=pareto_df[col_size], y=pareto_df[col_perf],
            mode='lines', name='Frente de Pareto',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        )
    )
    
    # Forzar el rango del eje Y para que siempre empiece en 0 si la tarea es regresión
    if task == 'regression':
        max_y = df_avg[col_perf].max()
        fig.update_yaxes(range=[0, max_y * 1.05])

    html_path = os.path.join(output_dir, "global_tradeoff_interactive.html")
    fig.write_html(html_path)

def plot_complexity_scatter(df, x_col, y_col, x_label, y_label, plot_title, output_filename, output_dir):
    """
    Genera un gráfico estático individual (PNG) que relaciona una variable de 
    complejidad (instancias o características) con una métrica de rendimiento.
    
    :param df: DataFrame completo.
    :param x_col: Columna del eje X (ej. 'Features' o 'Instances').
    :param y_col: Columna del eje Y (ej. latencia o tamaño).
    :param x_label: Etiqueta para el eje X.
    :param y_label: Etiqueta para el eje Y.
    :param plot_title: Título principal.
    :param output_filename: Nombre del archivo .png resultante.
    :param output_dir: Directorio de guardado.
    """
    plt.figure(figsize=(9, 6))
    
    sns.scatterplot(
        data=df, x=x_col, y=y_col, hue='Framework', 
        alpha=0.6, s=70, palette='Set1', edgecolor="black"
    )
    
    plt.yscale('log')
    plt.xscale('log')
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Frameworks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300)
    plt.close()

def plot_complexity_plotly(df, x_col, y_col, x_label, y_label, plot_title, output_filename, output_dir):
    """
    Genera un archivo HTML interactivo con Plotly para analizar el impacto 
    de las instancias o características sobre una métrica específica.

    :param df: DataFrame con los datos.
    :param x_col: Columna del eje X ('Features' o 'Instances').
    :param y_col: Columna del eje Y ('INFERENCE' o 'SIZE').
    :param x_label: Etiqueta visual para el eje X.
    :param y_label: Etiqueta visual para el eje Y.
    :param plot_title: Título principal del gráfico.
    :param output_filename: Nombre del archivo .html resultante.
    :param output_dir: Directorio de guardado.
    """
    fig = px.scatter(
        df, x=x_col, y=y_col, color='Framework',
        hover_name='Model',
        hover_data={'Dataset': True, x_col: True, y_col: ':.4f'},
        log_x=True, log_y=True,
        title=f"Explorador Interactivo: {plot_title} (Pase el ratón para detalles)",
        opacity=0.6, template="plotly_white"
    )

    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title_text="Framework"
    )

    html_path = os.path.join(output_dir, output_filename)
    fig.write_html(html_path)


# =================
# BLOQUE PRINCIPAL
# =================
if __name__ == "__main__":
    # 1. Configurar el analizador de argumentos por línea de comandos
    parser = argparse.ArgumentParser(description="Generador de gráficos Trade-off de Modelos ML.")
    parser.add_argument(
        '-t', '--task', 
        type=str, 
        choices=['classification', 'regression'], 
        required=True,
        help="Tarea a analizar: 'classification' o 'regression'."
    )
    parser.add_argument(
        '-m', '--metric', 
        type=str, 
        help="Métrica de rendimiento (ej. 'f1' para clasificación, 'r2' para regresión). "
             "Si no se especifica, se asignará f1 o r2 automáticamente."
    )

    # 2. Leer y parsear los argumentos introducidos por el usuario
    args = parser.parse_args()

    # 3. Asignación dinámica de variables operativas
    TASK = args.task
    if args.metric:
        METRIC_PERF = args.metric
    else:
        METRIC_PERF = "f1" if TASK == "classification" else "r2"
    METRIC_TIME = "inference"
    METRIC_SIZE = "size"

    # Definir rutas relativas seguras usando el directorio del propio script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.abspath(os.path.join(script_dir, "../../results2"))
    OUTPUT_GRAPHS_DIR = os.path.join(RESULTS_DIR, f"graphs/tradeoffs/{TASK}")
    os.makedirs(OUTPUT_GRAPHS_DIR, exist_ok=True) 

    print(f"Cargando datos para el análisis de compromisos ({TASK.upper()})...")
    print(f"Métrica de rendimiento seleccionada: {METRIC_PERF.upper()}")

    # 4. Proceso ETL (Extract, Transform, Load)
    df_perf = load_and_melt_metric(RESULTS_DIR, TASK, METRIC_PERF)
    df_time = load_and_melt_metric(RESULTS_DIR, TASK, METRIC_TIME)
    df_size = load_and_melt_metric(RESULTS_DIR, TASK, METRIC_SIZE)

    # Unir las tres métricas usando cruces internos por dataset y modelo
    master_df = pd.merge(df_perf, df_time, on=['Dataset', 'Model'], how='inner')
    master_df = pd.merge(master_df, df_size, on=['Dataset', 'Model'], how='inner')

    # Añadir columna de Framework basándose en el nombre de los modelos
    master_df['Framework'] = master_df['Model'].apply(extract_model_details)

    # 5. Mapeo de características e instancias por dataset
    def get_features(ds_name_csv):
        ds_name_clean = CSV_TO_DATASET_MAP.get(ds_name_csv.strip().upper(), ds_name_csv)
        if ds_name_clean in DATASETS_INFO[TASK]:
            return DATASETS_INFO[TASK][ds_name_clean]['features']
        return np.nan

    def get_instances(ds_name_csv):
        ds_name_clean = CSV_TO_DATASET_MAP.get(ds_name_csv.strip().upper(), ds_name_csv)
        if ds_name_clean in DATASETS_INFO[TASK]:
            return DATASETS_INFO[TASK][ds_name_clean]['instances']
        return np.nan

    master_df['Features'] = master_df['Dataset'].apply(get_features)
    master_df['Instances'] = master_df['Dataset'].apply(get_instances)
    master_df = master_df.dropna()
    
    # 6. Comprobación de seguridad
    if master_df.empty:
        print("\n¡ADVERTENCIA! No se han encontrado datos para los parámetros especificados.")
        print(f"Revise que existan ficheros con el patrón *_ {TASK} _{METRIC_PERF}.csv en {RESULTS_DIR}")
        exit()
    
    print(f"\nDatos combinados generados: {master_df.shape[0]} registros listos para graficar.")

    # 7. Agregación de datos: media de las métricas por cada modelo/framework
    df_avg_models = master_df.groupby(['Model', 'Framework']).mean(numeric_only=True).reset_index()

    col_time_upper = METRIC_TIME.upper()
    col_size_upper = METRIC_SIZE.upper()

    # 8. Gráfico general y frente de Pareto (estático e interactivo)
    plot_pareto_bubble(df_avg_models, METRIC_PERF, METRIC_TIME, METRIC_SIZE, OUTPUT_GRAPHS_DIR, TASK)
    plot_pareto_zoomed(df_avg_models, METRIC_PERF, METRIC_TIME, METRIC_SIZE, OUTPUT_GRAPHS_DIR, TASK)
    plot_pareto_plotly(df_avg_models, METRIC_PERF, METRIC_TIME, METRIC_SIZE, OUTPUT_GRAPHS_DIR, TASK)
    
    # 9. Gráficos estáticos de complejidad (características e instancias)
    plot_complexity_scatter(
        master_df, 'Features', col_time_upper, "Nº de características", "Tiempo de inferencia (s)",
        "Impacto de características en latencia", "complexity_features_latency.png", OUTPUT_GRAPHS_DIR
    )
    plot_complexity_scatter(
        master_df, 'Features', col_size_upper, "Nº de características", "Tamaño del modelo (KB)",
        "Impacto de características en tamaño", "complexity_features_size.png", OUTPUT_GRAPHS_DIR
    )
    plot_complexity_scatter(
        master_df, 'Instances', col_time_upper, "Nº de instancias", "Tiempo de inferencia (s)",
        "Impacto de instancias en latencia", "complexity_instances_latency.png", OUTPUT_GRAPHS_DIR
    )
    plot_complexity_scatter(
        master_df, 'Instances', col_size_upper, "Nº de instancias", "Tamaño del modelo (KB)",
        "Impacto de instancias en tamaño", "complexity_instances_size.png", OUTPUT_GRAPHS_DIR
    )
    
    # 10. Gráficos interactivos de complejidad (características e instancias)
    plot_complexity_plotly(
        master_df, 'Features', col_time_upper, "Nº de características", "Tiempo de inferencia (s)",
        "Impacto de características en latencia", "complexity_features_latency_interactive.html", OUTPUT_GRAPHS_DIR
    )
    plot_complexity_plotly(
        master_df, 'Features', col_size_upper, "Nº de características", "Tamaño del modelo (KB)",
        "Impacto de características en tamaño", "complexity_features_size_interactive.html", OUTPUT_GRAPHS_DIR
    )
    plot_complexity_plotly(
        master_df, 'Instances', col_time_upper, "Nº de instancias", "Tiempo de inferencia (s)",
        "Impacto de instancias en latencia", "complexity_instances_latency_interactive.html", OUTPUT_GRAPHS_DIR
    )
    plot_complexity_plotly(
        master_df, 'Instances', col_size_upper, "Nº de instancias", "Tamaño del modelo (KB)",
        "Impacto de instancias en tamaño", "complexity_instances_size_interactive.html", OUTPUT_GRAPHS_DIR
    )

    print("\nAnálisis completado. Gráficos guardados en:", OUTPUT_GRAPHS_DIR)

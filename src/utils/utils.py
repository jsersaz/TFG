import os


# ------------------------------------------------------
# OBTENER RAÍZ DEL PROYECTO BUSCANDO CARPETA 'results'
# ------------------------------------------------------
def get_project_root():
    """
    Busca hacia arriba en el árbol de directorios hasta encontrar una carpeta llamada 'results'.
    Devuelve la ruta absoluta de ese directorio raíz del proyecto.
    Si no se encuentra la carpeta 'results', lanza una excepción RuntimeError.
    """
    # Obtener la ruta absoluta del directorio donde se encuentra este fichero (utils.py)
    path = os.path.abspath(os.path.dirname(__file__))

    # Mientras no se haya llegado a la raíz del sistema de archivos
    while True:
        # Comprobar si existe una subcarpeta 'results' en el directorio actual
        if os.path.isdir(os.path.join(path, "results")):
            return path	# Si existe, hemos encontrado la raíz del proyecto

		# Subir un nivel (directorio padre)
        parent = os.path.dirname(path)

		# Si el padre es el mismo que el actual, hemos llegado a la raíz del sistema de archivos
        if parent == path:
            raise RuntimeError("Project root with 'results' folder not found")

		# Continuar la búsqueda en el directorio padre
        path = parent

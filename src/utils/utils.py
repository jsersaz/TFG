import os


# ------------------------------------------------------
# OBTENER RAÍZ DEL PROYECTO BUSCANDO CARPETA 'results2'
# ------------------------------------------------------
def get_project_root():
    """
    Busca hacia arriba en el árbol de directorios hasta encontrar una carpeta llamada 'results2'.
    Devuelve la ruta absoluta de ese directorio raíz del proyecto.
    Si no se encuentra la carpeta 'results2', lanza una excepción RuntimeError.
    """
    # Obtener la ruta absoluta del directorio donde se encuentra este fichero (utils.py)
    path = os.path.abspath(os.path.dirname(__file__))

    # Mientras no se haya llegado a la raíz del sistema de archivos
    while True:
        # Comprobar si existe una subcarpeta 'results2' en el directorio actual
        if os.path.isdir(os.path.join(path, "results2")):
            return path	# Si existe, hemos encontrado la raíz del proyecto

		# Subir un nivel (directorio padre)
        parent = os.path.dirname(path)

		# Si el padre es el mismo que el actual, hemos llegado a la raíz del sistema de archivos
        if parent == path:
            raise RuntimeError("Project root with 'results2' folder not found")

		# Continuar la búsqueda en el directorio padre
        path = parent

import logging
import numpy as np
import onnx
import onnxruntime as ort
import os
import tempfile
import time
from onnx import version_converter
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_dynamic, quantize_static, QuantType, shape_inference
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ====================================================
# CalibrationDataReader para la cuantización estática
# ====================================================
class CalibDataReader(CalibrationDataReader):
    """
    Clase que implementa la interfaz CalibrationDataReader de ONNX Runtime para la cuantización estática.
    Se utiliza para proporcionar datos de calibración al cuantizador estático de ONNX Runtime.
    """
    def __init__(self, data):
        """
        Constructor.
        
        :param self: Instancia del objeto.
        :param data: Datos de calibración.
        """
        self.data = data
        self.index = 0
        
    def get_next(self):
        """
        Obtiene el siguiente batch de datos de calibración.

        :param self: Instancia del objeto.

        :returns: Diccionario con los datos de calibración o None si no hay más.
        """
        if self.index >= len(self.data):
            return None
        # Usamos slicing [index:index+1] para mantener la forma 2D (1, num_features)
        # en lugar de indexación directa [index] que devolvería 1D (num_features).
        # ONNX Runtime espera estrictamente tensores de Rango 2 para la entrada.
        sample = self.data[self.index:self.index+1]
        # sample = self.data[self.index]
        self.index += 1
        return {"input": sample.astype(np.float32)}


# ==============
# CLASIFICACIÓN
# ==============
class ONNXModelFromSklearnClassification(BaseEstimator):
    """
    Wrapper que permite la conversión de modelos de clasificación de scikit-learn a ONNX.
    - Aplica técnicas de compresión según el tipo de modelo
    - Convierte a ONNX
    - Cuantización INT8
    - Inferencia mediante ONNX Runtime.
    """
    # ------------
    # CONSTRUCTOR
    # ------------
    def __init__(self, sklearn_model_builder, quantization_mode='static'):
        """
        Constructor.
        
        :param self: Instancia del objeto.
        :param sklearn_model_builder: Función que construye y devuelve un modelo de sklearn.
        :param quantization_mode: Modo de cuantización ('dynamic' o 'static').
        """
        self.sklearn_model_builder = sklearn_model_builder	# Función que construye el modelo sklearn
        self.quantization_mode = quantization_mode  		# 'dynamic' o 'static'
        self.sklearn_model = None           				# Modelo sklearn entrenado
        self.onnx_session = None            				# Sesión de ONNX Runtime para inferencia
        self._model_path = None             				# Ruta al archivo ONNX temporal
        self.train_time_ = None     						# Tiempo de entrenamiento (s)
        self.conversion_time_ = None        				# Tiempo de conversión a ONNX + cuantización (s)
        self.model_size_ = None             				# Tamaño del modelo (KB)
        self._scaler = None  								# Para almacenar el scaler si existe

    # -----------
    # UTILIDADES
    # -----------
    def _get_final_estimator(self, model):
        """
        Obtiene el estimador final de un Pipeline de sklearn (el último paso).

        :param self: Instancia del objeto.
        :param model: Modelo o Pipeline de sklearn.
        
        :returns: Estimador final (último paso) o el mismo modelo si no es Pipeline.
        """
        if hasattr(model, "named_steps"):
            return list(model.named_steps.values())[-1]
        return model

    def _is_pipeline(self, model):
        """
        Indica si un modelo es un Pipeline de sklearn.

        :param self: Instancia del objeto.
        :param model: Modelo a comprobar.
        
        :returns: True si es Pipeline, False en caso contrario.
        """
        return hasattr(model, "named_steps")

    def _replace_final_estimator(self, new_estimator):
        """
        Reemplaza el último paso del Pipeline (o el modelo completo) por un nuevo estimador.

        :param self: Instancia del objeto.
        :param new_estimator: Nuevo estimador que sustituirá al último paso.
        """
        if self._is_pipeline(self.sklearn_model):
            steps = list(self.sklearn_model.named_steps.items())
            steps[-1] = (steps[-1][0], new_estimator)
            from sklearn.pipeline import Pipeline
            self.sklearn_model = Pipeline(steps)
        else:
            self.sklearn_model = new_estimator
            
    def _quantize_onnx_dynamic(self, path):
        """
        Cuantiza un modelo ONNX a INT8 de forma dinámica.
        Si el opset del modelo es menor a 11, lo actualiza a 13 automáticamente.
        En caso de error en la cuantización, devuelve el modelo sin cuantizar.

        :param self: Instancia del objeto.
        :param path: Ruta al archivo ONNX de entrada.
        
        :returns: Ruta al archivo ONNX cuantizado (o al original si falla la cuantización).
        """
        model = onnx.load(path)
        opset_version = model.opset_import[0].version if model.opset_import else 0
        
        if opset_version < 11:
            # Actualizar a opset 13
            model = version_converter.convert_version(model, 13)
            updated_path = path.replace(".onnx", "_updated.onnx")
            onnx.save(model, updated_path)
            path_to_quant = updated_path
        else:
            path_to_quant = path
            
        quant_path = path.replace(".onnx", "_int8_dynamic.onnx")
        try:
            quantize_dynamic(path_to_quant, quant_path, weight_type=QuantType.QInt8)
            # Limpiar archivo actualizado si se creó
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return quant_path
        except Exception as e:
            # Fallback: sin cuantización
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return path  # Fallback a float
        
    def _quantize_onnx_static(self, path, X_calib):
        """
        Cuantiza un modelo ONNX a INT8 de forma estática usando un conjunto de calibración.
        Si el opset del modelo es menor a 11, lo actualiza a 13 automáticamente.
        En caso de error en la cuantización, devuelve el modelo sin cuantizar.
        
        :param self: Instancia del objeto.
        :param path: Ruta al archivo ONNX de entrada.
        :param X_calib: Array de calibración (numpy array).
        
        :returns: Ruta al archivo ONNX cuantizado (o al original si falla la cuantización).
        """
        model = onnx.load(path)
        opset_version = model.opset_import[0].version if model.opset_import else 0
        
        if opset_version < 11:
            # Actualizar a opset 13
            model = version_converter.convert_version(model, 13)
            updated_path = path.replace(".onnx", "_updated.onnx")
            onnx.save(model, updated_path)
            path_to_quant = updated_path
        else:
            path_to_quant = path

        calib_reader = CalibDataReader(X_calib)
        quant_path = path.replace(".onnx", "_int8_static.onnx")
        try:
            # Cuantización estática
            quantize_static(
                path_to_quant,
                quant_path,
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                extra_options={'ActivationSymmetric': True}
            )
            # Limpiar archivo actualizado si se creó
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return quant_path
        except Exception as e:
            # Fallback a cuantización dinámica si falla la estática
            print(f"Static quantization failed: {e}. Falling back to dynamic.")
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return self._quantize_onnx_dynamic(path)
   
    # ----------------------------------
    # 1. KNN -> REDUCCIÓN DE PROTOTIPOS
    # ----------------------------------
    def _prototype_reduce(self, X, y):
        """
        Reduce el número de prototipos en KNN mediante clustering por clase.
        Para cada clase, aplica MiniBatchKMeans y reemplaza los puntos originales por los centroides.
        La etiqueta de cada centroide es la clase mayoritaria de los puntos que lo componen.

        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        :param y: Array de etiquetas (n_samples,).
        
        :returns: Tuple (X_reduced, y_reduced) con los datos reducidos.
        """
        classes = np.unique(y)

        X_reduced, y_reduced = [], []

        for cls in classes:
            X_cls = X[y == cls]

            n_clusters = max(10, int(len(X_cls) * 0.05))
            n_clusters = min(n_clusters, 200)

            kmeans = MiniBatchKMeans(
                n_clusters=min(n_clusters, len(X_cls)),
                random_state=42
            )

            labels = kmeans.fit_predict(X_cls)

            X_reduced.append(kmeans.cluster_centers_)

            # Label por mayoría (mejor que asumir clase)
            y_cluster = []
            for i in range(len(kmeans.cluster_centers_)):
                # cluster_labels = y[(y == cls) & (labels == i)]
                cluster_labels = y[(y == cls)][labels == i] if len(X_cls) == len(labels) else y[y == cls]
                if len(cluster_labels) == 0:
                    y_cluster.append(cls)
                else:
                    values, counts = np.unique(cluster_labels, return_counts=True)
                    y_cluster.append(values[np.argmax(counts)])

            y_reduced.append(np.array(y_cluster))

        return np.vstack(X_reduced), np.concatenate(y_reduced)

    # ------------------------------------------
    # 2. LINEAR SVC -> SPARSITY (PODA DE PESOS)
    # ------------------------------------------
    def _sparsify_linear_model(self, threshold=1e-4):
        """
        Poda pesos pequeños en modelos lineales (LinearSVC) estableciendo a cero
        los coeficientes con valor absoluto inferior a 'threshold'.

        :param self: Instancia del objeto.
        :param threshold: Umbral por debajo del cual los coeficientes se anulan.
        """
        model = self._get_final_estimator(self.sklearn_model)

        if hasattr(model, "coef_"):
            model.coef_[np.abs(model.coef_) < threshold] = 0

    # -----------------------------------
    # 3. MLP -> SPARSITY (PODA DE PESOS)
    # -----------------------------------
    def _prune_mlp(self, threshold=1e-4):
        """
        Poda pesos en MLP (MLPClassifier) estableciendo a cero los pesos
        con valor absoluto inferior a 'threshold'.

        :param self: Instancia del objeto.
        :param threshold: Umbral de poda.
        """
        model = self._get_final_estimator(self.sklearn_model)

        if hasattr(model, "coefs_"):
            new_layers = []
            for w in model.coefs_:
                w = np.where(np.abs(w) < threshold, 0, w)
                new_layers.append(w)
            model.coefs_ = new_layers

    # -----------------------------------
    # 4. DECISION TREE -> PODA DEL ÁRBOL
    # -----------------------------------
    def _prune_tree(self):
        """
        Poda árboles de decisión (DecisionTreeClassifier) marcando como hojas
        aquellos nodos con impureza muy baja (impurity < 1e-6) para reducir el tamaño.
        
        :param self: Instancia del objeto.
        """
        model = self._get_final_estimator(self.sklearn_model)

        if hasattr(model, "tree_"):
            tree = model.tree_
            leaf_mask = tree.impurity < 1e-6
            tree.threshold[leaf_mask] = -2

    # --------------------------------
    # 5. RANDOM FOREST -> DESTILACIÓN
    # --------------------------------
    def _distill_rf(self, X):
        """
        Destila un Random Forest en un árbol de decisión más pequeño.

        Se entrena un DecisionTreeClassifier con profundidad 8 y máximo 64 hojas
        usando como etiquetas las predicciones del bosque original.

        :param self: Instancia del objeto.
        :param X: Datos de entrenamiento (características).
        """
        model = self._get_final_estimator(self.sklearn_model)

        if "RandomForest" not in str(type(model)):
            return

        teacher = self.sklearn_model
        soft_labels = np.argmax(teacher.predict_proba(X), axis=1)

        student = DecisionTreeClassifier(
            max_depth=8,
            max_leaf_nodes=64,
            random_state=42
        )

        student.fit(X, soft_labels)

        self._replace_final_estimator(student)

    # -----------------------
    # PIPELINE DE COMPRESIÓN
    # -----------------------
    def _apply_model_specific_compression(self, X, y):
        """
        Aplica técnicas de compresión específicas según el tipo de modelo.
        1. Construir el modelo con sklearn_model_builder.
        - KNN -> aplicar prototype reduction.
        2. Entrenar el modelo.
        - SVC, MLP -> aplicar sparsity/pruning.
        - DecisionTree, RandomForest -> aplicar pruning/distillation.

        :param self: Instancia del objeto.
        :param X: Características de entrenamiento.
        :param y: Etiquetas de entrenamiento.
        """
        model = self.sklearn_model_builder()
        name = str(type(self._get_final_estimator(model)))

        # ----
        # KNN
        # ----
        if "KNeighbors" in name:
            X, y = self._prototype_reduce(X, y)

		# ---------
  		# ENTRENAR
    	# ---------
        self.sklearn_model = model
        self.sklearn_model.fit(X, y)
        
        # Almacenar el scaler si existe en el pipeline
        if hasattr(self.sklearn_model, 'named_steps') and 'scaler' in self.sklearn_model.named_steps:
            self._scaler = self.sklearn_model.named_steps['scaler']
        else:
            self._scaler = None
            
        final = self._get_final_estimator(self.sklearn_model)

        # -----------
        # Linear SVC
        # -----------
        if isinstance(final, LinearSVC):
            self._sparsify_linear_model()

        # ----
        # MLP
        # ----
        if hasattr(final, "coefs_"):
            self._prune_mlp()

        # ---------------
        # Decision Tree
        # ---------------
        if hasattr(final, "tree_"):
            self._prune_tree()
            
        # --------------
        # Random Forest
        # --------------
        if "RandomForest" in name:
            self._distill_rf(X)

    # --------------
    # ENTRENAMIENTO
    # --------------
    def fit(self, X, y):
        """
        Entrena el modelo sklearn, aplica compresión y convierte a ONNX.
        1. Mide el tiempo de entrenamiento (incluyendo compresión).
        2. Convierte el modelo comprimido a ONNX.
        3. Cuantiza a INT8.
        4. Carga el modelo en ONNX Runtime y registra el tamaño.

        :param self: Instancia del objeto.
        :param X: Características de entrenamiento (numpy array).
        :param y: Etiquetas de entrenamiento (numpy array).
        
        :returns: Self (objeto entrenado).
        """
        # ---------------
        # FIX PARA MNIST
        # ---------------
        # ONNX Runtime rechaza constantes uint8 en varios operadores.
        # Si las etiquetas son numéricas (como en MNIST), las forzamos a int64.
        y = np.asarray(y)
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)
        
        start = time.perf_counter()
        
        self._apply_model_specific_compression(X, y)
        
        self.train_time_ = time.perf_counter() - start
        
        # --------------------------------
        # CONVERSIÓN Y EXPORTACIÓN A ONNX
        # --------------------------------
        start_conv = time.perf_counter()
        
        X_sample = X[:1].astype(np.float32)
        initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
        
        final_estimator = self._get_final_estimator(self.sklearn_model)
        
        options = {}
        if isinstance(final_estimator, LinearSVC):
            options = {id(final_estimator): {"raw_scores": True}}
            
        onnx_model = convert_sklearn(self.sklearn_model, initial_types=initial_type, options=options, target_opset=13)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        
        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR) # Solo mostrará errores críticos, ocultando los warnings hardcodeados
        
        try:
            # -------------
            # CUANTIZACIÓN
            # -------------
            # Decidir tipo de cuantización
            if self.quantization_mode == 'static' and self._scaler is not None:
                # Usar primeras 100 muestras escaladas para calibración
                X_calib = self._scaler.transform(X[:100]) if len(X) >= 100 else self._scaler.transform(X)
                quantized_path = self._quantize_onnx_static(path, X_calib)
            elif self.quantization_mode == 'static' and self._scaler is None:
                X_calib = X[:100].astype(np.float32) if len(X) >= 100 else X.astype(np.float32)
                quantized_path = self._quantize_onnx_static(path, X_calib)
            else:
                quantized_path = self._quantize_onnx_dynamic(path)
        finally:
            # Restaurar el nivel de logs original para no afectar al resto del script principal
            logger.setLevel(old_level)

        # Limpiar archivo base
        if quantized_path != path and os.path.exists(path):
            os.remove(path)
            
        self._model_path = quantized_path
        
        self.onnx_session = ort.InferenceSession(self._model_path)
        self.model_size_ = os.path.getsize(self._model_path) / 1024
        
        self.conversion_time_ = time.perf_counter() - start_conv
        return self

    # ------------
    # INFERENCIA
    # ------------
    def predict(self, X):
        """
        Infiere usando ONNX Runtime.

        :param self: Instancia del objeto.
        :param X: Características de entrada (numpy array).
        
        :returns: Predicciones (array 1D).
        """
        X = X.astype(np.float32)
        input_name = self.onnx_session.get_inputs()[0].name
        return self.onnx_session.run(None, {input_name: X})[0].ravel()

    def __del__(self):
        """
        Destructor: Elimina el archivo temporal que contiene el modelo para liberar recursos del sistema de archivos.
        
        :param self: Instancia del objeto.
        """
        model_path = getattr(self, '_model_path', None)
        if (model_path is not None and os.path.exists(model_path)):
            try:
                os.remove(model_path)
            except:
                pass
            
            
# ==========
# REGRESIÓN
# ==========
class ONNXModelFromSklearnRegression(BaseEstimator):
    """
    Wrapper que permite la conversión de modelos de regresión de scikit-learn a ONNX.
    - Aplica técnicas de compresión según el tipo de modelo
    - Convierte a ONNX
    - Cuantiza a INT8
    - Inferencia mediante ONNX Runtime.
    """
    # ------------
    # CONSTRUCTOR
    # ------------
    def __init__(self, sklearn_model_builder, quantization_mode='static'):
        """
        Constructor.
        
        :param self: Instancia del objeto.
        :param sklearn_model_builder: Función que construye y devuelve un modelo de sklearn.
        :param quantization_mode: Modo de cuantización ('dynamic' o 'static').
        """
        self.sklearn_model_builder = sklearn_model_builder	# Función que construye el modelo sklearn
        self.quantization_mode = quantization_mode  		# 'dynamic' o 'static'
        self.sklearn_model = None							# Modelo sklearn entrenado
        self.onnx_session = None							# Sesión de ONNX Runtime para inferencia
        self._model_path = None								# Ruta al archivo ONNX temporal
        self.train_time_ = None						        # Tiempo de entrenamiento del modelo
        self.conversion_time_ = None						# Tiempo de conversión a ONNX + cuantización (s)
        self.model_size_ = None								# Tamaño del modelo ONNX en KB
        self._scaler = None  								# Para almacenar el scaler si existe

    # -----------
    # UTILIDADES
    # -----------
    def _get_final_estimator(self, model):
        """
        Obtiene el estimador final de un Pipeline de sklearn (el último paso).

        :param self: Instancia del objeto.
        :param model: Modelo o Pipeline de sklearn.
        
        :returns: Estimador final (último paso) o el mismo modelo si no es Pipeline.
        """
        if hasattr(model, "named_steps"):
            return list(model.named_steps.values())[-1]
        return model

    def _is_pipeline(self, model):
        """
        Indica si un modelo es un Pipeline de sklearn.

        :param self: Instancia del objeto.
        :param model: Modelo a comprobar.
        
        :returns: True si es Pipeline,
        """
        return hasattr(model, "named_steps")

    def _replace_final_estimator(self, new_estimator):
        """
        Reemplaza el último paso del Pipeline (o el modelo completo) por un nuevo estimador.

        :param self: Instancia del objeto.
        :param new_estimator: Nuevo estimador que sustituirá al último paso.
        """
        if self._is_pipeline(self.sklearn_model):
            steps = list(self.sklearn_model.named_steps.items())
            steps[-1] = (steps[-1][0], new_estimator)

            from sklearn.pipeline import Pipeline
            self.sklearn_model = Pipeline(steps)
        else:
            self.sklearn_model = new_estimator
            
    def _quantize_onnx_dynamic(self, path):
        """
        Cuantiza un modelo ONNX a INT8 de forma dinámica.
        Si el opset del modelo es menor a 11, lo actualiza a 13 automáticamente.
        En caso de error en la cuantización, devuelve el modelo sin cuantizar.

        :param self: Instancia del objeto.
        :param path: Ruta al archivo ONNX de entrada.
        
        :returns: Ruta al archivo ONNX cuantizado (o al original si falla la cuantización).
        """
        model = onnx.load(path)
        opset_version = model.opset_import[0].version if model.opset_import else 0

        if opset_version < 11:
            # Actualizar a opset 13
            model = version_converter.convert_version(model, 13)
            updated_path = path.replace(".onnx", "_updated.onnx")
            onnx.save(model, updated_path)
            path_to_quant = updated_path
        else:
            path_to_quant = path

        quant_path = path.replace(".onnx", "_int8_dynamic.onnx")
        try:
            quantize_dynamic(path_to_quant, quant_path, weight_type=QuantType.QInt8)
            # Limpiar archivo actualizado si se creó
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return quant_path
        except Exception:
            # Fallback: sin cuantización
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return path  # Fallback a float
        
    def _quantize_onnx_static(self, path, X_calib):
        """
        Cuantiza un modelo ONNX a INT8 de forma estática usando un conjunto de calibración.
        Si el opset del modelo es menor a 11, lo actualiza a 13 automáticamente.
        En caso de error en la cuantización, devuelve el modelo sin cuantizar.
        
        :param self: Instancia del objeto.
        :param path: Ruta al archivo ONNX de entrada.
        :param X_calib: Array de calibración (numpy array).
        
        :returns: Ruta al archivo ONNX cuantizado (o al original si falla la cuantización).
        """
        model = onnx.load(path)
        opset_version = model.opset_import[0].version if model.opset_import else 0

        if opset_version < 11:
            # Actualizar a opset 13
            model = version_converter.convert_version(model, 13)
            updated_path = path.replace(".onnx", "_updated.onnx")
            onnx.save(model, updated_path)
            path_to_quant = updated_path
        else:
            path_to_quant = path

        calib_reader = CalibDataReader(X_calib)
        quant_path = path.replace(".onnx", "_int8_static.onnx")
        try:
            # Cuantización estática
            quantize_static(
                path_to_quant,
                quant_path,
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                extra_options={'ActivationSymmetric': True}
            )
            # Limpiar archivo actualizado si se creó
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return quant_path
        except Exception as e:
            # Fallback a cuantización dinámica si falla la estática
            print(f"Static quantization failed: {e}. Falling back to dynamic.")
            if path_to_quant != path and os.path.exists(path_to_quant):
                os.remove(path_to_quant)
            return self._quantize_onnx_dynamic(path)

    # ---------------------------------
    # 1. KNN → REDUCCIÓN DE PROTOTIPOS
    # ---------------------------------
    def _prototype_reduce(self, X, y):
        """
        Reduce el número de prototipos en KNN mediante clustering por clase.
        Para cada clase, aplica MiniBatchKMeans y reemplaza los puntos originales por los centroides.
        La etiqueta de cada centroide es la clase mayoritaria de los puntos que lo componen.

        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        :param y: Array de etiquetas (n_samples,).
        
        :returns: Tuple (X_reduced, y_reduced) con los datos reducidos.
        """
        n_clusters = max(10, int(len(X) * 0.05))
        n_clusters = min(n_clusters, 200)

        kmeans = MiniBatchKMeans(
            n_clusters=min(n_clusters, len(X)),
            random_state=42
        )

        labels = kmeans.fit_predict(X)

        X_reduced = kmeans.cluster_centers_
        y_reduced = np.zeros(len(X_reduced), dtype=np.float32)

        for i in range(len(X_reduced)):
            cluster_targets = y[labels == i]
            if len(cluster_targets) > 0:
                y_reduced[i] = np.mean(cluster_targets)

        return X_reduced, y_reduced

    # ------------------------------------------------
    # 2. LINEAR REGRESSION → SPARSITY (PODA DE PESOS)
    # ------------------------------------------------
    def _sparsify_linear_model(self, threshold=1e-4):
        """
        Poda pesos pequeños en modelos lineales (LinearRegression) estableciendo a cero
        los coeficientes con valor absoluto inferior a 'threshold'.

        :param self: Instancia del objeto.
        :param threshold: Umbral por debajo del cual los coeficientes se anulan.
        """
        model = self._get_final_estimator(self.sklearn_model)

        if hasattr(model, "coef_"):
            model.coef_[np.abs(model.coef_) < threshold] = 0

    # ----------------------------------
    # 3. MLP → SPARSITY (PODA DE PESOS)
    # ----------------------------------
    def _prune_mlp(self, threshold=1e-4):
        """
        Poda pesos en MLP (MLPRegressor) estableciendo a cero los pesos
        con valor absoluto inferior a 'threshold'.

        :param self: Instancia del objeto.
        :param threshold: Umbral de poda.
        """
        model = self._get_final_estimator(self.sklearn_model)

        if hasattr(model, "coefs_"):
            new_layers = []

            for w in model.coefs_:
                w = np.where(np.abs(w) < threshold, 0, w)
                new_layers.append(w)

            model.coefs_ = new_layers

    # ----------------------------------
    # 4. DECISION TREE → PODA DEL ÁRBOL
    # ----------------------------------
    def _prune_tree(self):
        """
        Poda árboles de decisión (DecisionTreeRegressor) marcando como hojas
        aquellos nodos con impureza muy baja (impurity < 1e-6) para reducir el tamaño.
        
        :param self: Instancia del objeto.
        """
        model = self._get_final_estimator(self.sklearn_model)

        if hasattr(model, "tree_"):
            tree = model.tree_
            leaf_mask = tree.impurity < 1e-6
            tree.threshold[leaf_mask] = -2

    # --------------------------------
    # 5. RANDOM FOREST -> DESTILACIÓN
    # --------------------------------
    def _distill_rf(self, X):
        """
        Destila un Random Forest en un árbol de decisión más pequeño.

        Se entrena un DecisionTreeRegressor con profundidad 8 y máximo 64 hojas
        usando como etiquetas las predicciones del bosque original.

        :param self: Instancia del objeto.
        :param X: Datos de entrenamiento (características).
        """
        model = self._get_final_estimator(self.sklearn_model)

        if "RandomForest" not in str(type(model)):
            return

        teacher = self.sklearn_model
        soft_targets = teacher.predict(X)

        student = DecisionTreeRegressor(
            max_depth=8,
            max_leaf_nodes=64,
            random_state=42
        )
        student.fit(X, soft_targets)

        self._replace_final_estimator(student)

    # -----------------------
    # PIPELINE DE COMPRESIÓN
    # -----------------------
    def _apply_model_specific_compression(self, X, y):
        """
        Aplica técnicas de compresión específicas según el tipo de modelo.
        1. Construir el modelo con sklearn_model_builder.
        - KNN -> aplicar prototype reduction.
        2. Entrenar el modelo.
        - LinearRegression, MLP -> aplicar sparsity/pruning.
        - DecisionTree, RandomForest -> aplicar pruning/distillation.

        :param self: Instancia del objeto.
        :param X: Características de entrenamiento.
        :param y: Valores objetivo.
        """
        model = self.sklearn_model_builder()
        name = str(type(self._get_final_estimator(model)))

        # ----
        # KNN
        # ----
        if "KNeighbors" in name:
            X, y = self._prototype_reduce(X, y)

        # ---------
        # ENTRENAR
        # ---------
        self.sklearn_model = model
        self.sklearn_model.fit(X, y)
        
        # Guardar scaler si existe en el pipeline
        if hasattr(self.sklearn_model, 'named_steps') and 'scaler' in self.sklearn_model.named_steps:
            self._scaler = self.sklearn_model.named_steps['scaler']
        else:
            self._scaler = None

        final = self._get_final_estimator(self.sklearn_model)

        # ------------------
        # Linear Regression
        # ------------------
        if isinstance(final, LinearRegression):
            self._sparsify_linear_model()

        # ----
        # MLP
        # ----
        if hasattr(final, "coefs_"):
            self._prune_mlp()

        # ---------------
        # Decision Tree
        # ---------------
        if hasattr(final, "tree_"):
            self._prune_tree()

        # --------------
        # Random Forest
        # --------------
        if "RandomForest" in name:
            self._distill_rf(X)

    # --------------
    # ENTRENAMIENTO
    # --------------
    def fit(self, X, y):
        """
        Entrena el modelo sklearn, aplica compresión y convierte a ONNX.
        1. Mide el tiempo de entrenamiento (incluyendo compresión).
        2. Convierte el modelo comprimido a ONNX.
        3. Cuantiza a INT8.
        4. Carga el modelo en ONNX Runtime y registra el tamaño.

        :param self: Instancia del objeto.
        :param X: Características de entrenamiento (numpy array).
        :param y: Valores objetivo (numpy array).
        
        :returns: Self (objeto entrenado).
        """
        start = time.perf_counter()
        
        self._apply_model_specific_compression(X, y)
        
        self.train_time_ = time.perf_counter() - start

        # --------------------------------
        # CONVERSIÓN Y EXPORTACIÓN A ONNX
        # --------------------------------
        start_conv = time.perf_counter()
        
        initial_type = [('input', FloatTensorType([None, X.shape[1]]))]

        onnx_model = convert_sklearn(self.sklearn_model, initial_types=initial_type, target_opset=13)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR) # Solo mostrará errores críticos, ocultando los warnings hardcodeados
        
        try:
            # -------------
            # CUANTIZACIÓN
            # -------------
            # Seleccionar tipo de cuantización
            if self.quantization_mode == 'static' and self._scaler is not None:
                # Usar primeras 100 muestras escaladas para calibración
                X_calib = self._scaler.transform(X[:100]) if len(X) >= 100 else self._scaler.transform(X)
                quantized_path = self._quantize_onnx_static(path, X_calib)
            elif self.quantization_mode == 'static' and self._scaler is None:
                X_calib = X[:100].astype(np.float32) if len(X) >= 100 else X.astype(np.float32)
                quantized_path = self._quantize_onnx_static(path, X_calib)
            else:
                quantized_path = self._quantize_onnx_dynamic(path)
        finally:
            # Restaurar el nivel de logs original para no afectar al resto del script principal
            logger.setLevel(old_level)

        # Limpiar archivo base
        if quantized_path != path and os.path.exists(path):
            os.remove(path)

        self._model_path = quantized_path
        
        self.onnx_session = ort.InferenceSession(self._model_path)
        self.model_size_ = os.path.getsize(self._model_path) / 1024
        
        self.conversion_time_ = time.perf_counter() - start_conv

        return self

    # -----------
    # INFERENCIA
    # -----------
    def predict(self, X):
        """
        Infiere usando ONNX Runtime.

        :param self: Instancia del objeto.
        :param X: Características de entrada (numpy array).
        
        :returns: Predicciones (array 1D).
        """
        X = X.astype(np.float32)
        input_name = (self.onnx_session.get_inputs()[0].name)
        pred = self.onnx_session.run(None, {input_name: X})[0].ravel()
        return pred.ravel()

    # -----------
    # DESTRUCTOR
    # -----------
    def __del__(self):
        """
        Destructor: Elimina el archivo temporal que contiene el modelo para liberar recursos del sistema de archivos.
        
        :param self: Instancia del objeto.
        """
        model_path = getattr(self, '_model_path', None)
        if (model_path is not None and os.path.exists(model_path)):
            try:
                os.remove(model_path)
            except:
                pass

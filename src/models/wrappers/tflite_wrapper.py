import numpy as np
import os
import random
import tempfile
import tensorflow as tf
import time
from ai_edge_litert.interpreter import Interpreter
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ======================
# CLASES DE DESTILACIÓN
# ======================
class KDKerasClassifier(tf.keras.Model):
    """
    Modelo Keras personalizado para Clasificación con Knowledge Distillation.
    Soporta profesores No-Neuronales (RF, XGBoost) mediante probabilidades directas.
    """
    def __init__(self, student, alpha=0.5, temperature=1.0, is_binary=True):
        """
        Constructor.
        
        :param self: Instancia del objeto.
        :param student: Modelo Keras del estudiante.
        :param alpha: Peso de la pérdida de destilación de conocimiento (0.0 a 1.0).
        :param temperature: Temperatura para suavizar las predicciones del profesor.
        :param is_binary: True si es clasificación binaria, False si es multiclase.
        """
        super().__init__()
        self.student = student
        self.alpha = alpha
        self.temperature = temperature
        self.is_binary = is_binary
        # Rastreadores individuales
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.dist_loss_tracker = tf.keras.metrics.Mean(name="dist_loss")
        
    @property
    def metrics(self):
        """
        Devuelve la lista de métricas para que Keras las reinicie automáticamente en cada época.
        
        :param self: Instancia del objeto.
        
        :returns: Lista de métricas para pérdida total, pérdida del estudiante y pérdida de distilación.
        """
        # Permite a Keras reiniciar las métricas automáticamente en cada época
        return [self.loss_tracker, self.student_loss_tracker, self.dist_loss_tracker]

    def call(self, inputs, training=False):
        """
        Llama al modelo del estudiante para obtener los logits.
        
        :param self: Instancia del objeto.
        :param inputs: Entradas del modelo (batch de características).
        :param training: Booleano que indica si es modo entrenamiento o inferencia.
        
        :returns: Logits del estudiante (sin activación).
        """
        return self.student(inputs, training=training)

    def train_step(self, data):
        """
        Paso de entrenamiento personalizado para Knowledge Distillation.
        
        :param self: Instancia del objeto.
        :param data: Tupla (X, (y_true, y_teacher)) donde:
                     - X: Batch de características.
                     - y_true: Etiquetas verdaderas (ground truth).
                     - y_teacher: Predicciones del profesor (soft targets).
                     
        :returns: Diccionario con métricas de pérdida total, pérdida del estudiante y pérdida de distilación.
        """
        x, y = data
        y_true, y_teacher = y[0], y[1]

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            
            if self.is_binary:
                y_true_casted = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
                y_teacher_casted = tf.cast(tf.reshape(y_teacher, [-1, 1]), tf.float32)
                # Hard loss (estudiante logits vs datos reales)
                student_loss = tf.keras.losses.binary_crossentropy(
                    y_true_casted, student_logits, from_logits=True
                )
                # Soft loss (estudiante probabilidades suavizadas vs profesor)
                student_soft_probs = tf.nn.sigmoid(student_logits / self.temperature)
                dist_loss = tf.keras.losses.binary_crossentropy(
                    y_teacher_casted, student_soft_probs
                )
            else:
                # Hard loss (estudiante logits vs datos reales)
                student_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true, student_logits, from_logits=True
                )
                # Soft loss (estudiante probabilidades suavizadas vs profesor)
                student_soft_probs = tf.nn.softmax(student_logits / self.temperature)
                dist_loss = tf.keras.losses.KLDivergence()(y_teacher, student_soft_probs)
            
            # Pérdida ponderada final
            loss = (self.alpha * student_loss) + ((1.0 - self.alpha) * dist_loss * (self.temperature ** 2))

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        # Actualizar métricas
        self.loss_tracker.update_state(loss)
        self.student_loss_tracker.update_state(student_loss)
        self.dist_loss_tracker.update_state(dist_loss)
        
        return {
            "loss": self.loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "dist_loss": self.dist_loss_tracker.result(),
        }

class KDKerasRegressor(tf.keras.Model):
    """
    Modelo Keras personalizado para Regresión con Knowledge Distillation.
    """
    def __init__(self, student, alpha=0.5):
        """
        Constructor.
        
        :param self: Instancia del objeto.
        :param student: Modelo Keras del estudiante.
        :param alpha: Peso de la pérdida de destilación de conocimiento (0.0 a 1.0).
        """
        super().__init__()
        self.student = student
        self.alpha = alpha
        # Rastreadores individuales
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.dist_loss_tracker = tf.keras.metrics.Mean(name="dist_loss")
        
    @property
    def metrics(self):
        """
        Devuelve la lista de métricas para que Keras las reinicie automáticamente en cada época.
        
        :param self: Instancia del objeto.
        
        :returns: Lista de métricas para pérdida total, pérdida del estudiante y pérdida de distilación.
        """
        # Permite a Keras reiniciar las métricas automáticamente en cada época
        return [self.loss_tracker, self.student_loss_tracker, self.dist_loss_tracker]

    def call(self, inputs, training=False):
        """
        Llama al modelo del estudiante para obtener las predicciones.
        
        :param self: Instancia del objeto.
        :param inputs: Entradas del modelo (batch de características).
        :param training: Booleano que indica si es modo entrenamiento o inferencia.
        
        :returns: Predicciones del estudiante (sin activación).
        """
        return self.student(inputs, training=training)

    def train_step(self, data):
        """
        Paso de entrenamiento personalizado para Knowledge Distillation en regresión.
        
        :param self: Instancia del objeto.
        :param data: Tupla (X, (y_true, y_teacher)) donde:
                        - X: Batch de características.
                        - y_true: Valores verdaderos (ground truth).
                        - y_teacher: Predicciones del profesor (soft targets).
                        
        :returns: Diccionario con métricas de pérdida total, pérdida del estudiante y pérdida de distilación.
        """
        x, y = data
        y_true, y_teacher = y[0], y[1]

        with tf.GradientTape() as tape:
            y_pred = self.student(x, training=True)
            
            y_true_casted = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
            y_teacher_casted = tf.cast(tf.reshape(y_teacher, [-1, 1]), tf.float32)
            # Hard loss
            student_loss = tf.keras.losses.MeanSquaredError()(y_true_casted, y_pred)
            # Soft loss
            dist_loss = tf.keras.losses.MeanSquaredError()(y_teacher_casted, y_pred)
            # Pérdida ponderada final
            loss = (self.alpha * student_loss) + ((1.0 - self.alpha) * dist_loss)

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        # Actualizar métricas
        self.loss_tracker.update_state(loss)
        self.student_loss_tracker.update_state(student_loss)
        self.dist_loss_tracker.update_state(dist_loss)
        
        return {
            "loss": self.loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "dist_loss": self.dist_loss_tracker.result(),
        }


# ==============
# CLASIFICACIÓN
# ==============
class TFLiteModelWrapperClassification(BaseEstimator):
    """
    Wrapper TensorFlow Lite (LiteRT) (clasificación).
    - Entrenamiento de un MLP con Keras
    - Conversión a formato TFLite con cuantización INT8
    - Inferencia mediante el intérprete de LiteRT.
    """
    # ------------
    # CONSTRUCTOR
    # ------------
    def __init__(self, input_dim, hidden_sizes=[32, 16], output_dim=1, epochs=20, lr=0.01, weight_decay=0.0001, 
                 batch_size=64, tol=0.001, n_iter_no_change=10, random_state=42, quantization_mode='static',
                 teacher_model=None, alpha=0.5, temperature=1.0):
        """
        Constructor.

        :param self: Instancia del objeto.
        :param input_dim: Número de características de entrada.
        :param hidden_sizes: Lista con el número de neuronas en cada capa oculta.
        :param output_dim: Número de clases (1 para binario, >1 para multiclase).
        :param epochs: Número de épocas de entrenamiento.
        :param lr: Tasa de aprendizaje (learning rate) para el optimizador Adam.
        :param weight_decay: Tegularización L2 para el optimizador Adam.
        :param batch_size: Tamaño del lote durante el entrenamiento.
        :param tol: tolerancia Para detener el entrenamiento.
        :param n_iter_no_change: Número de iteraciones sin mejora antes de detener el entrenamiento.
        :param random_state: Semilla para reproducibilidad.
        :param quantization_mode: Modo de cuantización ('static' o 'dynamic').
        :param teacher_model: Modelo profesor para destilación de conocimiento.
        :param alpha: Peso de la pérdida de destilación de conocimiento.
        :param temperature: Temperatura para suavizar las predicciones del profesor.
        """
        self.input_dim = input_dim                  # Número de características de entrada
        self.hidden_sizes = hidden_sizes            # Lista con el número de neuronas en cada capa oculta
        self.output_dim = output_dim                # Número de clases (1 para binario, >1 para multiclase)
        self.epochs = epochs                        # Número de épocas de entrenamiento
        self.lr = lr                                # Tasa de aprendizaje para el optimizador Adam
        self.weight_decay = weight_decay            # Regularización L2 para el optimizador Adam
        self.batch_size = batch_size                # Tamaño del lote para entrenamiento
        self.tol = tol                              # Tolerancia para detener el entrenamiento
        self.n_iter_no_change = n_iter_no_change    # Número de iteraciones sin mejora
        self.random_state = random_state            # Semilla para reproducibilidad
        self.quantization_mode = quantization_mode  # 'static' o 'dynamic'
        self.teacher_model = teacher_model          # Modelo profesor para destilación de conocimiento
        self.alpha = alpha                          # Peso de la pérdida de destilación de conocimiento
        self.temperature = temperature              # Temperatura para suavizar las predicciones del profesor
        self.scaler = StandardScaler()	            # Estandarizador de características
        self.label_encoder = None		            # Codificador de etiquetas
        self.model_size_kb_ = None		            # Tamaño del modelo (KB)
        self.train_time_ = None			            # Tiempo de entrenamiento (s)
        self.conversion_time_ = None	            # Tiempo de conversión a TFLite (s)
        self._model_path = None			            # Ruta al archivo .tflite temporal

    # -----------
    # UTILIDADES
    # -----------
    def _set_seeds(self):
        """
        Fija las semillas de las librerías aleatorias para reproducibilidad en los resultados.
        
        :param self: Instancia del objeto.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)

    def _adjust_labels(self, y):
        """
        Ajusta las etiquetas para que sean compatibles con las funciones de pérdida:
        - Si son strings, se usa LabelEncoder para convertirlas a enteros 0..C-1.
        - Si son numéricas, ya vienen convertidas a 0-index.
        
        :param self: Instancia del objeto.
        :param y: Array de etiquetas originales.
        
        :returns: Array de etiquetas ajustadas (enteros 0..C-1).
        """
        y = np.asarray(y)
        if y.dtype == object:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            if self.output_dim == 1:
                self.output_dim = len(self.label_encoder.classes_)

        return y.astype(np.int32)
    
    def _representative_data_gen(self, X):
        """
        Generador de datos representativos para la calibración de la cuantización INT8.
        Toma hasta 100 muestras de X y las devuelve en lotes de una muestra.

        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        
        :yields: Lote de una muestra con forma (1, n_features) y dtype float32.
        """
        X = np.asarray(X, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(1)
        for input_value in dataset.take(100):
            yield [input_value.numpy()]
            # yield [input_value]

    def _build_model(self):
        """
        Construye el modelo secuencial de Keras.
        - Capas densas con activación ReLU entre ellas.
        - Capa de salida con activación sigmoide (binaria) o sin activación (multiclase).
        
        :param self: Instancia del objeto.

        :returns: Modelo Keras (no compilado).
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        for h in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(h, activation="relu"))
            
        if self.output_dim == 1:
            model.add(tf.keras.layers.Dense(1))
        else:
            model.add(tf.keras.layers.Dense(self.output_dim))

        return model

    # --------------
    # ENTRENAMIENTO
    # --------------
    def fit(self, X, y):
        """
        Entrena el modelo con los datos proporcionados.
        1. Fijar semillas para reproducibilidad.
        2. Escalar características y ajustar etiquetas.
        3. Construir y compilar el modelo Keras.
        4. Entrenar el modelo y medir el tiempo.
        5. Convertir a TFLite usando el conversor de TensorFlow, aplicando cuantización INT8.
        6. Guardar el modelo .tflite en un archivo temporal y medir su tamaño.
        7. Cargar el intérprete y obtener detalles de entrada/salida.

        :param self: Instancia del objeto.
        :param X: Características de entrenamiento (numpy array).
        :param y: Etiquetas de entrenamiento (numpy array).
        
        :returns: Self (objeto entrenado).
        """
        self._set_seeds()

        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        y_adj = self._adjust_labels(y)

        self.model_ = self._build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', 
                patience=self.n_iter_no_change, 
                min_delta=self.tol, 
                restore_best_weights=True,
                verbose=0
            )
        ]

        start_train = time.perf_counter()
        
        if self.teacher_model is not None:
            print(f"\n[Knowledge Distillation] 1/3 Entrenando modelo profesor ({type(self.teacher_model).__name__})...")
            # 1. Ejecución del profesor fuera de Keras (Universal para Scikit-Learn)
            self.teacher_ = clone(self.teacher_model)
            self.teacher_.fit(X, y)
            
            print("[Knowledge Distillation] 2/3 Generando predicciones blandas del profesor sobre el dataset...")
            effective_temp = self.temperature
            
            if hasattr(self.teacher_, "predict_logits"):
                logits = self.teacher_.predict_logits(X)
                if self.output_dim == 1:
                    y_teacher = tf.nn.sigmoid(logits / effective_temp).numpy().flatten()
                else:
                    y_teacher = tf.nn.softmax(logits / effective_temp).numpy()
            elif hasattr(self.teacher_, "predict_proba"):
                if self.temperature != 1.0:
                    print("    Advertencia: El profesor no expone 'predict_logits'. La temperatura se ignorará.")
                    effective_temp = 1.0
                y_teacher = self.teacher_.predict_proba(X)
                if self.output_dim == 1 and y_teacher.shape[1] == 2:
                    y_teacher = y_teacher[:, 1]
            else:
                if self.temperature != 1.0:
                    print("    Advertencia: El profesor solo expone 'predict'. La temperatura efectiva se ajustará a 1.0.")
                    effective_temp = 1.0
                y_teacher = self.teacher_.predict(X)
                if self.output_dim > 1:
                    y_teacher = tf.keras.utils.to_categorical(y_teacher, num_classes=self.output_dim)

            y_teacher = np.asarray(y_teacher, dtype=np.float32)
            
            # 2. Empaquetar y entrenar
            train_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, (y_adj, y_teacher)))
            train_dataset = train_dataset.shuffle(len(X_scaled), seed=self.random_state).batch(self.batch_size)

            print(f"[Knowledge Distillation] 3/3 Entrenando modelo estudiante (Alpha: {self.alpha})...")
            kd_model = KDKerasClassifier(self.model_, alpha=self.alpha, temperature=effective_temp, is_binary=(self.output_dim == 1))
            kd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=self.weight_decay))
            
            kd_model.fit(train_dataset, epochs=self.epochs, verbose=1, callbacks=callbacks)
            print("[Knowledge Distillation] Proceso completado exitosamente.\n")
        else:
            if self.output_dim == 1:
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            self.model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=self.weight_decay), loss=loss_fn)
            self.model_.fit(X_scaled, y_adj, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=callbacks)
        
        self.train_time_ = (time.perf_counter() - start_train)
        
        # Conversión a TFLite
        start_conv = time.perf_counter()
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model_)
        
        if self.quantization_mode == 'static':
            # Cuantización INT8 completa (estática) con calibración
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: self._representative_data_gen(X_scaled)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif self.quantization_mode == 'dynamic':
            # Cuantización dinámica – sin calibración ni restricciones de tipos
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        self.conversion_time_ = (time.perf_counter() - start_conv)

        fd, self._model_path = tempfile.mkstemp(suffix=".tflite")
        os.close(fd)
        with open(self._model_path, "wb") as f:
            f.write(tflite_model)
            
        self.model_size_kb_ = (os.path.getsize(self._model_path) / 1024)
        
        self.interpreter = Interpreter(model_path=self._model_path)
        self.interpreter.allocate_tensors()
        self.input_details = (self.interpreter.get_input_details())
        self.output_details = (self.interpreter.get_output_details())

        return self
    
    # -----------
    # INFERENCIA
    # -----------
    def predict_logits(self, X):
        """
        Devuelve los logits (salida sin activación) del modelo.
        
        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        
        :returns: Array de logits (n_samples, output_dim).
        """
        X_scaled = self.scaler.transform(X).astype(np.float32)

        input_details = self.input_details[0]
        output_details = self.output_details[0]
        
        if self.quantization_mode == 'static':
            scale, zero_point = input_details["quantization"]
            batch = np.round(X_scaled / scale + zero_point).astype(np.int8)
        elif self.quantization_mode in ('dynamic', 'none', None):
            batch = X_scaled
        else:
            raise ValueError(f"Modo de cuantización no soportado: '{self.quantization_mode}'")

        input_index = input_details["index"]
        
        self.interpreter.resize_tensor_input(input_index, batch.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(input_index, batch)
        self.interpreter.invoke()

        outputs = self.interpreter.get_tensor(output_details["index"])
        
        # Decuantizar salida si es necesario
        if self.quantization_mode == 'static':
            out_scale, out_zero = output_details["quantization"]
            outputs = (outputs.astype(np.float32) - out_zero) * out_scale
            
        return outputs
    
    def predict_proba(self, X):
        """
        Devuelve las probabilidades de clase a partir de los logits.
        - Para clasificación binaria, aplica la función sigmoide.
        - Para clasificación multiclase, aplica la función softmax.
        
        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        
        :returns: Array de probabilidades (n_samples, n_classes).
        """
        logits = self.predict_logits(X)
        
        if self.output_dim == 1:
            # Sigmoide
            probs = 1.0 / (1.0 + np.exp(-logits))
            return np.hstack([1.0 - probs, probs])
        else:
            # Softmax
            e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        """
        Devuelve las predicciones de clase a partir de los logits.
        - Para clasificación binaria, aplica un umbral de 0.5.
        - Para clasificación multiclase, aplica argmax.
        
        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        
        :returns: Array de predicciones de clase (n_samples,).
        """
        logits = self.predict_logits(X)

        # Postprocesado según tarea
        if self.output_dim == 1:
            # Clasificación binaria: umbral en 0.5
            # Si el logit > 0, equivale a Sigmoide > 0.5
            preds = (logits > 0.0).astype(int).flatten()
        else:
            # Clasificación multiclase: argmax
            preds = np.argmax(logits, axis=1)

        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds

    # -----------
    # DESTRUCTOR
    # -----------
    def __del__(self):
        """
        Destructor: Elimina el archivo temporal que contiene el modelo para liberar recursos del sistema de archivos.
        
        :param self: Instancia del objeto.
        """
        model_path = getattr(self, '_model_path', None)
        if (model_path and os.path.exists(model_path)):
            try:
                os.remove(model_path)
            except:
                pass
            

# ==========
# REGRESIÓN
# ==========
class TFLiteModelWrapperRegression(BaseEstimator):
    """
    Wrapper TensorFlow Lite (LiteRT) (regresión).
    - Entrenamiento de un MLP con Keras
    - Conversión a formato TFLite con cuantización INT8
    - Inferencia mediante el intérprete de LiteRT.
    """
    # ------------
    # CONSTRUCTOR
    # ------------
    def __init__(self, input_dim, hidden_sizes=[32, 16], output_dim=1, epochs=20, lr=0.01, weight_decay=0.0001, 
                 batch_size=64, tol=0.001, n_iter_no_change=10, random_state=42, quantization_mode='static',
                 teacher_model=None, alpha=0.5):
        """
        Constructor.

        :param self: Instancia del objeto.
        :param input_dim: Número de características de entrada.
        :param hidden_sizes: Lista con el número de neuronas en cada capa oculta.
        :param output_dim: Dimensión de salida.
        :param epochs: Número de épocas de entrenamiento.
        :param lr: Tasa de aprendizaje (learning rate) para el optimizador Adam.
        :param weight_decay: Regularización L2 para el optimizador Adam.
        :param batch_size: Tamaño del lote durante el entrenamiento.
        :param tol: Tolerancia para detener el entrenamiento.
        :param n_iter_no_change: Número de iteraciones sin mejora antes de detener el entrenamiento.
        :param random_state: Semilla para reproducibilidad.
        :param quantization_mode: Modo de cuantización ('static' o 'dynamic').
        :param teacher_model: Modelo profesor para destilación de conocimiento.
        :param alpha: Peso de la pérdida de destilación de conocimiento.
        """
        self.input_dim = input_dim                  # Número de características de entrada
        self.hidden_sizes = hidden_sizes            # Lista con el número de neuronas en cada capa oculta
        self.output_dim = output_dim                # Dimensión de salida (1 para regresión)
        self.epochs = epochs                        # Número de épocas de entrenamiento
        self.lr = lr                                # Tasa de aprendizaje para el optimizador Adam
        self.weight_decay = weight_decay            # Regularización L2 para el optimizador Adam
        self.batch_size = batch_size                # Tamaño del lote para entrenamiento
        self.tol = tol                              # Tolerancia para detener el entrenamiento
        self.n_iter_no_change = n_iter_no_change    # Número de iteraciones sin mejora antes de detener el entrenamiento
        self.random_state = random_state            # Semilla para reproducibilidad
        self.quantization_mode = quantization_mode  # 'static' o 'dynamic'
        self.teacher_model = teacher_model          # Modelo profesor para destilación de conocimiento
        self.alpha = alpha                          # Peso de la pérdida de destilación de conocimiento
        self.scaler_X = StandardScaler()	        # Estandarizador de características
        self.scaler_y = StandardScaler()	        # Estandarizador de variable objetivo
        self.model_size_kb_ = None			        # Tamaño del modelo (KB)
        self.train_time_ = None				        # Tiempo de entrenamiento (s)
        self.conversion_time_ = None		        # Tiempo de conversión a TFLite (s)
        self._model_path = None				        # Ruta al archivo .tflite temporal

    # -----------
    # UTILIDADES
    # -----------
    def _set_seeds(self):
        """
        Fija las semillas de las librerías aleatorias para reproducibilidad en los resultados.
        
        :param self: Instancia del objeto.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
        
    def _representative_data_gen(self, X):
        """
        Generador de datos representativos para la calibración de la cuantización INT8.
        Toma hasta 100 muestras de X y las devuelve en lotes de una muestra.

        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        
        :yields: Lote de una muestra con forma (1, n_features) y dtype float32.
        """
        X = np.asarray(X, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(1)
        for input_value in dataset.take(100):
            yield [input_value.numpy()]
            # yield [input_value]

    def _build_model(self):
        """
        Construye el modelo secuencial de Keras.
        - Capas densas con activación ReLU entre ellas.
        - Capa de salida sin activación.
        
        :param self: Instancia del objeto.

        :returns: Modelo Keras (no compilado).
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        for h in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(h, activation="relu"))

        model.add(tf.keras.layers.Dense(1))

        return model

    # --------------
    # ENTRENAMIENTO
    # --------------
    def fit(self, X, y):
        """
        Entrena el modelo con los datos proporcionados.
        1. Fijar semillas para reproducibilidad.
        2. Escalar características (X) y variable objetivo (y).
        3. Construir y compilar el modelo Keras.
        4. Entrenar el modelo y medir el tiempo.
        5. Convertir a TensorFlow Lite usando el conversor de TensorFlow, aplicando cuantización.
        6. Guardar el modelo en un archivo temporal y medir su tamaño.
        7. Cargar el intérprete y obtener detalles de entrada/salida.

        :param self: Instancia del objeto.
        :param X: Características de entrenamiento (numpy array).
        :param y: Etiquetas de entrenamiento (numpy array).
        
        :returns: Self (objeto entrenado).
        """
        self._set_seeds()

        X_scaled = self.scaler_X.fit_transform(X).astype(np.float32)
        y_scaled = (self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()).astype(np.float32)

        self.model_ = self._build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', 
                patience=self.n_iter_no_change, 
                min_delta=self.tol, 
                restore_best_weights=True, 
                verbose=0
            )
        ]

        start_train = time.perf_counter()
        
        if self.teacher_model is not None:
            print(f"\n[Knowledge Distillation] 1/3 Entrenando modelo profesor ({type(self.teacher_model).__name__})...")
            self.teacher_ = clone(self.teacher_model)
            self.teacher_.fit(X, y)
            
            print("[Knowledge Distillation] 2/3 Generando predicciones blandas del profesor sobre el dataset...")
            y_teacher = self.teacher_.predict(X)
            y_teacher_scaled = self.scaler_y.transform(y_teacher.reshape(-1, 1)).ravel().astype(np.float32)

            train_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, (y_scaled, y_teacher_scaled)))
            train_dataset = train_dataset.shuffle(len(X_scaled), seed=self.random_state).batch(self.batch_size)

            print(f"[Knowledge Distillation] 3/3 Entrenando modelo estudiante (Alpha: {self.alpha})...")
            kd_model = KDKerasRegressor(self.model_, alpha=self.alpha)
            kd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=self.weight_decay))
            
            kd_model.fit(train_dataset, epochs=self.epochs, verbose=1, callbacks=callbacks)
            print("[Knowledge Distillation] Proceso completado exitosamente.\n")
        else:
            self.model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=self.weight_decay), loss="mse")
            self.model_.fit(X_scaled, y_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=callbacks)
        
        self.train_time_ = (time.perf_counter() - start_train)

        start_conv = time.perf_counter()
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model_)
        
        if self.quantization_mode == 'static':
            # Cuantización INT8 completa (estática) con calibración
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: self._representative_data_gen(X_scaled)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif self.quantization_mode == 'dynamic':
            # Cuantización dinámica – sin calibración ni restricciones de tipos
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        self.conversion_time_ = (time.perf_counter() - start_conv)

        fd, self._model_path = tempfile.mkstemp(suffix=".tflite")
        os.close(fd)
        with open(self._model_path, "wb") as f:
            f.write(tflite_model)

        self.model_size_kb_ = (os.path.getsize(self._model_path) / 1024)

        self.interpreter = Interpreter(model_path=self._model_path)
        self.interpreter.allocate_tensors()
        self.input_details = (self.interpreter.get_input_details())
        self.output_details = (self.interpreter.get_output_details())

        return self

    # -----------
    # INFERENCIA
    # -----------
    def predict(self, X):
        """
        Infiere usando el intérprete de TFLite.
        1. Escalar las características usando el scaler ya ajustado.
        2. Cuantizar manualmente la entrada a INT8 (usando escala y punto cero).
        3. Ejecutar el intérprete.
        4. Decuantizar la salida (si es necesario).
        5. Desescalar la salida para obtener el valor en la escala original.

        :param self: Instancia del objeto.
        :param X: Características de entrada (numpy array).
        
        :returns: Array de predicciones (numpy array).
        """
        X_scaled = self.scaler_X.transform(X).astype(np.float32)
        
        input_details = self.input_details[0]
        output_details = self.output_details[0]
        
        if self.quantization_mode == 'static':
            # Cuantizar entrada a INT8
            scale, zero_point = input_details["quantization"]
            batch = np.round(X_scaled / scale + zero_point).astype(np.int8)
        elif self.quantization_mode in ('dynamic', 'none', None):
            batch = X_scaled
        else:
            raise ValueError(f"Modo de cuantización no soportado: '{self.quantization_mode}'")
        
        input_index = input_details["index"]
        
        self.interpreter.resize_tensor_input(input_index, batch.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(input_index, batch)
        self.interpreter.invoke()
        
        outputs = self.interpreter.get_tensor(output_details["index"])
        
        # Decuantizar salida si es necesario
        if self.quantization_mode == 'static':
            # int8 -> float
            out_scale, out_zero = output_details["quantization"]
            outputs = (outputs.astype(np.float32) - out_zero) * out_scale
        # elif self.quantization_mode == 'dynamic':
        #     outputs = outputs.astype(np.float32)

        # Desescalar a la escala original de y
        preds = self.scaler_y.inverse_transform(outputs.reshape(-1, 1)).ravel()
        return preds
    
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

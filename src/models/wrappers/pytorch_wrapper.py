import copy
import numpy as np
import os
import random
import tempfile
import time
import torch
import torch.ao.quantization as quant
import torch.ao.quantization.quantize_fx as quant_fx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
from torch.utils.data import DataLoader, TensorDataset


# ===============
# CLASIFICACIÓN
# ===============
class TorchScriptModelWrapperClassification(BaseEstimator):
    """
    Wrapper PyTorch (clasificación).
    - Entrenamiento de un MLP con PyTorch.
    - Cuantización INT8.
    - Exportación a TorchScript para inferencia sin Python.
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
        :param hidden_sizes: Lista con el número de neuronas de cada capa oculta.
        :param output_dim: Número de clases (1 para binario, >1 para multiclase).
        :param epochs: Número de épocas de entrenamiento.
        :param lr: Tasa de aprendizaje.
        :param weight_decay: Regularización L2 para el optimizador Adam.
        :param batch_size: Tamaño del lote para entrenamiento.
        :param tol: Tolerancia para detener el entrenamiento.
        :param n_iter_no_change: Número de iteraciones sin mejora antes de detener el entrenamiento.
        :param random_state: Semilla para reproducibilidad.
        :param quantization_mode: Modo de cuantización ('dynamic' o 'static').
        :param teacher_model: Modelo profesor para destilación de conocimiento.
        :param alpha: Peso de la pérdida de destilación de conocimiento.
        :param temperature: Temperatura para suavizar las predicciones del profesor.
        """
        self.input_dim = input_dim                  # Número de características de entrada
        self.hidden_sizes = hidden_sizes            # Lista con el número de neuronas de cada capa oculta
        self.output_dim = output_dim                # Número de clases (1 para binario, >1 para multiclase)
        self.epochs = epochs                        # Número de épocas de entrenamiento
        self.lr = lr                                # Tasa de aprendizaje para el optimizador Adam
        self.weight_decay = weight_decay            # Regularización L2 para el optimizador Adam
        self.batch_size = batch_size                # Tamaño del lote para entrenamiento
        self.tol = tol                              # Tolerancia para detener el entrenamiento
        self.n_iter_no_change = n_iter_no_change    # Número de iteraciones sin mejora antes de detener el entrenamiento
        self.random_state = random_state            # Semilla para reproducibilidad
        self.quantization_mode = quantization_mode  # 'dynamic' o 'static'
        self.teacher_model = teacher_model          # Modelo profesor para destilación de conocimiento
        self.alpha = alpha                          # Peso de la pérdida de destilación de conocimiento
        self.temperature = temperature              # Temperatura para suavizar las predicciones del profesor
        self.scaler = StandardScaler()              # Estandarizador de características
        self.label_encoder = None                   # Codificador de etiquetas
        self.model_size_kb_ = None                  # Tamaño del modelo (KB)
        self.train_time_ = None                     # Tiempo de entrenamiento (s)
        self.conversion_time_ = None                # Tiempo de conversión a TorchScript (s)
        self._model_path = None                     # Ruta al archivo temporal .pt

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
            torch.manual_seed(self.random_state)

    def _build_model(self):
        """
        Construye el modelo MLP:
        - Capas lineales con activación ReLU entre ellas.
        - Capa de salida lineal.
        
        :param self: Instancia del objeto.
        
        :returns: nn.Sequential con la arquitectura del modelo.
        """
        layers = []
        prev = self.input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        return nn.Sequential(*layers)

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
        if y.dtype == object:   # Etiquetas textuales
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
            # Actualizar output_dim si aún no se había establecido correctamente
            if self.output_dim == 1:
                self.output_dim = len(self.label_encoder.classes_)
            return y_enc.astype(np.int64)
        else:
            # Actualizar output_dim si las etiquetas ya son numéricas pero > 2 clases
            num_classes = len(np.unique(y))
            if self.output_dim == 1 and num_classes > 2:
                self.output_dim = num_classes
            return y.astype(np.int64)

    # --------------
    # ENTRENAMIENTO
    # --------------
    def fit(self, X, y):
        """
        Entrena el modelo con los datos proporcionados.
        1. Fijar semillas para reproducibilidad.
        2. Estandarizar características (X) y ajustar las etiquetas (y) a formato numérico 0..C-1.
        3. Preparar DataLoader con mini-batches.
        4. Construir la red neuronal y el optimizador Adam.
        5. Ejecutar el bucle de entrenamiento.
        6. Aplicar cuantización.
        7. Conviertir el modelo a TorchScript y optimizar para inferencia.
        8. Guardar el modelo en un archivo temporal y registrar su tamaño.
        
        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        :param y: Array de etiquetas (n_samples,).
        
        :returns: Self (instancia entrenada).
        """        
        # 1. Fijar semillas
        self._set_seeds()

        # 2. Estandarizar características (media 0, desviación 1) y ajustar etiquetas
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        y_adj = self._adjust_labels(y)

        # 3. Crear dataset y DataLoader con shuffle reproducible
        effective_temp = self.temperature
        
        if self.teacher_model is not None:
            print(f"\n[Knowledge Distillation] 1/3 Entrenando modelo profesor ({type(self.teacher_model).__name__})...")
            self.teacher_ = clone(self.teacher_model)
            self.teacher_.fit(X, y)
            
            print("[Knowledge Distillation] 2/3 Generando predicciones blandas del profesor sobre el dataset...")
            if hasattr(self.teacher_, "predict_logits"):
                # logits = torch.tensor(self.teacher_.predict_logits(X))
                logits = torch.from_numpy(self.teacher_.predict_logits(X).astype(np.float32))
                if self.output_dim == 1:
                    y_teacher = torch.sigmoid(logits / effective_temp).numpy().flatten()
                else:
                    y_teacher = F.softmax(logits / effective_temp, dim=1).numpy()
            elif hasattr(self.teacher_, "predict_proba"):
                if self.temperature != 1.0:
                    print("    Advertencia: El profesor no expone 'predict_logits'. La temperatura asume logits, por lo que se ignorará.")
                    effective_temp = 1.0    # Anulamos la temperatura
                y_teacher = self.teacher_.predict_proba(X)
                if self.output_dim == 1 and y_teacher.shape[1] == 2:
                    y_teacher = y_teacher[:, 1]
            else:
                if self.temperature != 1.0:
                    print("    Advertencia: El profesor solo expone 'predict'. La temperatura efectiva se ajustará a 1.0.")
                    effective_temp = 1.0    # Anulamos la temperatura
                y_teacher = self.teacher_.predict(X)
                if self.output_dim > 1:
                    y_teacher = np.eye(self.output_dim)[y_teacher.astype(int)]
                    
            y_teacher = np.asarray(y_teacher, dtype=np.float32)
            
            dataset = TensorDataset(
                torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_adj, dtype=torch.long if self.output_dim > 1 else torch.float32),
                torch.tensor(y_teacher, dtype=torch.float32)
            )
            print(f"[Knowledge Distillation] 3/3 Entrenando modelo estudiante (Alpha: {self.alpha})...")
        else:
            dataset = TensorDataset(
                torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_adj, dtype=torch.long if self.output_dim > 1 else torch.float32)
            )
        
        # Generador de números aleatorios para el DataLoader
        g = torch.Generator()
        if self.random_state is not None:
            g.manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)

        # Elegir función de pérdida según el tipo de problema
        if self.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss()  # Clasificación binaria
        else:
            criterion = nn.CrossEntropyLoss()   # Clasificación multiclase

        # 4. Construir el modelo y el optimizador
        self.model_ = self._build_model()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        best_loss = float('inf')
        wait = 0
        loss_curve = []
        best_model_state = None

        # 5. Entrenamiento
        start_train = time.perf_counter()
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in loader:
                optimizer.zero_grad()   # Reiniciar gradientes
                
                # Desempaquetado dinámico según si hay KD o no
                if len(batch) == 3:
                    batch_X, batch_y, batch_teacher = batch
                    outputs = self.model_(batch_X)  # Forward pass
                    
                    if self.output_dim == 1:
                        # Para binario, PyTorch requiere exactamente las mismas dimensiones (N, 1)
                        student_loss = criterion(outputs, batch_y.view(-1, 1))
                        dist_loss = F.binary_cross_entropy_with_logits(outputs / effective_temp, batch_teacher.view(-1, 1))
                        # dist_loss = F.binary_cross_entropy_with_logits(outputs, batch_teacher.view(-1, 1))
                    else:
                        student_loss = criterion(outputs, batch_y)
                        # KL Divergence para multiclase: pred de estudiante en log-space vs prob del profesor
                        log_probs = F.log_softmax(outputs / effective_temp, dim=1)
                        # log_probs = F.log_softmax(outputs, dim=1)
                        dist_loss = F.kl_div(log_probs, batch_teacher, reduction='batchmean')
                        
                    loss = (self.alpha * student_loss) + ((1.0 - self.alpha) * dist_loss * (effective_temp ** 2))
                else:
                    batch_X, batch_y = batch
                    outputs = self.model_(batch_X)  # Forward pass
                    if self.output_dim == 1:
                        loss = criterion(outputs, batch_y.view(-1, 1))
                    else:
                        loss = criterion(outputs, batch_y)

                loss.backward()                     # Backward pass
                optimizer.step()                    # Actualizar pesos
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_epoch_loss = epoch_loss / num_batches
            loss_curve.append(avg_epoch_loss)
            
            # Verificar mejora según la tolerancia
            if best_loss - avg_epoch_loss > self.tol:
                best_loss = avg_epoch_loss
                wait = 0
                best_model_state = copy.deepcopy(self.model_.state_dict())
            else:
                wait += 1
                if wait >= self.n_iter_no_change:
                    print(f"    Detenido en época {epoch+1} (sin mejora de {self.tol} durante {self.n_iter_no_change} épocas)")
                    break

        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
        
        self.train_time_ = time.perf_counter() - start_train
        
        if self.teacher_model is not None:
            print("[Knowledge Distillation] Proceso completado exitosamente.\n")

        self.model_.eval()
        
        # Dummy input necesario para trazar el grafo en la nueva API FX y TorchScript
        example_inputs = (torch.randn(1, self.input_dim),)

        # Cuantización
        if self.quantization_mode == 'dynamic':
            # Cuantización dinámica con FX: solo las capas lineales se cuantizan a INT8 durante la inferencia
            # Configurar el backend
            backend = "x86"
            torch.backends.quantized.engine = backend
            qconfig_mapping = QConfigMapping().set_global(quant.default_dynamic_qconfig)
            
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            self.model_ = quant_fx.convert_fx(model_prepared)
        elif self.quantization_mode == 'static':
            # Cuantización estática con FX: se requiere calibración con datos representativos
            # (La fusión de Linear + ReLU y stubs es automática)
            # backend = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
            backend = "x86"
            torch.backends.quantized.engine = backend
            qconfig_mapping = get_default_qconfig_mapping(backend)
            
            # Preparar modelo
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            
            # Calibración (pasar datos representativos)
            with torch.no_grad():
                for batch in loader:
                    model_prepared(batch[0])
                    
            # Convertir a modelo cuantizado INT8 completo
            self.model_ = quant_fx.convert_fx(model_prepared)

        # 6. Convertir a TorchScript y optimizar para inferencia
        start_conv = time.perf_counter()
        # Los módulos FX se exportan mejor con 'trace' que con 'script'
        scripted_model = torch.jit.trace(self.model_, example_inputs[0])
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        self.conversion_time_ = time.perf_counter() - start_conv

		# 7. Guardar el modelo TorchScript en un archivo temporal
        fd, self._model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.jit.save(scripted_model, self._model_path)
        self.scripted_model_ = scripted_model

		# 8. Registrar el tamaño del archivo (KB)
        self.model_size_kb_ = os.path.getsize(self._model_path) / 1024

        return self

    # -----------
    # INFERENCIA
    # -----------
    def predict_logits(self, X):
        """
        Infiere logits sobre nuevas muestras.
        
        :param self: Instancia del objeto.
        :param X: Array de características de entrada (n_samples, n_features).
        
        :returns: Array de logits (n_samples, output_dim).
        """
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_scaled)
        # X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self.scripted_model_.eval()
        with torch.no_grad():
            return self.scripted_model_(X_t).numpy()
        
    def predict_proba(self, X):
        """
        Infiere probabilidades de clase sobre nuevas muestras.
        
        :param self: Instancia del objeto.
        :param X: Array de características de entrada (n_samples, n_features).
        
        :returns: Array de probabilidades (n_samples, n_classes).
        """
        # logits = torch.tensor(self.predict_logits(X))
        logits = torch.from_numpy(self.predict_logits(X))
        if self.output_dim == 1:
            probs = torch.sigmoid(logits).numpy()
            # Se esperan 2 columnas para binario [P(0), P(1)]
            return np.hstack([1 - probs, probs])
        else:
            return F.softmax(logits, dim=1).numpy()
    
    def predict(self, X):
        """
        Infiere predicciones sobre nuevas muestras.
        1. Estandarizar las características usando el scaler ya entrenado y convertir a tensor de PyTorch.
        2. Ejecutar el modelo TorchScript en modo evaluación.
        3. Conviertir las salidas a etiquetas de clase (binarias o multiclase).
        
        :param self: Instancia del objeto.
        :param X: Array de características de entrada (n_samples, n_features).
        
        :returns: Array de predicciones (n_samples,).
        """
        # 1. Estandarizar características y convertir a tensor
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_scaled)
        # X_t = torch.tensor(X_scaled, dtype=torch.float32)
        
        # 2. Inferencia
        self.scripted_model_.eval()
        
        # 3. Convertir salidas a etiquetas de clase
        with torch.no_grad():
            outputs = self.scripted_model_(X_t)
            if self.output_dim == 1:
                # Clasificación binaria: aplicar sigmoide y umbral 0.5
                preds = (torch.sigmoid(outputs).numpy() > 0.5).astype(int).flatten()
                if self.label_encoder is not None:
                    return self.label_encoder.inverse_transform(preds)
                return preds
            else:
                # Clasificación multiclase: elegir la clase con mayor puntuación
                preds = torch.argmax(outputs, dim=1).numpy()
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
class TorchScriptModelWrapperRegression(BaseEstimator):
    """
    Wrapper PyTorch (regresión).
    - Entrenamiento de un MLP con PyTorch.
    - Cuantización INT8.
    - Exportación a TorchScript para inferencia sin Python.
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
        :param hidden_sizes: Lista con el número de neuronas de cada capa oculta.
        :param output_dim: Número de clases (1 para binario, >1 para multiclase).
        :param epochs: Número de épocas de entrenamiento.
        :param lr: Tasa de aprendizaje.
        :param weight_decay: Regularización L2 para el optimizador Adam.
        :param batch_size: Tamaño del lote para entrenamiento.
        :param tol: Tolerancia para la convergencia.
        :param n_iter_no_change: Número de iteraciones sin mejora antes de detener.
        :param random_state: Semilla para reproducibilidad.
        :param quantization_mode: Modo de cuantización ('dynamic' o 'static').
        :param teacher_model: Modelo profesor para destilación de conocimiento.
        :param alpha: Peso de la pérdida de destilación de conocimiento.
        """
        self.input_dim = input_dim                  # Número de características de entrada
        self.hidden_sizes = hidden_sizes            # Lista con el número de neuronas de cada capa oculta
        self.output_dim = output_dim                # Dimensión de salida (1 para regresión)
        self.epochs = epochs                        # Número de épocas de entrenamiento
        self.lr = lr                                # Tasa de aprendizaje para el optimizador Adam
        self.weight_decay = weight_decay            # Regularización L2 para el optimizador Adam
        self.batch_size = batch_size                # Tamaño del lote para entrenamiento
        self.tol = tol                              # Tolerancia para la convergencia
        self.n_iter_no_change = n_iter_no_change    # Número de iteraciones sin mejora antes de detener
        self.random_state = random_state            # Semilla para reproducibilidad
        self.quantization_mode = quantization_mode	# 'dynamic' o 'static'
        self.teacher_model = teacher_model			# Modelo profesor para destilación de conocimiento
        self.alpha = alpha							# Peso de la pérdida de destilación de conocimiento
        self.scaler_X = StandardScaler()			# Estandarizador de características
        self.scaler_y = StandardScaler()			# Estandarizador de variable objetivo
        self.model_size_kb_ = None					# Tamaño del modelo (KB)
        self.train_time_ = None						# Tiempo de entrenamiento (s)
        self.conversion_time_ = None				# Tiempo de conversión a TorchScript (s)
        self._model_path = None						# Ruta al archivo temporal .pt

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
            torch.manual_seed(self.random_state)

    def _build_model(self):
        """
        Construye el modelo MLP para regresión:
        - Capas lineales con activación ReLU entre ellas.
        - Capa de salida lineal (sin activación, porque se usa MSELoss).
        
        :param self: Instancia del objeto.
        
        :returns: nn.Sequential con la arquitectura del modelo.
        """
        layers = []
        prev = self.input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        return nn.Sequential(*layers)

    # --------------
    # ENTRENAMIENTO
    # --------------
    def fit(self, X, y):
        """
        Entrena el modelo con los datos proporcionados.
        1. Fijar semillas para reproducibilidad.
        2. Estandarizar características (X) y variable objetivo (y).
        3. Preparar DataLoader con mini-batches.
        4. Construir la red neuronal y el optimizador Adam.
        5. Ejecutar el bucle de entrenamiento.
        6. Aplicar cuantización.
        7. Convertir el modelo a TorchScript y optimizar para inferencia.
        8. Guardar el modelo en un archivo temporal y registrar su tamaño.

        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        :param y: Array de valores objetivo (n_samples,).
        
        :returns: Self (instancia entrenada).
        """
        # 1. Fijar semillas
        self._set_seeds()

		# 2. Estandarizar características y variable objetivo (media 0, desviación 1)
        X_scaled = self.scaler_X.fit_transform(X).astype(np.float32)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

		# 3. Preparación de la destilación de conocimiento y creación de dataset/DataLoader
        if self.teacher_model is not None:
            print(f"\n[Knowledge Distillation] 1/3 Entrenando modelo profesor ({type(self.teacher_model).__name__})...")
            self.teacher_ = clone(self.teacher_model)
            self.teacher_.fit(X, y)
            
            print("[Knowledge Distillation] 2/3 Generando predicciones blandas del profesor sobre el dataset...")
            y_teacher = self.teacher_.predict(X)
            y_teacher_scaled = self.scaler_y.transform(y_teacher.reshape(-1, 1)).ravel().astype(np.float32)

            dataset = TensorDataset(
                torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_teacher_scaled, dtype=torch.float32).view(-1, 1)
            )
            print(f"[Knowledge Distillation] 3/3 Entrenando modelo estudiante (Alpha: {self.alpha})...")
        else:
            dataset = TensorDataset(
                torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)
            )
        
        # Generador de números aleatorios para el DataLoader
        g = torch.Generator()
        if self.random_state is not None:
            g.manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)

		# Función de pérdida (error cuadrático medio)
        criterion = nn.MSELoss()

		# 4. Construir modelo y optimizador
        self.model_ = self._build_model()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        best_loss = float('inf')
        wait = 0
        loss_curve = []
        best_model_state = None

		# 5. Entrenamiento
        start_train = time.perf_counter()
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in loader:
                optimizer.zero_grad()	# Reiniciar gradientes
                
                # Desempaquetado dinámico según si hay KD o no
                if len(batch) == 3:
                    batch_X, batch_y, batch_teacher = batch
                    outputs = self.model_(batch_X)  # Forward pass
                    
                    student_loss = criterion(outputs, batch_y)
                    dist_loss = criterion(outputs, batch_teacher)
                    loss = (self.alpha * student_loss) + ((1.0 - self.alpha) * dist_loss)
                else:
                    batch_X, batch_y = batch
                    outputs = self.model_(batch_X)  # Forward pass
                    loss = criterion(outputs, batch_y)

                loss.backward()                     # Backward pass
                optimizer.step()                    # Actualizar pesos
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_epoch_loss = epoch_loss / num_batches
            loss_curve.append(avg_epoch_loss)
            
            # Verificar mejora según la tolerancia
            if best_loss - avg_epoch_loss > self.tol:
                best_loss = avg_epoch_loss
                wait = 0
                best_model_state = copy.deepcopy(self.model_.state_dict())
            else:
                wait += 1
                if wait >= self.n_iter_no_change:
                    print(f"    Detenido en época {epoch+1} (sin mejora de {self.tol} durante {self.n_iter_no_change} épocas)")
                    break
            
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
        
        self.train_time_ = time.perf_counter() - start_train
        
        if self.teacher_model is not None:
            print("[Knowledge Distillation] Proceso completado exitosamente.\n")

        self.model_.eval()
        
        # Dummy input necesario para trazar el grafo en la nueva API FX y TorchScript
        example_inputs = (torch.randn(1, self.input_dim),)

        # Cuantización
        if self.quantization_mode == 'dynamic':
            # Cuantización dinámica con FX: solo las capas lineales se cuantizan a INT8 durante la inferencia
            # Configurar el backend
            backend = "x86"
            torch.backends.quantized.engine = backend
            qconfig_mapping = QConfigMapping().set_global(quant.default_dynamic_qconfig)
            
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            self.model_ = quant_fx.convert_fx(model_prepared)
        elif self.quantization_mode == 'static':
            # Cuantización estática con FX: se requiere calibración con datos representativos
            # (La fusión de Linear + ReLU y stubs es automática)
            # backend = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
            backend = "x86"
            torch.backends.quantized.engine = backend
            qconfig_mapping = get_default_qconfig_mapping(backend)
            
            # Preparar modelo
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            
            # Calibración (pasar datos representativos)
            with torch.no_grad():
                for batch in loader:
                    model_prepared(batch[0])
                    
            # Convertir a modelo cuantizado INT8 completo
            self.model_ = quant_fx.convert_fx(model_prepared)

        # 6. Convertir y exportar a TorchScript y optimizar para inferencia
        start_conv = time.perf_counter()
        # Los módulos FX se exportan mejor con 'trace' que con 'script'
        scripted_model = torch.jit.trace(self.model_, example_inputs[0])
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        self.conversion_time_ = time.perf_counter() - start_conv

		# 7. Guardar el modelo TorchScript en un archivo temporal
        fd, self._model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.jit.save(scripted_model, self._model_path)
        self.scripted_model_ = scripted_model

		# 8. Registrar el tamaño del archivo (KB)
        self.model_size_kb_ = os.path.getsize(self._model_path) / 1024

        return self

    # -----------
    # INFERENCIA
    # -----------
    def predict(self, X):
        """
        Infiere predicciones sobre nuevas muestras.
        1. Estandarizar las características usando el scaler ya entrenado y convertir a tensor de PyTorch.
        2. Ejecutar el modelo TorchScript en modo evaluación.
        3. Desescalar la salida usando scaler_y para obtener la predicción en la escala original.

        :param self: Instancia del objeto.
        :param X: Array de características (n_samples, n_features).
        
        :returns: Array de predicciones (n_samples,).
        """
        # 1. Estandarizar características y convertir a tensor
        X_scaled = self.scaler_X.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_scaled)
        # X_t = torch.tensor(X_scaled, dtype=torch.float32)

		# 2. Inferencia
        self.scripted_model_.eval()
        
        # 3. Desescalar la predicción (inversa de la normalización aplicada a y)
        with torch.no_grad():
            pred_scaled = self.scripted_model_(X_t).numpy().flatten()
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            return pred

    # -----------
    # DESTRUCTOR
    # -----------
    def __del__(self):
        """
        Destructor: elimina el archivo temporal que contiene el modelo para liberar recursos del sistema de archivos.
        
        :param self: Instancia del objeto.
        """
        model_path = getattr(self, '_model_path', None)
        if (model_path is not None and os.path.exists(model_path)):
            try:
                os.remove(model_path)
            except:
                pass

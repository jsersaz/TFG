import numpy as np
import os
import random
import tempfile
import time
import torch
import torch.ao.quantization as quant
import torch.ao.quantization.quantize_fx as quant_fx
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator
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
    def __init__(self, input_dim, hidden_sizes=[8], output_dim=1, epochs=20, lr=0.01,
                 batch_size=64, random_state=42, quantization_mode='dynamic'):
        """
        Constructor.
        
        :param self: instancia del objeto.
        :param input_dim: número de características de entrada.
        :param hidden_sizes: lista con el número de neuronas de cada capa oculta.
        :param output_dim: número de clases (1 para binario, >1 para multiclase).
        :param epochs: número de épocas de entrenamiento.
        :param lr: tasa de aprendizaje.
        :param batch_size: tamaño del lote para entrenamiento.
        :param random_state: semilla para reproducibilidad.
        :param quantization_mode: modo de cuantización ('dynamic' o 'static').
        """
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.quantization_mode = quantization_mode  # 'dynamic' o 'static'
        self.scaler = StandardScaler()              # Estandarizador de características
        self.label_encoder = None                   # Codificador de etiquetas
        self.model_size_kb_ = None                  # Tamaño del modelo (KB)
        self.train_time_ = None                     # Tiempo de entrenamiento (s)
        self.conversion_time_ = None                # Tiempo de conversión a TorchScript (s)
        self._model_path = None                     # ruta al archivo temporal .pt

    def _set_seeds(self):
        """
        Fija las semillas de las librerías aleatorias para reproducibilidad en los resultados.
        
        :param self: instancia del objeto.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            # Si se dispone de GPU, también se fijan las semillas de CUDA
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """
        Construye el modelo MLP:
        - Capas lineales con activación ReLU entre ellas.
        - Capa de salida lineal.
        
        :param self: instancia del objeto.
        
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
        
        :param self: instancia del objeto.
        :param y: array de etiquetas originales.
        
        :returns: array de etiquetas ajustadas (enteros 0..C-1).
        """
        y = np.asarray(y)
        if y.dtype == object:   # Etiquetas textuales
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
            # Actualizar output_dim si aún no se había establecido correctamente
            if self.output_dim == 1:
                self.output_dim = len(self.label_encoder.classes_)
            return y_enc.astype(np.int64)
        return y.astype(np.int64)

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
        
        :param self: instancia del objeto.
        :param X: array de características (n_samples, n_features).
        :param y: array de etiquetas (n_samples,).
        
        :returns: self (instancia entrenada).
        """
        # 1. Fijar semillas
        self._set_seeds()

        # 2. Estandarizar características (media 0, desviación 1) y ajustar etiquetas
        X_scaled = self.scaler.fit_transform(X)
        y_adj = self._adjust_labels(y)

        # 3. Crear dataset y DataLoader con shuffle reproducible
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
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # 5. Entrenamiento
        start_train = time.perf_counter()
        self.model_.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()				# Reiniciar gradientes
                outputs = self.model_(batch_X)		# Forward pass
                loss = criterion(outputs, batch_y)
                loss.backward()						# Backward pass
                optimizer.step()					# Actualizar pesos
        self.train_time_ = time.perf_counter() - start_train

        self.model_.eval()
        
        # Dummy input necesario para trazar el grafo en la nueva API FX y TorchScript
        example_inputs = (torch.randn(1, self.input_dim),)

        # 6. Selección del modo de cuantización
        if self.quantization_mode == 'dynamic':
            # Cuantización dinámica con FX: solo las capas lineales se cuantizan a INT8 durante la inferencia
            qconfig_mapping = QConfigMapping().set_global(quant.default_dynamic_qconfig)
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            self.model_ = quant_fx.convert_fx(model_prepared)
        else:
            # Cuantización estática con FX: se requiere calibración con datos representativos
            # (La fusión de Linear + ReLU y stubs es automática)
            backend = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
            torch.backends.quantized.engine = backend
            qconfig_mapping = get_default_qconfig_mapping(backend)
            
            # Preparar modelo
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            
            # Calibración (pasar datos representativos)
            with torch.no_grad():
                for batch_X, _ in loader:
                    model_prepared(batch_X)
                    
            # Convertir a modelo cuantizado INT8 completo
            self.model_ = quant_fx.convert_fx(model_prepared)

        # 7. Convertir a TorchScript y optimizar para inferencia
        start_conv = time.perf_counter()
        # Los módulos FX se exportan mejor con 'trace' que con 'script'
        scripted_model = torch.jit.trace(self.model_, example_inputs[0])
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        self.conversion_time_ = time.perf_counter() - start_conv

		# 8. Guardar el modelo TorchScript en un archivo temporal
        fd, self._model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.jit.save(scripted_model, self._model_path)
        self.scripted_model_ = scripted_model

		# 9. Registrar el tamaño del archivo (KB)
        self.model_size_kb_ = os.path.getsize(self._model_path) / 1024

        return self

    def predict(self, X):
        """
        Infiere predicciones sobre nuevas muestras.
        1. Estandarizar las características usando el scaler ya entrenado y convertir a tensor de PyTorch.
        2. Ejecutar el modelo TorchScript en modo evaluación.
        3. Conviertir las salidas a etiquetas de clase (binarias o multiclase).
        4. Si se usa LabelEncoder, devolver las etiquetas originales (strings).
        
        :param self: instancia del objeto.
        :param X: array de características de entrada (n_samples, n_features).
        
        :returns: array de predicciones (n_samples,).
        """
        # 1. Estandarizar características y convertir a tensor
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.scripted_model_.eval()
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

    def __del__(self):
        """
        Destructor: elimina el archivo temporal que contiene el modelo para liberar recursos del sistema de archivos.
        
        :param self: instancia del objeto.
        """
        if (self._model_path and os.path.exists(self._model_path)):
            try:
                os.remove(self._model_path)
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
    def __init__(self, input_dim, hidden_sizes=[8], output_dim=1, epochs=20, lr=0.01,
                 batch_size=64, random_state=42, quantization_mode='dynamic'):
        """
        Constructor.
        
        :param self: instancia del objeto.
        :param input_dim: número de características de entrada.
        :param hidden_sizes: lista con el número de neuronas de cada capa oculta.
        :param output_dim: número de clases (1 para binario, >1 para multiclase).
        :param epochs: número de épocas de entrenamiento.
        :param lr: tasa de aprendizaje.
        :param batch_size: tamaño del lote para entrenamiento.
        :param random_state: semilla para reproducibilidad.
        :param quantization_mode: modo de cuantización ('dynamic' o 'static').
        """
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.quantization_mode = quantization_mode	# 'dynamic' o 'static'
        self.scaler_X = StandardScaler()			# Estandarizador de características
        self.scaler_y = StandardScaler()			# Estandarizador de variable objetivo
        self.model_size_kb_ = None					# Tamaño del modelo (KB)
        self.train_time_ = None						# Tiempo de entrenamiento (s)
        self.conversion_time_ = None				# Tiempo de conversión a TorchScript (s)
        self._model_path = None						# Ruta al archivo temporal .pt

    def _set_seeds(self):
        """
        Fija las semillas de las librerías aleatorias para reproducibilidad en los resultados.
        
        :param self: instancia del objeto.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            # Si se dispone de GPU, también se fijan las semillas de CUDA
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """
        Construye el modelo MLP para regresión:
        - Capas lineales con activación ReLU entre ellas.
        - Capa de salida lineal (sin activación, porque se usa MSELoss).
        
        :param self: instancia del objeto.
        
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

        :param self: instancia del objeto.
        :param X: array de características (n_samples, n_features).
        :param y: array de valores objetivo (n_samples,).
        
        :returns: self (instancia entrenada).
        """
        # 1. Fijar semillas
        self._set_seeds()

		# 2. Estandarizar características y variable objetivo (media 0, desviación 1)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

		# 3. Crear dataset y DataLoader
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
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

		# 5. Entrenamiento
        start_train = time.perf_counter()
        self.model_.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()				# Reiniciar gradientes
                outputs = self.model_(batch_X)		# Forward pass
                loss = criterion(outputs, batch_y)
                loss.backward()						# Backward pass
                optimizer.step()					# Actualizar pesos
        self.train_time_ = time.perf_counter() - start_train

        self.model_.eval()
        
        # Dummy input necesario para trazar el grafo en la nueva API FX y TorchScript
        example_inputs = (torch.randn(1, self.input_dim),)

        # 6. Selección del modo de cuantización
        if self.quantization_mode == 'dynamic':
            # Cuantización dinámica con FX: solo las capas lineales se cuantizan a INT8 durante la inferencia
            qconfig_mapping = QConfigMapping().set_global(quant.default_dynamic_qconfig)
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            self.model_ = quant_fx.convert_fx(model_prepared)
        else:
            # Cuantización estática con FX: se requiere calibración con datos representativos
            # (La fusión de Linear + ReLU y stubs es automática)
            backend = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
            torch.backends.quantized.engine = backend
            qconfig_mapping = get_default_qconfig_mapping(backend)
            
            # Preparar modelo
            model_prepared = quant_fx.prepare_fx(self.model_, qconfig_mapping, example_inputs)
            
            # Calibración (pasar datos representativos)
            with torch.no_grad():
                for batch_X, _ in loader:
                    model_prepared(batch_X)
                    
            # Convertir a modelo cuantizado INT8 completo
            self.model_ = quant_fx.convert_fx(model_prepared)

        # # 7. Convertir y exportar a TorchScript y optimizar para inferencia
        start_conv = time.perf_counter()
        # Los módulos FX se exportan mejor con 'trace' que con 'script'
        scripted_model = torch.jit.trace(self.model_, example_inputs[0])
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        self.conversion_time_ = time.perf_counter() - start_conv

		# 8. Guardar el modelo TorchScript en un archivo temporal
        fd, self._model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.jit.save(scripted_model, self._model_path)
        self.scripted_model_ = scripted_model

		# 9. Registrar el tamaño del archivo (KB)
        self.model_size_kb_ = os.path.getsize(self._model_path) / 1024

        return self

    def predict(self, X):
        """
        Infiere predicciones sobre nuevas muestras.
        1. Estandarizar las características usando el scaler ya entrenado y convertir a tensor de PyTorch.
        2. Ejecutar el modelo TorchScript en modo evaluación.
        3. Desescalar la salida usando scaler_y para obtener la predicción en la escala original.

        :param self: instancia del objeto.
        :param X: array de características (n_samples, n_features).
        
        :returns: array de predicciones (n_samples,).
        """
        # 1. Estandarizar características y convertir a tensor
        X_scaled = self.scaler_X.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

		# 2. Inferencia
        self.scripted_model_.eval()
        with torch.no_grad():
            # 3. Desescalar la predicción (inversa de la normalización aplicada a y)
            pred_scaled = self.scripted_model_(X_t).numpy().flatten()
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            return pred

    def __del__(self):
        """
        Destructor: elimina el archivo temporal que contiene el modelo para liberar recursos del sistema de archivos.
        
        :param self: instancia del objeto.
        """
        if self._model_path and os.path.exists(self._model_path):
            os.remove(self._model_path)

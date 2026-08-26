import os
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from onnx2tf import convert
from ai_edge_litert.interpreter import Interpreter
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TorchToTFLiteModelWrapperClassification(BaseEstimator):
    """
    Wrapper que entrena un MLP en PyTorch y lo convierte a TFLite.
    Sigue la misma interfaz que los demás wrappers.
    """
    def __init__(self, input_dim, hidden_sizes=[8], output_dim=1, epochs=20, lr=0.01,
                 batch_size=64, quantize=True, random_state=42):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.quantize = quantize  # True → INT8 cuantización completa
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = None

        # Atributos para mediciones
        self.model_size_kb_ = None
        self.train_time_ = None
        self.conversion_time_ = None
        self._model_path = None  # ruta al .tflite
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self._torch_model = None  # modelo PyTorch entrenado (float)

    def _set_seeds(self):
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_torch_model(self):
        layers = []
        prev = self.input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        return nn.Sequential(*layers)

    def _adjust_labels(self, y):
        y = np.asarray(y)
        if y.dtype == object:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            if self.output_dim == 1:
                self.output_dim = len(self.label_encoder.classes_)
        return y.astype(np.int64)

    def _representative_data_gen(self, X_scaled):
        # Generador para calibración INT8
        for i in range(min(100, len(X_scaled))):
            yield [X_scaled[i:i+1].astype(np.float32)]

    def fit(self, X, y):
        # 1. Preparar datos
        self._set_seeds()
        X_scaled = self.scaler.fit_transform(X)
        y_adj = self._adjust_labels(y)

        # 2. Entrenar modelo PyTorch (float)
        self._torch_model = self._build_torch_model()
        optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_adj, dtype=torch.long if self.output_dim > 1 else torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        start_train = time.perf_counter()
        self._torch_model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self._torch_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        self.train_time_ = time.perf_counter() - start_train

        # 3. Exportar a ONNX (sin cuantización)
        self._torch_model.eval()
        dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)

        fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)

        torch.onnx.export(
            self._torch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )

        # 4. ONNX -> TensorFlow SavedModel
        saved_model_dir = tempfile.mkdtemp()
        convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
        )

        # 5. SavedModel -> TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if self.quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: self._representative_data_gen(X_scaled)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        start_conv = time.perf_counter()
        tflite_model = converter.convert()
        self.conversion_time_ = time.perf_counter() - start_conv

        # 6. Guardar .tflite y cargar intérprete
        fd_tflite, self._model_path = tempfile.mkstemp(suffix=".tflite")
        os.close(fd_tflite)
        with open(self._model_path, "wb") as f:
            f.write(tflite_model)

        self.model_size_kb_ = os.path.getsize(self._model_path) / 1024

        self.interpreter = Interpreter(model_path=self._model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Limpiar archivos temporales
        os.remove(onnx_path)
        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)

        return self

    def predict(self, X):
        if self.interpreter is None:
            raise RuntimeError("Model not fitted or conversion failed.")

        X_scaled = self.scaler.transform(X).astype(np.float32)

        input_details = self.input_details[0]
        output_details = self.output_details[0]

        # Cuantizar entrada si es necesario
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            X_quant = (X_scaled / scale + zero_point).astype(np.int8)
        else:
            X_quant = X_scaled.astype(np.float32)

        input_index = input_details['index']
        self.interpreter.resize_tensor_input(input_index, X_quant.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(input_index, X_quant)
        self.interpreter.invoke()

        outputs = self.interpreter.get_tensor(output_details['index'])

        # Decuantizar salida si es necesario
        if output_details['dtype'] == np.int8:
            out_scale, out_zero = output_details['quantization']
            outputs = (outputs.astype(np.float32) - out_zero) * out_scale

        # Postprocesado
        if self.output_dim == 1:
            preds = (outputs > 0.5).astype(int).flatten()
        else:
            preds = np.argmax(outputs, axis=1)

        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds

    def __del__(self):
        if self._model_path and os.path.exists(self._model_path):
            try:
                os.remove(self._model_path)
            except:
                pass
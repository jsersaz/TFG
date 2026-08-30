from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.models.wrappers.onnx_wrapper import ONNXModelFromSklearnRegression
from src.models.wrappers.pytorch_wrapper import TorchScriptModelWrapperRegression
from src.models.wrappers.tflite_wrapper import TFLiteModelWrapperRegression


def get_regression_models(input_dim=None, output_dim=1):
    """
    Construye y devuelve un diccionario con todos los regresores a evaluar.
    
    :param input_dim: Número de características de entrada (necesario para MLP).
    :param output_dim: Dimensión de salida (normalmente 1 para regresión univariante).
    
    :returns: Diccionario con nombre del modelo -> instancia del regresor.
    """
    models = {
        # -------------------------------
        # Modelos scikit-learn estándar
        # -------------------------------
        "DecisionTree_Reg": DecisionTreeRegressor(
            max_depth=10,       # profundidad máxima del árbol
            random_state=42     # semilla para reproducibilidad
        ),
        
        "LinearRegression_Reg": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(
                n_jobs=-1,                      # usar todos los núcleos de CPU
            ))
        ]),
        
        "KNN_Reg": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(
                n_neighbors=5,          # número de vecinos
                weights="distance",     # ponderar por distancia
                metric="minkowski",     # distancia euclídea
                p=2
            ))
        ]),
        
		"MLP_Reg": Pipeline([
			("scaler", StandardScaler()),
			("regressor", MLPRegressor(
				hidden_layer_sizes=(32, 16),    # dos capas ocultas
				max_iter=500,					# iteraciones máximas
				learning_rate_init=0.01,		# tasa de aprendizaje inicial
				tol=0.001,                      # tolerancia para detener el entrenamiento
				n_iter_no_change=10,            # número de iteraciones sin mejora antes de detener el entrenamiento
				batch_size=64,                  # tamaño del lote
				random_state=42
			))
		]),

        "RandomForest_Reg": RandomForestRegressor(
            n_estimators=20,        # número de árboles
            max_depth=10,           # profundidad máxima
            max_features="sqrt",    # número de características a considerar en cada división
            n_jobs=-1,              # usar todos los núcleos de CPU
            random_state=42
        ),
        
        # -------------------------------------------------------------------------------
        # Versiones ONNX de los mismos modelos (compresión y cuantización)
        # ONNXModelFromSklearnRegressor entrena el modelo sklearn y lo convierte a ONNX
        # -------------------------------------------------------------------------------
        "DecisionTree_ONNX_Reg": ONNXModelFromSklearnRegression(
            lambda: DecisionTreeRegressor(
                max_depth=10,
                random_state=42
            ),
            quantization_mode='static'
        ),
        
        "LinearRegression_ONNX_Reg": ONNXModelFromSklearnRegression(
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression(
                    n_jobs=-1
                ))
            ]),
            quantization_mode='static'
        ),
        
        "KNN_ONNX_Reg": ONNXModelFromSklearnRegression(
            lambda: KNeighborsRegressor(
                n_neighbors=5,
                weights="distance",     # ponderar por distancia
                metric="minkowski",     # distancia euclídea
                p=2
            ),
            quantization_mode='static'
        ),
        
		"MLP_ONNX_Reg": ONNXModelFromSklearnRegression(
			lambda: Pipeline([
				("scaler", StandardScaler()),
				("regressor", MLPRegressor(
					hidden_layer_sizes=(32, 16),
					max_iter=500,
					learning_rate_init=0.01,
					tol=0.001,
					n_iter_no_change=10,
					batch_size=64,
					random_state=42
				))
			]),
			quantization_mode='static'
		),
        
        "RandomForest_ONNX_Reg": ONNXModelFromSklearnRegression(
            lambda: RandomForestRegressor(
                n_estimators=20,
                max_depth=10,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ),
            quantization_mode='static'
        )
    }
    
    # ------------------------------------------------------------
    # Modelos TinyML: solo se añaden si se proporciona input_dim
    # ------------------------------------------------------------
    if input_dim is not None:
        # --- Modelo PyTorch (TorchScript) para regresión ---
        pytorch_teacher = TorchScriptModelWrapperRegression(
            input_dim=input_dim,
            hidden_sizes=[128, 64, 32],
            output_dim=output_dim,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode=None,
            teacher_model=None,
            alpha=0.5
        )
        
        models["MLP_PyTorch_Reg"] = TorchScriptModelWrapperRegression(
            input_dim=input_dim,
            hidden_sizes=[32, 16],
            output_dim=output_dim,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode='static',
            teacher_model=pytorch_teacher,
            alpha=0.5
        )

        # --- Modelo TensorFlow Lite para regresión ---
        tflite_teacher = TFLiteModelWrapperRegression(
            input_dim=input_dim,
            hidden_sizes=[128, 64, 32],
            output_dim=output_dim,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode=None,
            teacher_model=None,
            alpha=0.5
        )
        
        models["MLP_TFLite_Reg"] = TFLiteModelWrapperRegression(
            input_dim=input_dim,
            hidden_sizes=[32, 16],
            output_dim=output_dim,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode='static',
            teacher_model=tflite_teacher,
            alpha=0.5
        )

    return models

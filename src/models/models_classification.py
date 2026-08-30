from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.models.wrappers.onnx_wrapper import ONNXModelFromSklearnClassification
from src.models.wrappers.pytorch_wrapper import TorchScriptModelWrapperClassification
from src.models.wrappers.tflite_wrapper import TFLiteModelWrapperClassification


def get_classification_models(input_dim=None, num_classes=None):
    """
    Construye y devuelve un diccionario con todos los clasificadores a evaluar.

    :param input_dim: Número de características de entrada (para MLP).
    :param num_classes: Número de clases del problema (necesario para configurar la salida).
    
	:returns: Diccionario con nombre del modelo -> instancia del clasificador.
    """
    models = {
        # ------------------------------
        # Modelos scikit-learn estándar
        # ------------------------------
        "DecisionTree_Clas": DecisionTreeClassifier(
            max_depth=10,       # profundidad máxima del árbol
            random_state=42     # semilla para reproducibilidad
        ),

        "LinearSVC_Clas": Pipeline([
            ("scaler", StandardScaler()),	# estandarización de características
            ("clf", LinearSVC(
                C=1.0,						# parámetro de regularización
                class_weight="balanced",	# ajusta pesos por clase automáticamente
                max_iter=1000,				# iteraciones máximas
                dual=False,					# formulación primal (recomendado para n_muestras > n_features)
                random_state=42
            ))
        ]),

        "KNN_Clas": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,			# número de vecinos
                weights="distance",		# ponderar por distancia
                metric="minkowski",		# distancia euclídea
                p=2
            ))
        ]),

        "MLP_Clas": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16),	# dos capas ocultas
                max_iter=500,					# iteraciones máximas
                learning_rate_init=0.01,		# tasa de aprendizaje inicial
                tol=0.001,                      # tolerancia para detener el entrenamiento
                n_iter_no_change=10,            # número de iteraciones sin mejora antes de detener el entrenamiento
                batch_size=64,					# tamaño del lote
                random_state=42,
            ))
        ]),
        
        "RandomForest_Clas": RandomForestClassifier(
            n_estimators=20,		# número de árboles
            max_depth=10,			# profundidad máxima de cada árbol
            max_features="sqrt",	# número de características a considerar en cada división
            n_jobs=-1,				# usar todos los núcleos de CPU
            random_state=42
        ),
        
		# -----------------------------------------------------------------------------------
        # Versiones ONNX de los modelos (compresión y cuantización)
        # ONNXModelFromSklearn entrena internamente el modelo sklearn y lo convierte a ONNX
        # -----------------------------------------------------------------------------------
        "DecisionTree_ONNX_Clas": ONNXModelFromSklearnClassification(
            lambda: DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            quantization_mode='static'
        ),

        "LinearSVC_ONNX_Clas": ONNXModelFromSklearnClassification(
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,      # mayor iteraciones para convergencia
                    dual=False,         # selección automática de dual/primal
                    random_state=42
                ))
            ]),
            quantization_mode='static'
        ),
        
        "KNN_ONN_ClasX": ONNXModelFromSklearnClassification(
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    metric="minkowski",
                    p=2
                ))
            ]),
            quantization_mode='static'
        ),

        "MLP_ONNX_Clas": ONNXModelFromSklearnClassification(
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
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
        
        "RandomForest_ONNX_Clas": ONNXModelFromSklearnClassification(
            lambda: RandomForestClassifier(
                n_estimators=20,
                max_depth=10,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ),
            quantization_mode='static'
        )
    }
    
    # --------------------------------------------------------------------------
    # Modelos TinyML: solo se añaden si se proporcionan input_dim y num_classes
    # --------------------------------------------------------------------------
    if input_dim is not None and num_classes is not None:
        # --- Modelo PyTorch ---
        pytorch_teacher = TorchScriptModelWrapperClassification(
            input_dim=input_dim,
            hidden_sizes=[128, 64, 32],
            output_dim=num_classes,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode=None,
            teacher_model=None,
            alpha=0.5,
            temperature=1.0
        )
        
        models["MLP_PyTorch_Clas"] = TorchScriptModelWrapperClassification(
            input_dim=input_dim,
            hidden_sizes=[32, 16],
            output_dim=num_classes,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode='static',
            teacher_model=pytorch_teacher,
            alpha=0.5,
            temperature=2.0
        )
        
        # --- Modelo TensorFlow Lite (LiteRT) ---
        tflite_teacher = TFLiteModelWrapperClassification(
            input_dim=input_dim,
            hidden_sizes=[128, 64, 32],
            output_dim=num_classes,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode=None,
            teacher_model=None,
            alpha=0.5,
            temperature=1.0
        )
        
        models["MLP_TFLite_Clas"] = TFLiteModelWrapperClassification(
            input_dim=input_dim,
            hidden_sizes=[32, 16],
            output_dim=num_classes,
            epochs=500,
            lr=0.01,
            weight_decay=0.0001,
            batch_size=64,
            tol=0.001,
            n_iter_no_change=10,
            random_state=42,
            quantization_mode='static',
            teacher_model=tflite_teacher,
            alpha=0.5,
            temperature=2.0
        )

    return models

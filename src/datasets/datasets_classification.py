from src.datasets.loaders.load_car import load_car
from src.datasets.loaders.load_diabetes import load_diabetes
from src.datasets.loaders.load_gamma import load_gamma
from src.datasets.loaders.load_har import load_har
from src.datasets.loaders.load_iris import load_iris
from src.datasets.loaders.load_letter_recognition import load_letter_recognition
from src.datasets.loaders.load_mnist import load_mnist
from src.datasets.loaders.load_soybean import load_soybean
from src.datasets.loaders.load_student_dropout import load_student_dropout
from src.datasets.loaders.load_wine import load_wine_classification


def get_classification_datasets():
    """
    Devuelve un diccionario con los datasets de clasificación disponibles.
    Cada entrada del diccionario tiene como clave el nombre del dataset y como valor la función de carga correspondiente.
    
    :returns: Diccionario con los datasets de clasificación.
    """
    return {
        "CAR": load_car,
        "DIABETES": load_diabetes,
        "GAMMA": load_gamma,
        "HAR": load_har,
        "IRIS": load_iris,
        "LETTER RECOGNITION": load_letter_recognition,
        "MNIST": load_mnist,
        "SOYBEAN": load_soybean,
        "STUDENT DROPOUT": load_student_dropout,
        "WINE": load_wine_classification
    }

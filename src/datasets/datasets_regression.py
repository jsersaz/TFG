from src.datasets.loaders.load_abalone import load_abalone
from src.datasets.loaders.load_air_quality import load_air_quality
from src.datasets.loaders.load_automobile import load_automobile
from src.datasets.loaders.load_energy_data import load_energy_data
from src.datasets.loaders.load_bike_sharing import load_bike_sharing
from src.datasets.loaders.load_communities_and_crime import load_communities_crime
from src.datasets.loaders.load_obesity_levels import load_obesity_levels
from src.datasets.loaders.load_real_estate_valuation import load_real_estate_valuation
from src.datasets.loaders.load_student_performance import load_student_performance
from src.datasets.loaders.load_wine import load_wine_quality_regression


def get_regression_datasets():
    """
    Devuelve un diccionario con los datasets de regresión disponibles.
    Cada entrada del diccionario tiene como clave el nombre del dataset y como valor la función de carga correspondiente.
    
    :returns: Diccionario con los datasets de regresión.
    """
    
    return {
        "ABALONE": load_abalone,
        "AIR QUALITY": load_air_quality,
        "APPLIANCES ENERGY PREDICTION": load_energy_data,
        "AUTOMOBILE": load_automobile,
        "BIKE SHARING HOURLY": lambda: load_bike_sharing('hour'),
        "COMMUNITIES AND CRIME": load_communities_crime,
        "OBESITY LEVELS": load_obesity_levels,
        "REAL ESTATE VALUATION": load_real_estate_valuation,
        "STUDENT PERFORMANCE": lambda: load_student_performance('math'),
        "WINE QUALITY": load_wine_quality_regression
    }

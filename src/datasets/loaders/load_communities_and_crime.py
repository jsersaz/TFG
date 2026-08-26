import numpy as np
import os
import pandas as pd


def load_communities_crime(data_dir=None):
    """
    Carga el dataset "Communities and Crime".
    Nombre de archivo: 'communities.data'.
    Predecir: ViolentCrimesPerPop (tasa de crímenes violentos por población).
    
    :param data_dir: Directorio donde se encuentra el archivo de datos.
    
    :returns: X (características), y (objetivo).
    """
    if data_dir is None:
        # Obtener la ruta del directorio donde está este script
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir hasta la raíz del proyecto (code): loaders -> datasets -> src -> code
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(loader_dir)))
        data_dir = os.path.join(project_root, 'data', 'communities_and_crime')
    
    # Leer el archivo
    filepath = os.path.join(data_dir, 'communities.data')
    # filepath = f'{data_dir}communities.data'
    
    # Asignar nombres a las columnas
    columns = [
        'state', 'county', 'community', 'communityname', 'fold',
        'population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian',
        'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
        'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc',
        'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc',
        'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap',
        'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad',
        'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ',
        'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
        'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par',
        'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg',
        'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
        'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly',
        'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
        'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous',
        'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc',
        'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone',
        'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ',
        'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn',
        'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
        'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop',
        'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop',
        'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp',
        'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
        'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars',
        'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn',
        'PolicBudgPerPop', 'ViolentCrimesPerPop'
    ]
    
    # Cargar el archivo
    df = pd.read_csv(filepath, header=None, names=columns,
                     na_values='?', encoding='utf-8')
    
    # Eliminar columnas no predictivas: state, county, community, communityname, fold
    non_predictive = ['state', 'county', 'community', 'communityname', 'fold']
    df = df.drop(columns=non_predictive)
    
    # Separar características y objetivo
    X_raw = df.drop(columns=['ViolentCrimesPerPop'])
    y = df['ViolentCrimesPerPop'].values.astype(np.float32)
    
    # Eliminar columnas con más del 30% de valores faltantes
    missing_ratio = X_raw.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio < 0.3].index
    X_raw = X_raw[cols_to_keep]
    # print(f"Columnas eliminadas por alto missing: {list(missing_ratio[missing_ratio >= 0.3].index)}")
    
    # Imputar el resto con la mediana de cada columna
    for col in X_raw.columns:
        if X_raw[col].isnull().any():
            median_val = X_raw[col].median()
            X_raw[col] = X_raw[col].fillna(median_val)
            
    # Eliminar columnas que aún tengan NaN (por si alguna columna era todo NaN)
    X_raw = X_raw.dropna(axis=1, how='any')
    
    # Eliminar filas con NaN residuales
    valid_idx = ~(X_raw.isnull().any(axis=1) | np.isnan(y))
    X_raw = X_raw[valid_idx]
    y = y[valid_idx]
    
    # Convertir a arrays numpy
    X = X_raw.values.astype(np.float32)
    
    # print(f"Communities and Crime cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    # print(f"ViolentCrimesPerPop: min={y.min():.4f}, max={y.max():.4f}, media={y.mean():.4f}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_communities_crime(None)
    print("Dimensiones de X:", X.shape)
    print("Primeras 5 filas de X:\n", X[:5])
    print("Primeros 5 valores de y:", y[:5])

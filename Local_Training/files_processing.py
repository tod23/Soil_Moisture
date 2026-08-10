import os
import numpy as np
import pandas as pd
from config import DATE_COL, TARGET_COL
from features import resample_timeseries, interpolate_timeseries, update_soil_property, engineer_features
from typing import List, Dict, Tuple

def load_csv(feat_cfg: Dict, csv_path: str, df_master = None, depth = None, MONTHS = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    DENSE_FEATURE_COLS = feat_cfg["dense"]
    SOIL_PROPERTIES_COLS = feat_cfg["soil"]
    SPARSE_FEATURE_COLS = feat_cfg["sparse"]
    FEATURE_COLS = DENSE_FEATURE_COLS + SOIL_PROPERTIES_COLS + SPARSE_FEATURE_COLS

    # Standardisation du nom de colonne date
    if 'date' in df.columns:
        df = df.rename(columns={'date': DATE_COL})
    if 'Unnamed: 0' in df.columns and DATE_COL not in df.columns:
        df = df.rename(columns={'Unnamed: 0': DATE_COL})

    if DATE_COL not in df.columns:
        raise ValueError(f"[{csv_path}] Colonne date ({DATE_COL}) manquante")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # --- RESAMPLE d'abord (horaire → journalier) ---
    df = df.set_index(DATE_COL)
    if len(df) > 0 and (df.index.value_counts().mean() > 1.5):
        df = resample_timeseries(df, freq='D', method='mean')
    df = df.reset_index()

    # --- MERGE météo ensuite ---
    station_dir = os.path.dirname(csv_path)
    meteo_path = os.path.join(station_dir, 'meteo_daily.csv')
    if os.path.exists(meteo_path):
        df_meteo = pd.read_csv(meteo_path)
        df_meteo[DATE_COL] = pd.to_datetime(df_meteo[DATE_COL])
        for c in DENSE_FEATURE_COLS:
            if c in df_meteo.columns:
                df_meteo[c] = pd.to_numeric(df_meteo[c], errors="coerce")
        df = pd.merge(df, df_meteo, on=DATE_COL, how='inner', suffixes=('', '_meteo'))
    
    if TARGET_COL in df.columns:
        df = interpolate_timeseries(df, col=TARGET_COL, n=1)

    df = update_soil_property(df, df_master, feat_cfg["soil"], depth)
    if df is None:
        return None
    
    df = engineer_features(df, feat_cfg)
    if MONTHS is not None:
        df = df[df[DATE_COL].dt.month.isin(MONTHS)].reset_index(drop=True)

    # --- Filtrage colonnes ---
    cols_to_keep = [DATE_COL] + feat_cfg["soil"] + feat_cfg["dense"]
    if TARGET_COL not in cols_to_keep:
        cols_to_keep.append(TARGET_COL)
    for c in feat_cfg["sparse"]:
        if c in df.columns:
            cols_to_keep.append(c)

    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        raise ValueError(f"[{csv_path}] Colonnes manquantes dans le final: {missing}")

    df = df[cols_to_keep].copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def get_file_paths(files_path: str) -> List[str]:
    """Parcourt depth_X/station_Y/ et retourne tous les *soil_moisture*.csv"""
    file_paths = []
    for station_dir in os.listdir(files_path):
        station_path = os.path.join(files_path, station_dir)
        if not os.path.isdir(station_path):
            continue
        for fname in os.listdir(station_path):
            if fname.endswith('.csv') and 'soil_moisture' in fname:
                file_paths.append(os.path.join(station_path, fname))
    return file_paths



def flush_results(csv_path: str) -> None:
    if not _RESULTS_BUFFER:
        return
    df = pd.DataFrame(_RESULTS_BUFFER)
    df.to_csv(csv_path, mode='a',
              header=not os.path.exists(csv_path), index=False)
    print(f"✓ {len(_RESULTS_BUFFER)} résultats flushés vers {csv_path}")
    _RESULTS_BUFFER.clear()


_RESULTS_BUFFER: List[Dict] = []
def save_to_results_csv(
    results_dict: Dict,
    csv_path: str,
    feat_cfg: Dict
) -> None:
    """
    Sauvegarde les résultats dans un CSV centralisé en mode append.
    
    Args:
        results_dict : Dictionnaire avec les colonnes à sauvegarder
        csv_path : Chemin du fichier CSV cible
    
    Structure des colonnes:
        timestamp | type | depth | lookback | horizon | nb_windows | model |
        features | metric_name | value | value_std | Network
    
    Types possibles: GLOBAL, DETAILED, GRANDVILLERS, CROSSVAL
    """
    import os
    from datetime import datetime
    FEATURE_COLS = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Ajouter timestamp automatiquement
    results_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Ajouter les features automatiquement si non spécifiées, en utilisant la variable globale FEATURE_COLS
    if 'features' not in results_dict:
        results_dict['features'] = ",".join(FEATURE_COLS)
    
    # Colonnes standard (certaines peuvent être vides selon le type)
    columns = [
        'timestamp', 'features', 'experience_name', 'type', 'depth', 'lookback', 'horizon', 'nb_windows',
        'model', 'metric_name', 'value', 'value_std', 'network'
       ]

    # Créer une nouvelle ligne avec les colonnes manquantes remplies de None
    row_dict = {col: results_dict.get(col, None) for col in columns}

    _RESULTS_BUFFER.append(row_dict)
    print(f"✓ Résultat mis en tampon: {results_dict.get('type', '?')} - "
          f"{results_dict.get('metric_name', '?')} = {results_dict.get('value', '?')}")
    
def split_spatial_files(file_paths: List[str], df_master: pd.DataFrame, depth: float, 
                        horizons: List[int], lookbacks: List[int], feat_cfg: Dict, MONTHS: List[int]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    # Grouper les chemins par station
    station_map = {}
    for path in file_paths:
        sid = os.path.basename(os.path.dirname(path))  # "station_1"
        station_map.setdefault(sid, []).append(path)

    # Shuffle des stations (pas des fichiers)
    station_ids = list(station_map.keys())
    np.random.shuffle(station_ids)

    n = len(station_ids)
    n_train = max(1, int(n * 0.7))
    n_val   = max(1, int(n * 0.15))

    train_stations = station_ids[:n_train]
    val_stations   = station_ids[n_train:n_train + n_val]
    test_stations  = station_ids[n_train + n_val:]

    def load_station_group(station_ids_subset):
            result = []
            for sid in station_ids_subset:
                for p in station_map[sid]:
                    df = load_csv(feat_cfg, p, df_master, depth, MONTHS)  
                    if df is None:
                        continue
                    if len(df) >= max(lookbacks) + max(horizons) + 1:
                        result.append(df)
            return result

    train_list = load_station_group(train_stations)
    val_list   = load_station_group(val_stations)
    test_list  = load_station_group(test_stations)


    print(f"Stations: train={len(train_stations)}, val={len(val_stations)}, test={len(test_stations)}")
    print(f"Fichiers chargés: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")
    return train_list, val_list, test_list

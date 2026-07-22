import os
import numpy as np
import tensorflow as tf

FOLDER_NAME = "Fine_Tuning" 

ROOT_DIR = "/home/theo/Dataset" # Si on local machine

OSIRIS_DIR = os.path.join(ROOT_DIR, "Osiris_dataset")

Grandvillers_path = os.path.join(ROOT_DIR, "Grandvillers_data")

drive_dir = os.path.join("/home/theo/Documents", FOLDER_NAME, "outputs") # Si on local machine

RESULTS_CSV_PATH = os.path.join(drive_dir, "results.csv") # Si on local machine

# ============================================================
# Configuration
# ============================================================

FOLDER_ISMN = "station_depth_csv"
MASTER_CSV_PATH = os.path.join(ROOT_DIR, FOLDER_ISMN, "Soil_Properties_Master.csv")

base_path = os.path.join(ROOT_DIR, FOLDER_ISMN, "depth")

FULL_DENSE_h = [ "soil_moisture", "ET0", "IRRAD", "TMIN", "TMAX", "VAP", "WIND", "RAIN",
              "VPD", "T_RANGE",
              "RAIN_CUM_3D", "RAIN_CUM_7D", "RAIN_CUM_14D",
              "doy_sin", "doy_cos"]

FULL_DENSE = [  "ET0", "IRRAD", "TMIN", "TMAX", "VAP", "WIND", "RAIN",
              "VPD", "T_RANGE",
              "RAIN_CUM_3D", "RAIN_CUM_7D", "RAIN_CUM_14D",
              "doy_sin", "doy_cos"]

FULL_SOIL  = [ "clay", "silt", "bulk", "sand", "dem", "ksat_m_1km", "dem_slope", "dem_aspect", "dem_twi"]

FULL_SPARSE = ["S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7",
               "S2_B8", "S2_B8A", "S2_B11", "S2_B12",
               "S2_NDVI", "S2_NDWI", "S2_SAVI", "S2_MNDWI", "S2_NBR",
               "S1_VV", "S1_VH", "S1_angle",
               "S1_VV_over_VH", "S1_VH_over_VV"]

FULL_SPARSE_S1 = ["S1_VV", "S1_VH", "S1_angle",
                   "S1_VV_over_VH", "S1_VH_over_VV"]

FULL_SPARSE_S2 = ["S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7",
               "S2_B8", "S2_B8A", "S2_B11", "S2_B12",
               "S2_NDVI", "S2_NDWI", "S2_SAVI", "S2_MNDWI", "S2_NBR"]


# Training settings
HORIZONS = [7]              # predict * days ahead 
LOOKBACK = [7]                  # use past * days to predict next day
DEPTHS = [0.1]  # we will loop over these depths and train one model per depth
ALL_NETWORKS = [
    "all"]

    # ["COSMOS-UK", "GROW", "PTSMN", "TAHMO", "TERENO"]
SEPARATE_NETWORKS = [

     ["COSMOS-UK"],
             ["DWD"],
             ["FR_Aqui", "GROW"],
             ["PTSMN"], 
            ["SMOSMANIA"], ["SOILSCAPE"],
            ["TAHMO","TERENO"],
        ["TERENO"], 
            ["TWENTE"], 
            ["XMS-CAT"]
        ]

NB_WINDOWS = [100000]    
# Models: xgboost and lightgbm are fast, non-DL
MODELS = ["lstm"]
# , "lightgbm"]

def _without(lst, *items):
    return [x for x in lst if x not in items]

FEATURE_CONFIGS = [

    # ── 4.2.1 Comparaison des modèles ──
    {"name": "Models", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lightgbm", "lstm", "gru", "tcn", "transformer"],
     "networks": ALL_NETWORKS},

    # ── 4.2.2 Effet du lookback ──
    {"name": "Lookback", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [1, 2, 4, 7, 14, 21, 28],
     "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    # ── 4.2.3 Effet de l'horizon ──
    {"name": "Horizon", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [7],
     "horizons": [1, 3, 7, 14, 21],
     "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    # ── 4.2.4 Effet du nombre de fenêtres ──
    {"name": "Nb_Windows", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS,
     "nb_windows": [1000, 5000, 10000, 20000, 50000, 100000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    # ── 4.2.5 Contribution des sources de données ──
    {"name": "Features_Dense", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": [],
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    {"name": "Features_Dense_S1", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    {"name": "Features_Dense_S2", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S2,
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    {"name": "Features_All", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE,
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost", "lstm"],
     "networks": ALL_NETWORKS},

    # ── 4.2.6 Importance de la proximité géographique (par réseau) ──
    {"name": "Networks", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
     "models": ["xgboost"],
     "networks": SEPARATE_NETWORKS},
]

SAVE_PLOTS = False
SAVE_NETWORKS_DIR = True
SAVE_MODELS_DIR = True
SAVE_RESULTS_CSV = True



TARGET_COL = "soil_moisture"
DATE_COL = "date_time"  # L'index temporel est sauvegardé sous date_time par process_timeseries
EPOCHS = 150
BATCH_SIZE = 32
SEED = 8

if SAVE_MODELS_DIR:
    os.makedirs(drive_dir, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import os
import numpy as np
import tensorflow as tf


# OUTPUT_NAME = "Fine_Tuning_Osiris" 
# Grandvillers_path = '/content/gdrive/My Drive/Grandvillers/Grandvillers'
# ROOT_DIR = "/content/gdrive/MyDrive/Soil_Moisture/dataset_training" # Si on Colab
# OSIRIS_DIR = os.path.join(ROOT_DIR, "Osiris_dataset")

# drive_dir = os.path.join("/content/gdrive/MyDrive/Soil_Moisture/outputs", OUTPUT_NAME) # Si on Colab
# RESULTS_CSV_PATH = os.path.join(drive_dir, "results.csv") # Si on Colab


FOLDER_NAME = "Fine_Tuning" 

ROOT_DIR = "/home/theo/Dataset" # Si on local machine

OSIRIS_DIR = os.path.join(ROOT_DIR, "Osiris_dataset")

# Répertoires Osiris bruts (2024 et 2025), chargés ensemble pour le leave-one-field-out
OSIRIS_DIR_2024 = "/home/theodore/Documents/Get_Datasets/Osiris_data/Osiris_2024"
OSIRIS_DIR_2025 = "/home/theodore/Documents/Get_Datasets/Osiris_data/Osiris_2025"

# Dataset Osiris unifié (format commun hour + colonnes harmonisées), produit par
# la section "ADAPTATION" de csv_for_hrsm.ipynb. C'est le répertoire de référence
# pour le leave-one-field-out. Sur Colab, placé à côté de station_depth_csv
# (dataset_training/Osiris_unified).
# Local, uncomment :
# OSIRIS_DIR_UNIFIED = "/home/theodore/Documents/Get_Datasets/Osiris_data/Osiris_unified"
OSIRIS_DIR_UNIFIED = os.path.join(ROOT_DIR, "Osiris_unified")

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
               "S1_VV_over_VH", "S1_VH_over_VV",
               "HLS_B2", "HLS_B3", "HLS_B4", "HLS_B5", 
                     "HLS_B6", "HLS_B7", "HLS_B9", "HLS_B10", "HLS_B11", "HLS_NDVI"]

FULL_SPARSE_S1 = ["S1_VV", "S1_VH", "S1_angle",
                   "S1_VV_over_VH", "S1_VH_over_VV"]

FULL_SPARSE_S2 = ["S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7",
               "S2_B8", "S2_B8A", "S2_B11", "S2_B12",
               "S2_NDVI", "S2_NDWI", "S2_SAVI", "S2_MNDWI", "S2_NBR"]

FULL_SPARSE_HLS30 = ["HLS_B2", "HLS_B3", "HLS_B4", "HLS_B5", 
                     "HLS_B6", "HLS_B7", "HLS_B9", "HLS_B10", "HLS_B11", "HLS_NDVI"]

# Training settings
HORIZONS = [7]              # predict * days ahead 
LOOKBACK = [7]                  # use past * days to predict next day
DEPTHS = [0.1, #0.5]
           0.2, 0.3, 0.4, 0.5]  # we will loop over these depths and train one model per depth
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


    # # ── 4.2.1 Comparaison des modèles ──
    # {"name": "Models", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE,
    #  "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
    #  "models": ["xgboost", "lightgbm", "lstm", "gru", "tcn", "transformer"],
    #  "networks": ALL_NETWORKS},

#     # ── 4.2.2 Effet du lookback ──
#     {"name": "Lookback", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
#      "lookbacks": [1, 2, 4, 7, 14, 21, 28],
#      "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
#      "models": ["xgboost", "lstm"],
#      "networks": ALL_NETWORKS},

    # ── 4.2.3 Effet de l'horizon ──
    {"name": "Horizon_lightgbm", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE,
     "lookbacks": [1],
     "horizons": [1, 3, 7, 14, 21],
     "depths": DEPTHS, "nb_windows": [50000],
     "models": ["lightgbm"],
     "networks": ALL_NETWORKS},

    # ── 4.2.4 Effet du nombre de fenêtres ──
    {"name": "Nb_Windows_lightgbm", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [1], "horizons": [7], "depths": DEPTHS,
     "nb_windows": [1000, 5000, 10000, 20000, 50000, 100000],
     "models": ["lightgbm"],
     "networks": ALL_NETWORKS},

    {"name": "Features_S1_lightgbm", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": [],
     "lookbacks": [1], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
        "models": ["lightgbm"],
        "networks": ALL_NETWORKS},

    {"name": "Features_S1_lightgbm", "dense": FULL_DENSE, "soil": [], "sparse": FULL_SPARSE_S1,
     "lookbacks": [1], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
        "models": ["lightgbm"],
        "networks": ALL_NETWORKS},

    {"name": "Features_S1_lightgbm", "dense": [], "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [1], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
        "models": ["lightgbm"],
        "networks": ALL_NETWORKS},

    {"name": "Features_S1_lightgbm", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
     "lookbacks": [1], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
        "models": ["lightgbm"],
        "networks": ALL_NETWORKS},

#     # ── 4.2.6 Importance de la proximité géographique (par réseau) ──
#     {"name": "Networks", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE_S1,
#      "lookbacks": [7], "horizons": [7], "depths": DEPTHS, "nb_windows": [50000],
#      "models": ["xgboost"],
#      "networks": SEPARATE_NETWORKS},
]

SAVE_PLOTS = True
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

# MONTHS=[4,5,6,7,8,9]
MONTHS = None

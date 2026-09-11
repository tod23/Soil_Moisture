import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------
# Chemins (these)
# ---------------------------------------------------------------

FOLDER_NAME = "Fine_Tuning"

ROOT_DIR = os.path.join("/home/theo", "Dataset")

OSIRIS_DIR = os.path.join(ROOT_DIR, "Osiris_unified")

drive_dir = os.path.join("/home/theo", "Documents", FOLDER_NAME, "outputs")

RESULTS_CSV_PATH = os.path.join(drive_dir, "results.csv")



# ============================================================
# Configuration
# ============================================================

FOLDER_ISMN = "station_depth_csv"
MASTER_CSV_PATH = os.path.join(ROOT_DIR, FOLDER_ISMN, "Soil_Properties_Master.csv")

base_path = os.path.join(ROOT_DIR, FOLDER_ISMN, "depth")


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
NB_WINDOWS = [100000]    # number of windows to sample for training
HORIZONS = [7]              # predict * days ahead 
LOOKBACK = [7]                  # use past * days to predict next day
DEPTHS = [0.1, #0.5]
           0.2, 0.3, 0.4, 0.5]  # we will loop over these depths and train one model per depth
ALL_NETWORKS = [
    "all"]
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

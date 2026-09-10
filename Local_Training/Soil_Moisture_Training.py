from Fcn_Training import get_osiris_data, osiris_train

from config import (OSIRIS_DIR, drive_dir, 
                    FULL_DENSE, FULL_SOIL, FULL_SPARSE,
                    DEPTHS, LOOKBACK, HORIZONS, MODELS, ALL_NETWORKS, NB_WINDOWS)

FEATURE_CONFIGS = [
    {"name": "Osiris_train", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE,
     "lookbacks": LOOKBACK, "horizons": HORIZONS, "depths": DEPTHS,
     "nb_windows": NB_WINDOWS, "models": MODELS, "networks": ALL_NETWORKS},
]

all_dfs = get_osiris_data(OSIRIS_DIR)

for feat_cfg in FEATURE_CONFIGS:
    feature_cols = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
    print(f"\n========== FEATURE SET: {feat_cfg['name']} ==========")
    print(f"  Features: {feature_cols}")

    ############### Entraînement Osiris from scratch (LOO) ###############
    osiris_train(feat_cfg, drive_dir, all_dfs)
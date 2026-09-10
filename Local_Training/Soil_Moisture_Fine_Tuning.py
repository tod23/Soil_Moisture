from Fcn_Training import full_training, get_osiris_data, osiris_fine_tuning, full_eval_osiris

from config import (OSIRIS_DIR, drive_dir, base_path, MONTHS,
                    FULL_DENSE, FULL_SOIL, FULL_SPARSE,
                    DEPTHS, LOOKBACK, HORIZONS, MODELS, ALL_NETWORKS, NB_WINDOWS)

FEATURE_CONFIGS = [
    {"name": "Fine_Tuning", "dense": FULL_DENSE, "soil": FULL_SOIL, "sparse": FULL_SPARSE,
     "lookbacks": LOOKBACK, "horizons": HORIZONS, "depths": DEPTHS,
     "nb_windows": NB_WINDOWS, "models": MODELS, "networks": ALL_NETWORKS},
]

all_dfs = get_osiris_data(OSIRIS_DIR)

for feat_cfg in FEATURE_CONFIGS:
    feature_cols = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
    print(f"\n========== FEATURE SET: {feat_cfg['name']} ==========")
    print(f"  Features: {feature_cols}")

    ############### Training ISMN (modèles de base)  ###############
    full_training(feat_cfg, base_path, drive_dir, MONTHS)

    ############### Evaluation Osiris avec les modèles ISMN ###############
    full_eval_osiris(feat_cfg, drive_dir, all_dfs)

    ############### Fine-Tuning Osiris (LOO) ###############
    osiris_fine_tuning(feat_cfg, drive_dir, all_dfs)
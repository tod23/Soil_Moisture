import os

from Fcn_Training import (full_training, 
                        get_osiris_data,
                        Get_Grandvillers_data,
                        prepare_era5_dataset,
                        osiris_train,
                        osiris_fine_tuning,
                        full_eval_grandvillers,
                        full_eval_osiris)
from config import OSIRIS_DIR, FEATURE_CONFIGS, drive_dir, Grandvillers_path, base_path, MONTHS

# 1. Charger les données de test Grandvillers
test_list_local = Get_Grandvillers_data(os.path.join(Grandvillers_path, "Grandvillers_satellites"))
# 2. Préparer directement la version de test ERA5 pour chaque fichier
test_list_era5 = []
print("--- PRÉPARATION DU JEU DE TEST ERA5 ---")
for i, df in enumerate(test_list_local):
    df_substituted = prepare_era5_dataset(df, display_name=f"Fichier {i}")
    test_list_era5.append(df_substituted)

all_dfs = get_osiris_data(OSIRIS_DIR)

# for feat_cfg in FEATURE_CONFIGS:
#     feature_cols = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
#     osiris_train(feat_cfg, drive_dir, all_dfs)
#     full_eval_grandvillers(feat_cfg, drive_dir, test_list_local, test_list_era5)

### Training
for feat_cfg in FEATURE_CONFIGS:
    # Réassigner les globales
    feature_cols = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
    print(f"\n========== FEATURE SET: {feat_cfg['name']} ==========")
    print(f"  Features: {feature_cols}")

    ############### Training ISMN ###############
    full_training(feat_cfg, base_path, drive_dir, MONTHS)

    # ########### Evaluation on Osiris ###############
    # full_eval_osiris(feat_cfg, drive_dir, all_dfs)

    # ########## Evaluation on Grandvillers ###############
    # full_eval_grandvillers(feat_cfg, drive_dir, test_list_local, test_list_era5)

    # ############ Fine-Tuning Osiris ###############
    # osiris_fine_tuning(feat_cfg, drive_dir, all_dfs)

    ########## Evaluation on Grandvillers ###############
    # for i in range(len(feat_cfg["models"])):
    #     feat_cfg["models"][i] = feat_cfg["models"][i] + "_fine_tuned"
    # feat_cfg["name"] = feat_cfg["name"] + "_fine_tuned"
    # full_eval_grandvillers(feat_cfg, drive_dir, test_list_local, test_list_era5)
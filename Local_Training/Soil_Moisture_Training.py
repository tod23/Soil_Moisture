from Fcn_Training import (full_training,
                        get_osiris_data,
                        osiris_train,
                        osiris_fine_tuning,
                        full_eval_osiris)

from config import OSIRIS_DIR, FEATURE_CONFIGS, drive_dir, base_path, MONTHS

all_dfs = get_osiris_data(OSIRIS_DIR)

for feat_cfg in FEATURE_CONFIGS:
    feature_cols = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
    osiris_train(feat_cfg, drive_dir, all_dfs)

### Training
# for feat_cfg in FEATURE_CONFIGS:
#     # Réassigner les globales
#     feature_cols = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]
#     print(f"\n========== FEATURE SET: {feat_cfg['name']} ==========")
#     print(f"  Features: {feature_cols}")

#     ############### Training ISMN ###############
#     full_training(feat_cfg, base_path, drive_dir, MONTHS)

#     # ########### Evaluation on Osiris ###############
#     # full_eval_osiris(feat_cfg, drive_dir, all_dfs)

#     # ############ Fine-Tuning Osiris ###############
#     # osiris_fine_tuning(feat_cfg, drive_dir, all_dfs)
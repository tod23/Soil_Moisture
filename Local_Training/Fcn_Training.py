import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Any
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import callbacks
import warnings

from config import (RESULTS_CSV_PATH, MASTER_CSV_PATH, SAVE_MODELS_DIR, SAVE_RESULTS_CSV, SAVE_NETWORKS_DIR,
                    SAVE_PLOTS, EPOCHS, BATCH_SIZE, DATE_COL, BATCH_SIZE,
                    Grandvillers_path)
from preprocessing import cut_and_filter_dfs, make_supervised, fit_scalers, cut_timeseries
from files_processing import get_file_paths, save_to_results_csv, flush_results, split_spatial_files
from features import engineer_features, update_soil_name, resample_timeseries
from models import build_models
from metrics import compute_horizon_metrics, inverse_y
from plots import plot_loss, plot_test_true_vs_pred

def train_eval_predict_one_probe(
    drive_dir: str,
    lookback: int,
    horizon: int,
    choose_model: str,
    save_test_plots: str,
    max_windows: int,
    max_test_plots: int,
    depth: float,
    train_list: List[pd.DataFrame],
    val_list: List[pd.DataFrame],
    test_list: List[pd.DataFrame],
    network_str: str,
    feat_cfg: dict
) -> None:
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # 2. Filtrer train_list et val_list pour correspondre à max_windows
    train_list = cut_and_filter_dfs(
        train_list, lookback, horizon, int(0.8 * max_windows), feat_cfg
    )
    val_list = cut_and_filter_dfs(
        val_list, lookback, horizon, max_windows - int(0.8 * max_windows), feat_cfg
    )
    
    # 3. Générer X_train, y_train, X_val, y_val avec fit=True (car les données sont déjà nettoyées)
    model_format = choose_model.lower()

    fit_scalers(train_list, scaler_x, scaler_y, feat_cfg)
    X_train, y_train, _ = make_supervised(train_list, scaler_x, scaler_y, lookback, horizon, feat_cfg, model_format)
    X_val, y_val, _ = make_supervised(val_list, scaler_x, scaler_y, lookback, horizon, feat_cfg, model_format)

    print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}")


    # 3. Modèle
    model = build_models(choose_model, lookback, horizon, feat_cfg)
    
    # 4. Paths

    is_dl = choose_model in ["convlstm", "lstm", "transformer", "gru", "tcn"]
    if SAVE_MODELS_DIR:
        try:
            model_ext = "keras" if is_dl else "pkl"
            model_path = os.path.join(drive_dir, f"model.{model_ext}")
            scaler_x_path = os.path.join(drive_dir, "scaler_x.pkl")
            scaler_y_path = os.path.join(drive_dir, "scaler_y.pkl")
            history_csv = os.path.join(drive_dir, "history.csv")
            loss_png = os.path.join(drive_dir, "loss_train_val.png")
        except Exception as e:
            print(f"Could not setup output paths: {e}")
    print(f"Samples: train={len(X_train)}, val={len(X_val)}  (lookback={lookback})")
    # 5. Entraînement
    if is_dl:
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        if SAVE_MODELS_DIR:
            cbs.insert(0, callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True))
            cbs.append(callbacks.CSVLogger(history_csv, append=False))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            shuffle=True, callbacks=cbs, verbose=1
        )
        if SAVE_MODELS_DIR:
            model.save(model_path)
            plot_loss(history, loss_png, title=f"Loss: {choose_model}")
    elif choose_model == "xgboost" :
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        if SAVE_MODELS_DIR:
            joblib.dump(model, model_path)
    elif choose_model == "lightgbm" :
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        if SAVE_MODELS_DIR:
            joblib.dump(model, model_path)
    
    else:
        model.fit(X_train, y_train)
        if SAVE_MODELS_DIR:
            joblib.dump(model, model_path)
    if SAVE_MODELS_DIR:
        joblib.dump(scaler_x, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path) 

    # 6. Test evaluation (par fichier original, en découpant les NaN en interne)
    for i, df_test in enumerate(test_list):
        if len(df_test) < lookback + horizon + 1:
            print(f"Test file {i} too short, skipping.")
            continue

        segments = cut_timeseries(df_test, min_length=lookback + horizon + 1, feat_cfg=feat_cfg)
        file_y_true, file_y_pred, file_dates = [], [], []

        for seg in segments:
            X_seg, y_seg, d_seg = make_supervised(
                [seg], scaler_x, scaler_y, lookback, horizon, feat_cfg, model_format
            )
            if len(X_seg) == 0:
                continue
            y_pred_scaled = model.predict(X_seg, verbose=0) if is_dl else model.predict(X_seg)
            file_y_true.append(inverse_y(scaler_y, y_seg))
            file_y_pred.append(inverse_y(scaler_y, y_pred_scaled))
            file_dates.append(d_seg)

        if not file_y_true:
            continue

        y_true = np.concatenate(file_y_true, axis=0)
        y_pred = np.concatenate(file_y_pred, axis=0)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        d_test = np.concatenate(file_dates)

        rmse_h, ubrmse_h, smape_h, r2_h, kge_h, corr_h = compute_horizon_metrics(y_true, y_pred, horizon)
        print(f"[TEST File {i} ] RMSE={rmse_h} ubRMSE={ubrmse_h} SMAPE={smape_h} R²={r2_h} KGE={kge_h} Corr={corr_h}")

        # ── Sauvegarde par fichier de test (remplace CROSSVAL) ──
        def _fmt(arr, prec):
            return ','.join(f'{v:.{prec}f}' for v in arr)

        if SAVE_RESULTS_CSV:
            for metricname, arr, prec in [
                ('rmse', rmse_h, 6), ('ubrmse', ubrmse_h, 6),
                ('smape', smape_h, 2), ('r2', r2_h, 4), ('kge', kge_h, 4), ('corr', corr_h, 4)
            ]:
                save_to_results_csv({
                    'type': f'FILE_{i}',
                    'depth': depth, 'lookback': lookback,
                    'horizon': horizon, 'nb_windows': max_windows,
                    'model': choose_model,
                    'metric_name': metricname,
                    'value': _fmt(arr, prec),
                    'experience_name': feat_cfg['name'],
                    'network' : network_str
                }, RESULTS_CSV_PATH, feat_cfg)
                
        if SAVE_PLOTS and i < max_test_plots:

            if save_test_plots == "first_last":
                horizons_to_plot = [0, horizon - 1]
            elif save_test_plots == "all":
                horizons_to_plot = range(horizon)
            else:
                horizons_to_plot = []
            for j in horizons_to_plot:
                test_png = os.path.join(drive_dir, f"test_true_vs_pred_file_{i}_horizon_{j}.png")
                plot_test_true_vs_pred(
                    pd.to_datetime(d_test), y_true[:, j],
                    y_pred_local=y_pred[:, j],
                    out_png=test_png,
                    title=f"Test True vs Pred File {i} horizon {j} ({len(d_test)} samples)"
                )

    flush_results(RESULTS_CSV_PATH)
    if SAVE_MODELS_DIR:
        print(f"Saved models to: {drive_dir}")
    return

# %%
def feature_name_path(results_csv_path, feature_cols):
    
    if not os.path.exists(results_csv_path):
        return "features_0"
    
    df = pd.read_csv(results_csv_path)
    
    liste_test_features = df["features"].unique().tolist()

    features_name = "features_" + str(len(liste_test_features))
    r = ",".join(feature_cols)
    for i in range(len(liste_test_features)):
        if r == liste_test_features[i]:
            features_name = "features_" + str(i)
            break
    return features_name

# %%

def full_training(feat_cfg, base_path, drive_dir, MONTHS):

    depths = feat_cfg["depths"]
    lookbacks = feat_cfg["lookbacks"]
    horizons = feat_cfg["horizons"]
    nb_windows = feat_cfg["nb_windows"]
    model_name = feat_cfg["models"]
    networks = feat_cfg["networks"]

    features_cols = feat_cfg['dense'] + feat_cfg['soil'] + feat_cfg['sparse']
    df_master = pd.read_csv(MASTER_CSV_PATH)
    
    features_name = feature_name_path(RESULTS_CSV_PATH, features_cols)

    for d in depths:
        
        for network in networks:

            print(f"\n====================== NEW DEPTH: {d} ======================")
            base_path_d = base_path + "_" + str(d)
            all_file_paths = get_file_paths(base_path_d)

            if network != "all":
                meta_path = os.path.join(base_path_d, "metadata.csv")
                if os.path.exists(meta_path):
                    df_meta = pd.read_csv(meta_path)
                    stations_ok = set(df_meta[df_meta['network'].isin(network)]['station_id'])
                    all_file_paths = [
                        p for p in all_file_paths
                        if int(os.path.basename(os.path.dirname(p)).split('_')[1]) in stations_ok
                    ]
                    print(f"Réseaux {network}: {len(all_file_paths)} fichiers conservés")


            all_file_paths = np.random.permutation(all_file_paths).tolist()

            # 1. Split spatial des fichiers
            train_list, val_list, test_list = split_spatial_files(all_file_paths, df_master, d, 
                                                                  horizons, lookbacks, feat_cfg, MONTHS)
            print(f"Spatial split: train={len(train_list)} files, val={len(val_list)} files, test={len(test_list)} files")
        

            for LB in lookbacks:
                print(f"--- Lookback: {LB} ---")
                for horizon in horizons:
                    print(f"--- Horizon: {horizon} ---")
                    for NB in nb_windows:
                        print(f"\n=== Training with NB={NB} windows ===")


                        
                        for m in model_name:
                            print(f"\n      > Modèle: {m}" )                       

                            if SAVE_NETWORKS_DIR : 
                                output_dir = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}", f"network_{network}")
                            else :
                                output_dir = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}")
                                
                            if SAVE_MODELS_DIR:
                                os.makedirs(output_dir, exist_ok=True)

                            train_eval_predict_one_probe(
                                drive_dir=output_dir, lookback=LB, horizon=horizon, choose_model=m,
                                save_test_plots="first_last",
                                max_windows=NB,
                                max_test_plots=5,
                                depth = d,
                                train_list=train_list, val_list=val_list, test_list=test_list,
                                network_str=network,
                                feat_cfg=feat_cfg
                            )


# %% [markdown]
# # Features selection

# %%

# def cut_timeseries(df, col='soil_moisture', min_length=0):
#     """
#     Découpe une série temporelle (DataFrame) en séquences (DataFrames) sans NaN pour LSTM.
#     min_length permet d'ignorer les séquences qui sont trop courtes.
#     """
#     sequences = []
    
#     # Identifier les valeurs valides
#     mask = df[col].notna()
    
#     # Créer un identifiant de groupe qui s'incrémente à chaque présence de NaN
#     groups = (~mask).cumsum()
    
#     # Grouper les données valides par l'identifiant et ajouter chaque sous-dataframe
#     for _, group in df[mask].groupby(groups):
#         if not group.empty and len(group) >= min_length:
#             sequences.append(group)
    
#     return sequences



# """
# Extrait toutes les features pour toutes les stations,
# exporte un DataFrame prêt pour le ML.
# """
# DENSE_FEATURE_COLS = FULL_DENSE 
# SOIL_PROPERTIES_COLS = FULL_SOIL
# SPARSE_FEATURE_COLS = []
# FEATURE_COLS = FULL_DENSE + FULL_SOIL 
# TARGET_COL = "soil_moisture"
# DATE_COL = "date_time"


# # ── 3. Parcourir toutes les stations ──
# # BASE_DIR = "/home/theodore/Documents/Get_Datasets/station_depth_csv"
# BASE_DIR = os.path.join(ROOT_DIR, "station_depth_csv")
# MASTER_CSV = os.path.join(BASE_DIR, "Soil_Properties_Master.csv")
# df_master = pd.read_csv(MASTER_CSV)

# compteur = 0
# all_dfs = []
# for depth_dir in sorted(os.listdir(BASE_DIR)):
#     if not depth_dir.startswith("depth_"):
#         continue
#     depth = float(depth_dir.split("_")[1])
#     depth_path = os.path.join(BASE_DIR, depth_dir)
    
#     for station_dir in sorted(os.listdir(depth_path)):
#         if not station_dir.startswith("station_"):
#             continue
#         station_path = os.path.join(depth_path, station_dir)
        
#         for csv_file in os.listdir(station_path):
#             if compteur <=50 :
                
#                 if not csv_file.endswith(".csv") or csv_file.startswith("meteo"):
#                     continue
#                 compteur += 1
#                 csv_path = os.path.join(station_path, csv_file)
#                 df = load_csv(csv_path, df_master, depth)base_p
#                 df_list = cut_timeseries(df, col='soil_moisture', min_length=0)
#                 for df in df_list:
#                     if df is not None:
#                         df["depth"] = depth
#                         all_dfs.append(df)

#             else :
#                 break

# # ── 4. Concaténer et sauvegarder ──
# dataset = pd.concat(all_dfs, ignore_index=True)
# print(f"Dataset final shape: {dataset.shape}")

# %%
# %pip install deepdiff
# %pip install pingouin
# %pip install tigramite
# %pip install pydash

# %%
# import sys, os, importlib

# sys.path.append(ROOT_DIR)

# from OHE_chrono import OHE_chrono

# importlib.reload(sys.modules['OHE_chrono'])
# new = OHE_chrono(dataset, None, 'soil_moisture',14, 'date_time')


# %%
def load_saved_model_and_scalers(model_dir: str) -> Tuple[Any, StandardScaler, StandardScaler]:

    scaler_x_path = os.path.join(model_dir, "scaler_x.pkl")
    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")

    model_path_keras = os.path.join(model_dir, "model.keras")
    model_path_pkl   = os.path.join(model_dir, "model.pkl")

    if os.path.exists(model_path_keras):
        model_path = model_path_keras
        is_dl = True
    elif os.path.exists(model_path_pkl):
        model_path = model_path_pkl
        is_dl = False
    else:
        print(f"-> Modèle introuvable (ni .keras ni .pkl) dans {model_dir}. On passe...")
        return None, None, None

    if is_dl:
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        model = joblib.load(model_path)
    
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_x, scaler_y

# %%
# %% [markdown]
# # TEST OSIRIS

# %% [markdown]
# Creation d'une liste de tableaux

# %%
def get_osiris_data(osiris_dir: str) -> List[pd.DataFrame]:

    # fichier de correspondance sites -> coordonnées
    sites_file = os.path.join(osiris_dir, "sites_and_soil.csv")
    sites = pd.read_csv(sites_file)

    all_dfs = []
    for _, row in sites.iterrows():

        site_id = row["site_id"]

        # Chargement des csv du site
        dossier_site = os.path.join(osiris_dir, site_id)

        # meteo_daily.csv
        meteo_path = os.path.join(dossier_site, "meteo_daily.csv")
        df_meteo_daily = (pd.read_csv(meteo_path, parse_dates=["timestamp"])
                        .rename(columns={"timestamp": "date_time"})
                        .set_index("date_time")
                        .sort_index()
                        )
        
        # sentinel_data.csv
        sentinel_path = os.path.join(dossier_site, "sentinel_data.csv")
        df_satellite = (pd.read_csv(sentinel_path, parse_dates=["timestamp"])
                        .rename(columns={"timestamp": "date_time"})
                        .set_index("date_time")
                        .sort_index()
                        )

        # Sondes du site (tous les fichiers csv sauf meteo_daily et sentinel_data)
        sonde_files = sorted(
            f for f in os.listdir(dossier_site)
            if f.endswith(".csv")
            and f not in ["meteo_hourly.csv", "meteo_daily.csv", "sentinel_data.csv"])

        for sonde_file in sonde_files:
            # colonne date_time
            sonde_path = os.path.join(dossier_site, sonde_file)
            df_sonde = (pd.read_csv(sonde_path, parse_dates=["timestamp"])
                                .rename(columns={"timestamp": "date_time"})
                                )

            if "date_time" not in df_sonde.columns:
                print(f"Colonne 'date_time' manquante dans {sonde_file}. Ignoré.")
                continue

            # Resample journalier
            df_sonde = df_sonde.set_index("date_time").sort_index()
            df_sonde = resample_timeseries(df_sonde, freq="D", method="mean")

            # Variables statiques
            for key, value in row.items():
                if key not in ["latitude", "longitude", "site_id"]:
                    df_sonde[key] = value
                    
            # Jointure météo et satellite
            df_meteo_daily.index = df_meteo_daily.index.tz_localize(None)  # Force en tz-naive
            df_satellite.index = df_satellite.index.tz_localize(None)        # Force en tz-naive
            df_sonde.index = df_sonde.index.tz_localize(None)  # Si déjà tz-aware

            df_sonde = df_sonde.join(df_meteo_daily, how="left")
            df_sonde = df_sonde.join(df_satellite, how="left")

            # retour colonne date_time
            df_sonde = df_sonde.reset_index()
            df_sonde["no_serie"] = os.path.splitext(sonde_file)[0]
            df_sonde["site_id"] = site_id
            all_dfs.append(df_sonde)
    return all_dfs

# %%
def run_osiris_evaluation(model, scaler_x, scaler_y, all_dfs, model_dir, lookback, horizon, depth, nb_windows=None, model_name="unknown", network_str="", feat_cfg=None):
    """
    Évalue les données osiris et calcule les métriques.
    Prépare les DataFrames (rename, features) puis délègue à evaluate_on_probes.
    """

    is_dl = model_name in ["convlstm", "lstm", "transformer", "gru", "tcn"]

    filtered_dfs = []
    for df in all_dfs:
        if not all(c in df.columns for c in feat_cfg['sparse']):
            print(f"Skipping {df['site_id'].iloc[0]}_{df['no_serie'].iloc[0]}: missing sparse features.")
            continue
        df = df.copy()
        col_name = f'humidity_{int(depth*100)}cm'
        if col_name in df.columns:
            df[col_name] = df[col_name] / 100
            df.rename(columns={col_name: 'soil_moisture'}, inplace=True)
        df = update_soil_name(df, depth)
        df = engineer_features(df, feat_cfg)
        filtered_dfs.append(df)

    evaluate_on_probes(filtered_dfs, model, scaler_x, scaler_y, lookback, horizon,
                        model_name, feat_cfg, model_dir, depth, nb_windows, network_str)


def _fmt_list(arr, prec):
    return ','.join(f'{v:.{prec}f}' for v in arr)


def prepare_osiris_data_for_depth(all_dfs, d, feat_cfg):
    """Prépare les données OSIRIS pour une profondeur donnée: rename, features, split spatial, validation sparse."""
    train_list, val_list, test_list = [], [], []

    valid_dfs = []
    for df in all_dfs:
        df = df.copy()
        col_name = f'humidity_{int(d*100)}cm'
        if col_name in df.columns:
            df[col_name] = df[col_name] / 100
            df.rename(columns={col_name: 'soil_moisture'}, inplace=True)
        df = update_soil_name(df, d)
        df = engineer_features(df, feat_cfg)
        if not all(c in df.columns for c in feat_cfg['sparse']):
            continue
        valid_dfs.append(df)

    site_ids = sorted(list(set(df['site_id'].iloc[0] for df in valid_dfs)))
    np.random.shuffle(site_ids)

    n = len(site_ids)
    n_train = max(1, int(n * 0.7))
    n_val   = max(1, int(n * 0.15))

    for df in valid_dfs:
        sid = df['site_id'].iloc[0]
        if sid in site_ids[:n_train]:
            train_list.append(df)
        elif sid in site_ids[n_train:n_train + n_val]:
            val_list.append(df)
        else:
            test_list.append(df)

    sparse_cols = feat_cfg.get("sparse", [])
    MIN_VAL_LENGTH = 30

    val_clean = []
    for df in val_list:
        has_sparse = not sparse_cols or df[sparse_cols].notna().any().any()
        has_length = len(df) >= MIN_VAL_LENGTH
        if has_sparse and has_length:
            val_clean.append(df)
        else:
            for i, train_df in enumerate(train_list):
                train_has_sparse = not sparse_cols or train_df[sparse_cols].notna().any().any()
                train_has_length = len(train_df) >= MIN_VAL_LENGTH
                if train_has_sparse and train_has_length:
                    val_clean.append(train_list.pop(i))
                    break
    val_list = val_clean

    return train_list, val_list, test_list


def evaluate_on_probes(probe_dfs, model, scaler_x, scaler_y, LB, horizon, model_name, feat_cfg,
                        output_dir, depth, nb_windows, network, metric_prefix='osiris', probe_type='OSIRIS',
                        plot_horizons=None):
    """Évalue un modèle sur une liste de sondes OSIRIS. Logique unique pour les métriques + CSV + plots."""
    is_dl = model_name.replace("_fine_tuned", "") in ["convlstm", "lstm", "transformer", "gru", "tcn"]
    if plot_horizons is None:
        plot_horizons = [0, 6]

    for i, df_L in enumerate(probe_dfs):
        probe_name = df_L['site_id'].iloc[0] + "_" + df_L['no_serie'].iloc[0]
        segments = cut_timeseries(df_L, min_length=LB + horizon + 1, feat_cfg=feat_cfg)
        file_y_true, file_y_pred, file_dates = [], [], []

        for seg in segments:
            X_seg, y_seg, d_seg = make_supervised([seg], scaler_x, scaler_y, LB, horizon, feat_cfg, model_name)
            if len(X_seg) == 0:
                continue
            y_pred_scaled = model.predict(X_seg, verbose=0) if is_dl else model.predict(X_seg)
            file_y_true.append(inverse_y(scaler_y, y_seg))
            file_y_pred.append(inverse_y(scaler_y, y_pred_scaled))
            file_dates.append(d_seg)

        if not file_y_true:
            continue

        y_true = np.concatenate(file_y_true, axis=0)
        y_pred = np.concatenate(file_y_pred, axis=0)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        d_test = np.concatenate(file_dates)

        rmse_h, ubrmse_h, smape_h, r2_h, kge_h, corr_h = compute_horizon_metrics(y_true, y_pred, horizon)
        print(f"[Fichier {i}] LOCAL: RMSE={rmse_h} ubRMSE={ubrmse_h} SMAPE={smape_h}% R²={r2_h} KGE={kge_h} Corr={corr_h}")

        if SAVE_RESULTS_CSV:
            for metricname, val_L, prec in [
                (f'rmse_{metric_prefix}', rmse_h, 6),
                (f'ubrmse_{metric_prefix}', ubrmse_h, 6),
                (f'smape_{metric_prefix}', smape_h, 2),
                (f'r2_{metric_prefix}', r2_h, 4),
                (f'kge_{metric_prefix}', kge_h, 4),
                (f'corr_{metric_prefix}', corr_h, 4)
            ]:
                save_to_results_csv({'type': probe_type + (f"_{probe_name}" if probe_name else ""),
                                    'depth': depth, 'lookback': LB, 'horizon': horizon,
                                    'nb_windows': nb_windows, 'model': model_name, 'metric_name': metricname,
                                    'value': _fmt_list(val_L, prec), 'value_std': '',
                                    'experience_name': feat_cfg['name'], 'network': network},
                                    RESULTS_CSV_PATH, feat_cfg)

        if SAVE_PLOTS:
            for j in plot_horizons:
                if j >= y_pred.shape[1]:
                    continue
                test_png = os.path.join(output_dir, f"test_Grandvillers_COMBINED_{i}_horizon_{j}.png")
                plot_test_true_vs_pred(
                    pd.to_datetime(d_test), y_true[:, j],
                    y_pred_local=y_pred[:, j],
                    out_png=test_png,
                    title=f"Fichier {i} horizon {j} ({len(d_test)} samples)"
                )

    flush_results(RESULTS_CSV_PATH)


def fine_tune_model(model_ft, scaler_x_ft, scaler_y_ft, train_segments, val_segments,
                     is_dl, m, LB, horizon, feat_cfg):
    """Entraîne (fine-tune) un modèle sur les données OSIRIS. Retourne le modèle entraîné."""
    fit_scalers(train_segments, scaler_x_ft, scaler_y_ft, feat_cfg)
    X_ft, y_ft, _ = make_supervised(train_segments, scaler_x_ft, scaler_y_ft, LB, horizon, feat_cfg, m)
    X_val_ft, y_val_ft, _ = make_supervised(val_segments, scaler_x_ft, scaler_y_ft, LB, horizon, feat_cfg, m)

    if is_dl:
        model_ft.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse')
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        val_data = (X_val_ft, y_val_ft) if len(X_val_ft) > 0 else None
        if val_data is None:
            cbs = [cb for cb in cbs if not isinstance(cb, callbacks.ReduceLROnPlateau)]
        model_ft.fit(X_ft, y_ft, validation_data=val_data,
                     epochs=100, batch_size=BATCH_SIZE, callbacks=cbs, verbose=1)
    else:
        model_ft = build_models(m, LB, horizon, feat_cfg)
        model_ft.fit(X_ft, y_ft)

    return model_ft


def osiris_train(feat_cfg, drive_dir, all_dfs):
    """Entraîne des modèles from scratch sur les données OSIRIS (pas de fine-tuning)."""

    depths = feat_cfg["depths"]
    lookbacks = feat_cfg["lookbacks"]
    horizons = feat_cfg["horizons"]
    nb_windows = feat_cfg["nb_windows"]
    model_name = feat_cfg["models"]
    networks = feat_cfg["networks"]
    feature_cols = feat_cfg['dense'] + feat_cfg['soil'] + feat_cfg['sparse']

    features_name = feature_name_path(RESULTS_CSV_PATH, feature_cols)

    site_ids = sorted(list(set(site['site_id'].iloc[0] for site in all_dfs)))
    print(f"Unique site_ids: {site_ids}")

    for d in depths:
        train_list, val_list, test_list = prepare_osiris_data_for_depth(all_dfs, d, feat_cfg)
        print(f"Spatial split: train={len(train_list)} files, val={len(val_list)} files, test={len(test_list)} files")

        for network in networks:
            for LB in lookbacks:
                print(f"--- Lookback: {LB} ---")
                for horizon in horizons:
                    print(f"--- Horizon: {horizon} ---")
                    for NB in nb_windows:
                        print(f"\n=== Training with NB={NB} windows ===")

                        for m in model_name:
                            print(f"\n      > Modèle: {m}")

                            if SAVE_NETWORKS_DIR:
                                output_dir = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}", f"network_{network}")
                            else:
                                output_dir = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}")

                            if SAVE_MODELS_DIR:
                                os.makedirs(output_dir, exist_ok=True)

                            train_eval_predict_one_probe(
                                drive_dir=output_dir, lookback=LB, horizon=horizon, choose_model=m,
                                save_test_plots="first_last",
                                max_windows=NB,
                                max_test_plots=5,
                                depth=d,
                                train_list=train_list, val_list=val_list, test_list=test_list,
                                network_str=network,
                                feat_cfg=feat_cfg
                            )


def osiris_fine_tuning(feat_cfg, drive_dir, all_dfs):

    depths = feat_cfg["depths"]
    lookbacks = feat_cfg["lookbacks"]
    horizons = feat_cfg["horizons"]
    nb_windows = feat_cfg["nb_windows"]
    model_name = feat_cfg["models"]
    networks = feat_cfg["networks"]
    feature_cols = feat_cfg['dense'] + feat_cfg['soil'] + feat_cfg['sparse']

    features_name = feature_name_path(RESULTS_CSV_PATH, feature_cols)

    site_ids = sorted(list(set(site['site_id'].iloc[0] for site in all_dfs)))
    print(f"Unique site_ids: {site_ids}")

    for d in depths:
        train_list, val_list, test_list = prepare_osiris_data_for_depth(all_dfs, d, feat_cfg)
        print(f"Spatial split: train={len(train_list)} files, val={len(val_list)} files, test={len(test_list)} files")

        for network in networks:
            for LB in lookbacks:
                print(f"--- Lookback: {LB} ---")
                for horizon in horizons:
                    print(f"--- Horizon: {horizon} ---")
                    for NB in nb_windows:
                        print(f"\n=== Training with NB={NB} windows ===")

                        for m in model_name:
                            print(f"\n      > Modèle: {m}")

                            if SAVE_NETWORKS_DIR:
                                output_dir = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}", f"network_{network}")
                            else:
                                output_dir = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}")

                            if SAVE_NETWORKS_DIR:
                                output_dir_ft = os.path.join(
                                    drive_dir, features_name, f"depth_{d}", f"lookback_{LB}",
                                    f"horizon_{horizon}", f"nbwindows_{NB}",
                                    f"model_{m}_fine_tuned", f"network_{network}")
                            else:
                                output_dir_ft = output_dir + "_fine_tuned"

                            if SAVE_MODELS_DIR:
                                os.makedirs(output_dir, exist_ok=True)

                            model_ft, scaler_x_ft, scaler_y_ft = load_saved_model_and_scalers(output_dir)
                            if model_ft is None:
                                continue

                            is_dl = m in ["convlstm", "lstm", "transformer", "gru", "tcn"]

                            train_segments = cut_and_filter_dfs(train_list, LB, horizon, 10**6, feat_cfg)
                            val_segments   = cut_and_filter_dfs(val_list, LB, horizon, 10**6, feat_cfg)

                            if not train_segments:
                                print(f"  Pas de segments valides pour LB={LB}, H={horizon}. Skip.")
                                continue

                            model_ft = fine_tune_model(model_ft, scaler_x_ft, scaler_y_ft,
                                                       train_segments, val_segments, is_dl, m, LB, horizon, feat_cfg)

                            if SAVE_MODELS_DIR:
                                os.makedirs(output_dir_ft, exist_ok=True)
                                if is_dl:
                                    model_ft.save(os.path.join(output_dir_ft, "model.keras"))
                                else:
                                    joblib.dump(model_ft, os.path.join(output_dir_ft, "model.pkl"))
                                joblib.dump(scaler_x_ft, os.path.join(output_dir_ft, "scaler_x.pkl"))
                                joblib.dump(scaler_y_ft, os.path.join(output_dir_ft, "scaler_y.pkl"))

                            evaluate_on_probes(test_list, model_ft, scaler_x_ft, scaler_y_ft,
                                                LB, horizon, m, feat_cfg, output_dir_ft, d, NB, network)



def full_eval_osiris(feat_cfg, drive_dir, all_dfs):

    depths = feat_cfg["depths"]
    lookbacks = feat_cfg["lookbacks"]
    horizons = feat_cfg["horizons"]
    nb_windows = feat_cfg["nb_windows"]
    model_name = feat_cfg["models"]
    networks = feat_cfg["networks"]

    feature_cols = feat_cfg['dense'] + feat_cfg['soil'] + feat_cfg['sparse']

    for depth in depths:

        for lookback in lookbacks:
            for horizon in horizons:
                for nb_window in nb_windows : 
                    for m in model_name:
                        
                        for network in networks:

                            features_name = feature_name_path(RESULTS_CSV_PATH, feature_cols)


                            if SAVE_NETWORKS_DIR :
                                model_dir = os.path.join(drive_dir,  features_name, f"depth_{depth}", f"lookback_{lookback}", 
                                                        f"horizon_{horizon}", f"nbwindows_{nb_window}", f"model_{m}", f"network_{network}")
                            else :
                                # Mise à jour du chemin pour suivre l'architecture "Spatial Block Cross-Validation"
                                model_dir = os.path.join(drive_dir,  features_name, f"depth_{depth}", f"lookback_{lookback}", 
                                                        f"horizon_{horizon}", f"nbwindows_{nb_window}", f"model_{m}")
                            
                            if not os.path.exists(model_dir):
                                print(f"Model directory not found: {model_dir}")
                                continue

                            print(f"\n=======================================================")
                            print(f"Evaluation Osiris: D={depth}, LB={lookback}, H={horizon}, NB={nb_window}, Modèle={m}")
                            print(f"Dossier du bloc exploité:  -> {model_dir}")
                            print(f"=======================================================\n")

                            model, scaler_x, scaler_y = load_saved_model_and_scalers(model_dir)

                            if model is None:
                                continue
                            
                            run_osiris_evaluation(
                                model, scaler_x, scaler_y, 
                                all_dfs, 
                                model_dir, 
                                lookback, 
                                horizon, 
                                depth,
                                nb_windows=nb_window,
                                model_name=m,
                                network_str=network,
                                feat_cfg=feat_cfg
                            )
# %% [markdown]
# # Grandvillers

# %%
def Get_Grandvillers_data(project_path: str) -> List[pd.DataFrame]:

    # ------------------------------------------------------------
    # 1. Charger et préparer les données de test UNE SEULE FOIS
    # ------------------------------------------------------------
    test_list = []
    if os.path.exists(project_path):
        print(f"Chargement des fichiers Grandvillers depuis: {project_path}")
        for filename in os.listdir(project_path):
            file_path = os.path.join(project_path, filename)
            if not os.path.exists(file_path):
                continue

            df_local_full = pd.read_csv(file_path)
            if 'date' in df_local_full.columns:
                df_local_full[DATE_COL] = pd.to_datetime(df_local_full['date'])

            # Vérifier s'il y a plusieurs sondes
            probes = df_local_full['probe_name'].unique() if 'probe_name' in df_local_full.columns else [None]

            for probe in probes:
                if probe is not None:
                    df_local = df_local_full[df_local_full['probe_name'] == probe].copy()
                else:
                    df_local = df_local_full.copy()

                df_local.set_index(DATE_COL, inplace=True)
                
                # Conversion des données d'humidité pour toutes les profondeurs
                for depth in [10, 20, 30, 40, 50, 60]:
                    col_name = f'sm_{depth}cm'
                    if col_name in df_local.columns:
                        df_local[col_name] = df_local[col_name] / 100
                        
                if 'WIND' in df_local.columns:
                    df_local['WIND'] = df_local['WIND'] / 0.6

                df_local.attrs['probe_name'] = probe
                
                test_list.append(df_local)
        print(f"-> {len(test_list)} sondes/fichiers chargés et prêts pour l'évaluation.\n")
    else:
        print(f"Avertissement : Le dossier {project_path} n'existe pas.")
    return test_list

# %%
def prepare_era5_dataset(df_local, display_name, lat=49.4727, lon=2.6203):
    """
    Télécharge les données ERA5, affiche la comparaison avec les données locales (graphiques & stats)
    et retourne un nouveau DataFrame avec les features remplacées par celles d'ERA5.
    """
    df_temp = df_local.copy()
    start_date = df_temp.index.min().strftime('%Y-%m-%d')
    end_date = df_temp.index.max().strftime('%Y-%m-%d')
    
    meteo_path = os.path.join(Grandvillers_path, "meteo_daily.csv")
    
    df_era5 = pd.read_csv(meteo_path, parse_dates=["date"])
    df_era5 = df_era5.set_index("date").sort_index()

    # Garder uniquement les dates d'intérêt
    mask = (df_era5.index >= start_date) & (df_era5.index <= end_date)
    df_era5 = df_era5.loc[mask]
    
    df_merged = df_temp.join(df_era5, how='left', rsuffix='_ERA5')
    
    comparison_map = {
        'TMAX': 'TMAX_ERA5',
        'TMIN': 'TMIN_ERA5',
        'RAIN': 'RAIN_ERA5',
        'VAP': 'VAP_ERA5',
        'IRRAD': 'IRRAD_ERA5',
        'WIND': 'WIND_ERA5'
    }
    
    print(f"\n--- Comparaison ({len(df_merged)} jours en commun pour {display_name}) ---")
    vars_to_plot = []
    
    for local_var, era_var in comparison_map.items():
        if local_var in df_merged.columns and era_var in df_merged.columns:
            y_L = df_merged[local_var]
            y_E = df_merged[era_var]
            valid = y_L.notna() & y_E.notna()
            y_L_valid = y_L[valid]
            y_E_valid = y_E[valid]
            
            if len(y_L_valid) > 0:
                rmse = np.sqrt(np.mean((y_L_valid - y_E_valid)**2))
                mae = np.mean(np.abs(y_L_valid - y_E_valid))
                biais = np.mean(y_L_valid - y_E_valid)
                print(f"• {local_var}:  RMSE = {rmse:.2f} | MAE = {mae:.2f} | Biais = {biais:.2f} (Local - ERA5)")
                vars_to_plot.append(local_var)
                
    # 4. Affichage graphique de la comparaison
    if vars_to_plot:
        n_rows = int(np.ceil(len(vars_to_plot) / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, local_var in enumerate(vars_to_plot):
            era_var = comparison_map[local_var]
            axes[i].plot(df_merged.index, df_merged[local_var], label=f'{local_var} (Local)', alpha=0.8, color='black')
            axes[i].plot(df_merged.index, df_merged[era_var], label=f'{era_var} (ERA5)', alpha=0.8, linestyle='--')
            axes[i].set_title(f'Comparaison {local_var} ({display_name})')
            axes[i].legend()
            
        # Cacher les subplots vides s'il y en a
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()

    # Remplacer les valeurs locales par les valeurs ERA5_Land
    df_substituted = df_merged.copy()
    for local_var, era_var in comparison_map.items():
        if era_var in df_substituted.columns:
            # On remplit avec la locale si ERA5 n'est pas dispo ce jour-là
            df_substituted[local_var] = df_substituted[era_var].fillna(df_substituted[local_var])
            df_substituted.drop(columns=[era_var], inplace=True)
            
        
    return df_substituted

# %%
def run_comparative_evaluation(model, scaler_x, scaler_y, lst_local, lst_era5, model_dir, 
                               lookback, horizon, depth, nb_windows=None, model_name="unknown", 
                               network_str="", feat_cfg=None):
    """
    Évalue simultanément les données météo locales et ERA5, calcule les statistiques,
    et trace les deux prédictions sur le même graphique.
    Sauvegarde les résultats dans le CSV centralisé.
    """
    features_cols = feat_cfg['dense'] + feat_cfg['soil'] + feat_cfg['sparse']
    num_valid_files = 0
    
    is_dl = model_name.lower().replace("_fine_tuned", "") in ["convlstm", "lstm", "transformer", "gru", "tcn"]
    
    # On itère parallèlement sur les paires de DataFrames Local / ERA5
    for i, (df_orig_L, df_orig_E) in enumerate(zip(lst_local, lst_era5)):
        df_L = df_orig_L.copy()
        df_E = df_orig_E.copy()
        
        probe_name = df_L.attrs.get('probe_name', None)
        print(f"\n[Fichier {i}] Profondeur={depth}m, Sonde={probe_name}, Taille={len(df_L)} jours")
        # Normalisation du nom de la colonne cible
        col_name = f'sm_{int(depth*100)}cm'
        if col_name in df_L.columns:
            df_L.rename(columns={col_name: 'soil_moisture'}, inplace=True)
            df_E.rename(columns={col_name: 'soil_moisture'}, inplace=True)

        # Les fichiers Grandvillers contiennent déjà les variables de sol,
        # on ne fait ici que normaliser les noms selon la profondeur.
        df_L = update_soil_name(df_L, depth)
        df_E = update_soil_name(df_E, depth)
        
        # Calculer les features dérivées (VPD, T_RANGE, cumuls, etc.)
        df_L[DATE_COL] = df_L.index
        df_E[DATE_COL] = df_E.index
        df_L = engineer_features(df_L, feat_cfg)
        df_E = engineer_features(df_E, feat_cfg)

        missing_cols = [c for c in features_cols if c not in df_L.columns]
        if missing_cols:
            print(f"   Manquantes: {missing_cols}")
        
        if len(df_L) < lookback + horizon:
            continue
        
        # LOCAL : Feature construction & Prediction
        X_L, y_yL, d_dtest = make_supervised([df_L], scaler_x, scaler_y, lookback, horizon, feat_cfg, model_name)
        if len(X_L) == 0:
            continue
            
        if is_dl:
            y_pred_scaled_L = model.predict(X_L, verbose=0)
        else:
            y_pred_scaled_L = model.predict(X_L)
            
        if y_pred_scaled_L.ndim == 1:
            y_pred_scaled_L = y_pred_scaled_L.reshape(-1, horizon) # pour horizon = 1

        y_true = inverse_y(scaler_y, y_yL)
        y_pred_L = inverse_y(scaler_y, y_pred_scaled_L)

        # ERA5 : Feature construction & Prediction (sur la même cible par définition)
        X_E, _, _ = make_supervised([df_E], scaler_x, scaler_y, lookback, horizon, feat_cfg, model_name)
        if is_dl:
            y_pred_scaled_E = model.predict(X_E, verbose=0)
        else:
            y_pred_scaled_E = model.predict(X_E)
            
        if y_pred_scaled_E.ndim == 1:
            y_pred_scaled_E = y_pred_scaled_E.reshape(-1, horizon) # pour horizon = 1

        y_pred_E = inverse_y(scaler_y, y_pred_scaled_E)

        rmse_L_h, ubrmse_L_h, smape_L_h, r2_L_h, kge_L_h, corr_L_h = compute_horizon_metrics(
            y_true, y_pred_L, horizon)
        rmse_E_h, ubrmse_E_h, smape_E_h, r2_E_h, kge_E_h, corr_E_h = compute_horizon_metrics(
            y_true, y_pred_E, horizon)
        print(f"[Fichier {i}] LOCAL: RMSE={rmse_L_h} ubRMSE={ubrmse_L_h} SMAPE={smape_L_h}% R²={r2_L_h} KGE={kge_L_h} Corr={corr_L_h}")
        print(f"[Fichier {i}] ERA5:  RMSE={rmse_E_h} ubRMSE={ubrmse_E_h} SMAPE={smape_E_h}% R²={r2_E_h} KGE={kge_E_h} Corr={corr_E_h}")

        def fmt_list(arr, prec):
            return ','.join(f'{v:.{prec}f}' for v in arr)
        
        if SAVE_RESULTS_CSV:
            # Enregistrer une ligne par fichier/probe_name
            for metricname, val_L, val_E, prec in [
                ('rmse_local', rmse_L_h, rmse_E_h, 6),
                ('ubrmse_local', ubrmse_L_h, ubrmse_E_h, 6),
                ('smape_local', smape_L_h, smape_E_h, 2),
                ('r2_local', r2_L_h, r2_E_h, 4),
                ('kge_local', kge_L_h, kge_E_h, 4),
                ('corr_local', corr_L_h, corr_E_h, 4)
            ]:
                save_to_results_csv({'type': 'GRANDVILLERS' + (f"_{probe_name}" if probe_name else ""), 
                                    'depth': depth, 'lookback': lookback, 'horizon': horizon, 
                                    'nb_windows': nb_windows, 'model': model_name, 'metric_name': metricname, 
                                    'value': fmt_list(val_L, prec), 'value_std': '', 
                                    'experience_name': feat_cfg['name'], 'network' : network_str}, RESULTS_CSV_PATH, feat_cfg)
                
                save_to_results_csv({'type': 'GRANDVILLERS' + (f"_{probe_name}" if probe_name else ""), 
                                    'depth': depth, 'lookback': lookback,
                                    'horizon': horizon, 'nb_windows': nb_windows, 'model': model_name,
                                        'metric_name': metricname.replace('local', 'era5'),
                                        'value': fmt_list(val_E, prec), 'value_std': ''
                                        , 'experience_name': feat_cfg['name'], 'network' : network_str}, RESULTS_CSV_PATH, feat_cfg)

        num_valid_files += 1

        if SAVE_PLOTS :
            # Graphes combinés
            for j in [0,horizon-1]:  # Premier et dernier horizon
                test_png = os.path.join(model_dir, f"test_Grandvillers_COMBINED_{i}_horizon_{j}.png")
                plot_test_true_vs_pred(
                    pd.to_datetime(d_dtest),
                    y_true[:, j],
                    y_pred_local=y_pred_L[:, j],
                    y_pred_era5=y_pred_E[:, j],
                    out_png=test_png,
                    title=f"Local vs ERA5 - Fichier {i} horizon {j} ({len(d_dtest)} samples)"
                )

    flush_results(RESULTS_CSV_PATH)




############ Grandvillers ###############

def full_eval_grandvillers(feat_cfg, drive_dir, test_list_local, test_list_era5):
                                
    depths = feat_cfg["depths"]
    lookbacks = feat_cfg["lookbacks"]
    horizons = feat_cfg["horizons"]
    nb_windows = feat_cfg["nb_windows"]
    model_name = feat_cfg["models"]
    networks = feat_cfg["networks"]

    feature_cols = feat_cfg['dense'] + feat_cfg['soil'] + feat_cfg['sparse']

    for depth in depths:

        for lookback in lookbacks:
            for horizon in horizons:
                for nb_window in nb_windows : 
                    for m in model_name:
                        
                        for network in networks:

                            features_name = feature_name_path(RESULTS_CSV_PATH, feature_cols)


                            if SAVE_NETWORKS_DIR :
                                model_dir = os.path.join(drive_dir,  features_name, f"depth_{depth}", f"lookback_{lookback}", 
                                                        f"horizon_{horizon}", f"nbwindows_{nb_window}", f"model_{m}", f"network_{network}")
                            else :
                                # Mise à jour du chemin pour suivre l'architecture "Spatial Block Cross-Validation"
                                model_dir = os.path.join(drive_dir,  features_name, f"depth_{depth}", f"lookback_{lookback}", 
                                                        f"horizon_{horizon}", f"nbwindows_{nb_window}", f"model_{m}")
                            
                            if not os.path.exists(model_dir):
                                print(f"Model directory not found: {model_dir}")
                                continue

                            print(f"\n=======================================================")
                            print(f"Evaluation Grandvillers : D={depth}, LB={lookback}, H={horizon}, NB={nb_window}, Modèle={m}")
                            print(f"Dossier du bloc exploité:  -> {model_dir}")
                            print(f"=======================================================\n")

                            model, scaler_x, scaler_y = load_saved_model_and_scalers(model_dir)

                            if model is None:
                                continue
                                
                            run_comparative_evaluation(
                                model, scaler_x, scaler_y, 
                                test_list_local, test_list_era5, 
                                model_dir, 
                                lookback, 
                                horizon, 
                                depth,
                                nb_windows=nb_window,
                                model_name=m,
                                network_str=network,
                                feat_cfg=feat_cfg
                            )


# %%

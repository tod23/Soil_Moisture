import numpy as np
import pandas as pd
from typing import Tuple, Dict
from config import TARGET_COL, SEED, DATE_COL



def count_valid_windows(df: pd.DataFrame, lookback: int, horizon: int, feat_cfg: Dict) -> int:
    # Vérifie où il n'y a pas de NaN dans DENSE_FEATURE_COLS et TARGET_COL
    mask_dense = df[feat_cfg["dense"]].notna().all(axis=1)
    mask_soil = df[feat_cfg["soil"]].notna().all(axis=1)
    mask_target = df[[TARGET_COL]].notna().all(axis=1)
    mask = mask_dense & mask_soil & mask_target

    # Compte le nombre de fenêtres valides
    valid_indices = np.where(mask)[0]
    count = max(0, len(valid_indices) - lookback - horizon + 1)
    return count

def count_valid_sparse_windows(df, lookback, horizon, sparse_cols):
    n = len(df)
    starts = np.arange(lookback, n - horizon + 1)
    # Vérifier si les colonnes sparses sont valides à l'index X (start - lookback)
    return df[sparse_cols].iloc[starts - lookback].notna().all(axis=1).sum()

def cut_timeseries(df, min_length=0, feat_cfg=None):
    """
    Découpe une série temporelle (DataFrame) en séquences (DataFrames) sans NaN pour LSTM.
    min_length permet d'ignorer les séquences qui sont trop courtes.
    """
    sequences = []
    mask_tot = None
    # Identifier les valeurs valides
    for col in df.columns:
        if col in feat_cfg["dense"] + feat_cfg["soil"] + [TARGET_COL]:
            mask = df[col].notna()
            mask_tot = mask if mask_tot is None else mask_tot & mask
    
    # Créer un identifiant de groupe qui s'incrémente à chaque présence de NaN
    groups = (~mask_tot).cumsum()
    
    # Grouper les données valides par l'identifiant et ajouter chaque sous-dataframe
    for _, group in df[mask_tot].groupby(groups):
        if not group.empty and len(group) >= min_length:
            sequences.append(group)
    
    return sequences

def cut_and_filter_dfs(df_list, lookback, horizon, max_windows, feat_cfg):
    all_segments = []
    for df in df_list:
        segments = cut_timeseries(df, min_length=lookback + horizon + 1, feat_cfg=feat_cfg)
        all_segments.extend(segments)
    if not all_segments:
        return []

    selected = []
    remaining = max_windows

    # Mélanger les segments pour éviter de toujours prendre les mêmes blocs
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_segments)

    # ── Mode sparse : sélectionner des fenêtres individuelles de façon aléatoire ──
    if feat_cfg["sparse"] and lookback == 1:
        for seg in all_segments:
            if remaining <= 0:
                break
            starts = np.arange(lookback, len(seg) - horizon + 1)
            if len(starts) == 0:
                continue
            valid = seg[feat_cfg["sparse"]].iloc[starts - lookback].notna().all(axis=1)
            valid_starts = starts[valid]
            if len(valid_starts) == 0:
                continue
            rng.shuffle(valid_starts)
            for s in valid_starts:
                if remaining <= 0:
                    break
                selected.append(seg.iloc[s - lookback : s + horizon])
                remaining -= 1
        return selected

    # ── Mode normal : sous-échantillonnage aléatoire des fenêtres valides ──
    candidate_windows = []
    for seg in all_segments:
        n_windows = len(seg) - lookback - horizon + 1
        if n_windows <= 0:
            continue
        for start in range(n_windows):
            candidate_windows.append(seg.iloc[start : start + lookback + horizon])

    if not candidate_windows:
        return []

    rng.shuffle(candidate_windows)
    return candidate_windows[:max_windows]


def make_supervised(
    df_list, scaler_x, scaler_y, lookback, horizon, feat_cfg, model_format
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    sparse_means = getattr(scaler_x, 'sparse_means_', None)
    sparse_stds = getattr(scaler_x, 'sparse_stds_', None)

    X_list, y_list, d_list = [], [], []
    for df in df_list:
        X_dense = scaler_x.transform(
            df[feat_cfg["dense"] + feat_cfg["soil"]].values.astype(np.float32)
        )

         # Scaling sparse avec stats fittées
        X_sparse = np.zeros((len(df), len(feat_cfg["sparse"])), dtype=np.float32)
        for j, col in enumerate(feat_cfg["sparse"]):
            vals = df[col].values.astype(np.float32)
            mask = ~np.isnan(vals)
            if sparse_means is not None and mask.sum() > 0:
                vals[mask] = (vals[mask] - sparse_means[col]) / sparse_stds[col]
            if lookback > 1:
                vals[~mask] = 0.0
            X_sparse[:, j] = vals

        # Concat : dense + sparse + mask
        Xs = np.concatenate([X_dense, X_sparse], axis=1)

        y_raw = df[[TARGET_COL]].values.astype(np.float32)
        ys = scaler_y.transform(y_raw).reshape(-1)
        dates = df[DATE_COL].values


        n = len(df)
        starts = np.arange(lookback, n - horizon + 1)

        # ── FILTRAGE lookback=1 : supprimer les NaN sparses ──
        if lookback == 1 and feat_cfg["sparse"]:
            sparse_col_start = len(feat_cfg["dense"]) + len(feat_cfg["soil"])
            sparse_valid = ~np.isnan(Xs[:, sparse_col_start:]).any(axis=1)
            starts = starts[sparse_valid[starts - lookback]]
            if len(starts) == 0:
                continue


        x_idx = starts[:, None] + np.arange(-lookback, 0)
        y_idx = starts[:, None] + np.arange(horizon)

        X_windows = Xs[x_idx]       # (n_windows, lookback, n_features)
        y_windows = ys[y_idx]       # (n_windows, horizon)
        d_windows = dates[starts]

        if model_format.lower() not in ["convlstm", "lstm", "transformer", "gru", "tcn"]:
            X_windows = X_windows.reshape(len(X_windows), -1)

        X_list.append(X_windows)
        y_list.append(y_windows)
        d_list.append(d_windows)

    if not X_list:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    d = np.concatenate(d_list)

    if model_format.lower() == "convlstm":
        X = X.reshape(X.shape[0], lookback, 1, len(feat_cfg["dense"]) + len(feat_cfg["soil"]) + len(feat_cfg["sparse"]), 1)

    return X, y, d

def fit_scalers(df_list, scaler_x, scaler_y, feat_cfg):
    """Ajuste les scalers sur les données denses (sans générer de fenêtres)."""
    for df in df_list:
        X_dense = df[feat_cfg["dense"] + feat_cfg["soil"]].values.astype(np.float32)
        scaler_x.partial_fit(X_dense)
        y_raw = df[[TARGET_COL]].values.astype(np.float32)
        scaler_y.partial_fit(y_raw)

    sparse_means = {}
    sparse_stds = {}
    for col in feat_cfg["sparse"]:
        all_vals = np.concatenate([df[col].dropna().values for df in df_list])
        sparse_means[col] = all_vals.mean()
        sparse_stds[col] = all_vals.std() if all_vals.std() > 0 else 1.0
    scaler_x.sparse_means_ = sparse_means
    scaler_x.sparse_stds_ = sparse_stds
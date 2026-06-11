# Soil Moisture Project

Ce dépôt contient des scripts Python et des notebooks Jupyter dédiés à l'acquisition, au traitement, à la modélisation et à la visualisation de données d'humidité du sol (Soil Moisture) et de météo.

## Structure du projet

* **Acquisition des données :**
  * `GetDatasetFromISMN.ipynb` et `Fcn_GetDatasetFromISMN.py` : Notebooks pour télécharger et traiter les données topographiques à partir des données ISMN (International Soil Moisture Network).
* **Préparation et Cartographie :**
  * `Dataset_map.ipynb` : Visualisation spatiale / cartographie des jeux de données.
  * `csv_for_hrsm.ipynb` et `Fcn_for_csv.py` : Préparation et formatage des CSV pour les modèles haute résolution (HRSM). Découpage des fichiers ISMN, ajout de données météos, ajout de données topographiques, création d'un CSV pour l'entraînement de modèles.
* **Modélisation et Entraînement :**
  * `Soil_Moisture_Training.ipynb` : Notebook dédié à l'entraînement des modèles d'humidité du sol (détaillé ci-dessous).
* **Analyse des Résultats :**
  * `Results.ipynb` : Interface interactive utilisant `ipywidgets` (avec intégration Google Drive) pour analyser, filtrer et tracer les résultats finaux.
* **Autres :**
  * `HRSM_Grandvillers.ipynb` : Application spécifique des modèles pour la localisation de Grandvillers.
  * `tri_courbes.py` : Script de tri et traitement de courbes.

---

# Soil Moisture Training — Documentation détaillée

## 1. Problème et objectif

**Tâche :** Prédire l'humidité du sol future (`sm_30cm`) à partir d'observations passées.

**Entrée :** Séries temporelles multivariées quotidiennes (7 variables).

**Sortie :** Humidité du sol à une profondeur donnée pour les `horizon` prochains jours.

**Structure des données :** Multiples fichiers CSV, chacun contenant les données d'une région, subdivisée par `probe_name` (station).

---

## 2. Architecture du jeu de données

### 2.1 Organisation des fichiers

```
dataset_training/
└── station_depth_csv/
    └── depth/
        ├── depth_0.1/
        │   ├── station_0/
        │   │   ├── soil_moisture_0.1.csv
        │   │   └── meteo_daily.csv
        │   ├── station_1/
        │   └── ...
        ├── depth_0.3/
        └── depth_0.5/
```

Les chemins sont organisés par **profondeur** (`depth_0.1`, `depth_0.3`, `depth_0.5`), chaque dossier contenant des sous-dossiers par **station**. Chaque station contient un fichier CSV d'humidité du sol et optionnellement un fichier `meteo_daily.csv` pour les variables météorologiques.

### 2.2 Variables d'entrée

Trois catégories de features sont configurées via `FEATURE_CONFIGS` :

| Configuration | Variables |
|---|---|
| `soil_only` | `soil_moisture` uniquement |
| `soil_meteo` | `soil_moisture`, `IRRAD`, `TMIN`, `TMAX`, `VAP`, `WIND`, `RAIN` |
| `soil_meteo_soil` | Variables météo + propriétés du sol (clay, silt, bulk, sand, dem, Saturation, ksat, dem_slope, dem_aspect, dem_twi) |
| `meteo` | Variables météo uniquement (sans soil_moisture en entrée) |

Les propriétés du sol statiques sont chargées depuis `Soil_Properties_Master.csv` et fusionnées par correspondance de coordonnées GPS (latitude/longitude) avec une tolérance de `1e-4`. Les colonnes sont renommées dynamiquement selon la profondeur (`0-30cm`, `30-60cm`, `60-100cm`).

La saisonnalité est encodée via `doy_sin` et `doy_cos` (sinus/cosinus du jour de l'année).

### 2.3 Format des échantillons (X, Y)

Pour chaque échantillon, le modèle reçoit une fenêtre de `window` jours d'observations passées :

```
X.shape = (N_samples, window, 1, 7, 1)   # Pour ConvLSTM2D
          (batch, time, rows, cols, channels)

Y.shape = (N_samples, horizon)            # soil_moisture pour les prochains jours
```

- `window` (lookback) : nombre de jours d'historique (7 ou 14)
- `horizon` : nombre de jours à prédire (7)
- Les 7 variables sont organisées en une "grille pseudo-spatiale" de dimensions `1x7x1`

Pour les modèles non-DL (XGBoost, LightGBM, RandomForest), la fenêtre est aplatie :
```
X.shape = (N_samples, window * nb_features)
```

### 2.4 Split spatial (train/val/test)

Le split est effectué au **niveau des stations** (pas des fichiers) :
- **70%** des stations → train
- **15%** des stations → validation
- **15%** des stations → test

Cette approche garantit que le modèle est évalué sur des stations **totalement indépendantes** jamais vues durant l'entraînement, évitant le data leakage temporel.

---

## 3. Fonctionnement détaillé

### 3.1 Pipeline de chargement des données

1. **`get_file_paths(files_path)`** : Parcourt récursivement les dossiers de profondeur et retourne tous les fichiers `*soil_moisture*.csv`.

2. **`load_csv(csv_path, df_master, depth)`** : Fonction principale de chargement :
   - Standardise la colonne date (`date` → `date_time`)
   - Calcule les features de saisonnalité (`doy_sin`, `doy_cos`)
   - **Resample** les données horaires en journalières (moyenne quotidienne) si la fréquence est infra-journalière
   - **Fusion météo** : charge et merge `meteo_daily.csv` du même dossier
   - **Interpolation** : interpole les valeurs manquantes isolées de `soil_moisture` (limite=1, méthode linéaire)
   - **Propriétés du sol** : appelle `update_soil_property()` pour enrichir avec les données statiques du master CSV
   - Filtre les colonnes pour ne garder que les features configurées

3. **`update_soil_property(df, df_master, cols, depth)`** : Associe les propriétés du sol (clay, silt, sand, bulk, dem, ksat...) à chaque DataFrame en faisant correspondre les coordonnées GPS. Les noms de colonnes du master sont renommés dynamiquement selon la plage de profondeur.

4. **`resample_timeseries(df_temp, freq, method, start_date, end_date, specific_hour)`** : Agrège les séries temporelles à la fréquence désirée (moyenne ou somme). Supporte le filtrage par heure spécifique.

### 3.2 Découpage spatial et fenêtrage

5. **`split_spatial_files(file_paths, lookback, horizon, df_master, depth)`** :
   - Groupe les fichiers par station
   - Mélange aléatoirement les stations (seed fixe pour reproductibilité)
   - Répartit en train/val/test (70/15/15)
   - Charge chaque fichier via `load_csv()` et filtre ceux trop courts

6. **`cut_timeseries(df, col, min_length)`** : Découpe une série temporelle en sous-séquences contiguës sans NaN (les NaN servent de séparateurs). Utile pour les séries avec des trous.

7. **`cut_and_filter_dfs(df_list, lookback, horizon, max_windows)`** :
   - Filtre les lignes avec NaN
   - Découpe en segments contigus via `cut_timeseries()`
   - Trie les segments par taille décroissante
   - Sélectionne jusqu'à `max_windows` fenêtres valides (remplit les plus grands segments en premier)

8. **`count_valid_windows(df, lookback, horizon)`** : Compte le nombre de fenêtres valides (sans NaN) disponibles dans un DataFrame.

### 3.3 Préparation supervisée

9. **`make_supervised(df_list, scaler_x, scaler_y, fit, lookback, horizon, format)`** :
   - **Phase fit** : standardise les features denses et la cible avec `StandardScaler.partial_fit()`
   - **Phase transform** : pour chaque DataFrame :
     - Standardise les features denses (météo + sol) et la cible
     - Standardise les features sparse (moyenne/écart-type sur valeurs disponibles, NaN → 0)
     - Concatène : features denses + sparse + masques
     - Génère les fenêtres glissantes : `X[i-lookback:i]` → `y[i:i+horizon]`
     - Vérifie l'absence de NaN dans chaque fenêtre
     - Pour ConvLSTM : reshape en `(batch, lookback, 1, nb_features, 1)`
     - Pour les autres modèles : aplati ou conserve la forme 2D

### 3.4 Architectures de modèles

#### Modèles Deep Learning (TensorFlow/Keras)

| Modèle | Architecture | Particularité |
|---|---|---|
| **`build_convlstm()`** | 2× ConvLSTM2D (16 filtres, kernel 1×3) + BN + Dropout + Dense 64 → horizon | Traite les features comme une pseudo-image 1×7 |
| **`build_lstm()`** | 3× LSTM (128→64→32) + Dropout + Dense 64 → horizon | Double couche LSTM avec régularisation L2 |
| **`build_transformer()`** | Embedding Dense + 2× Transformer Encoder (MultiHeadAttention, FF) + GlobalAvgPooling | Self-attention sur la dimension temporelle |
| **`build_gru()`** | 3× GRU (128→64→32) + Dropout + Dense → horizon | Variante plus légère du LSTM |
| **`build_tcn()`** | 3× Conv1D causal (dilation 1,2,4) + BN + GlobalAvgPooling | Réseau convolutif temporel |

Tous les modèles DL sont compilés avec **Adam (lr=1e-3)** et **loss MSE**.

#### Modèles non-DL (scikit-learn / XGBoost / LightGBM)

| Modèle | Hyperparamètres |
|---|---|
| **`build_xgboost()`** | n_estimators=200, max_depth=7, lr=0.1, subsample=0.8, early_stopping=20 |
| **`build_lightgbm()`** | n_estimators=200, max_depth=7, lr=0.1, wrapped in `MultiOutputRegressor` |
| **`build_random_forest()`** | Optimisé via Optuna (n_estimators, max_depth, min_samples_split) sur 10 trials |

### 3.5 Entraînement

10. **`train_eval_predict_one_probe(file_paths, drive_dir, lookback, horizon, choose_model, save_test_plots, max_windows, max_test_plots, df_master, depth)`** : Fonction centrale qui orchestre tout le pipeline :

    1. **Split spatial** → listes de DataFrames train/val/test
    2. **Filtrage** par `max_windows` (80% train, 20% val pour les DL)
    3. **Standardisation** via `make_supervised()` avec fit sur train uniquement
    4. **Construction du modèle** via `build_models()`
    5. **Entraînement** :
       - **DL** : 150 epochs max, batch_size=32, callbacks :
         - `ModelCheckpoint` (sauvegarde meilleur modèle sur val_loss)
         - `EarlyStopping` (patience=20, restore best weights)
         - `ReduceLROnPlateau` (facteur 0.5, patience=7, min_lr=1e-5)
         - `CSVLogger` (historique de loss)
       - **XGBoost** : `fit()` avec `eval_set=(X_val, y_val)`, early stopping intégré
       - **LightGBM/RF** : `fit()` standard
    6. **Sauvegarde** : modèle (.keras ou .pkl), scalers (joblib), courbes de loss (PNG)
    7. **Évaluation test** : pour chaque fichier test, génère les prédictions, calcule les métriques par horizon, sauvegarde les graphiques true vs predicted

### 3.6 Boucle d'entraînement exhaustive

11. **`full_training(DEPTHS, LOOKBACK, HORIZONS, NB_WINDOWS, MODELS, base_path, drive_dir)`** : Boucle principale qui itère sur toutes les combinaisons :

```
pour chaque profondeur d dans [0.1, 0.3, 0.5] :
  pour chaque lookback LB dans [7, 14] :
    pour chaque horizon H dans [7] :
      pour chaque nb_windows NB dans [5000, 10000, 15000] :
        pour chaque modèle M dans [lstm, transformer, xgboost, lightgbm] :
          pour chaque trial (n_trials répétitions) :
            entraîner et évaluer
          agréger les métriques (moyenne ± écart-type)
          sauvegarder dans results.csv
```

Les résultats sont organisés dans une arborescence de sortie :
```
outputs/station_depth_csv/
└── features_{id}/
    └── depth_{d}/
        └── lookback_{LB}/
            └── horizon_{H}/
                └── nbwindows_{NB}/
                    └── model_{M}/
                        └── trial_{trial}/
                            ├── model.keras (ou .pkl)
                            ├── scaler_x.pkl
                            ├── scaler_y.pkl
                            ├── history.csv
                            ├── loss_train_val.png
                            └── test_true_vs_pred_file_{i}_horizon_{j}.png
```

### 3.7 Métriques d'évaluation

12. **`compute_horizon_metrics(y_true, y_pred, horizon)`** : Calcule 5 métriques pour chaque pas d'horizon :
    - **MAE** : Mean Absolute Error
    - **RMSE** : Root Mean Squared Error
    - **MAPE** : Mean Absolute Percentage Error (avec protection division par zéro)
    - **SMAPE** : Symmetric Mean Absolute Percentage Error
    - **R²** : Coefficient de détermination

### 3.8 Visualisation

13. **`plot_loss(history, out_png, title)`** : Courbe d'apprentissage (loss train/val par epoch).
14. **`plot_test_true_vs_pred(dates, y_true, y_pred_local, y_pred_era5, out_png, title)`** : Comparaison temporelle des prédictions vs observations réelles.

### 3.9 Sauvegarde des résultats

15. **`save_to_results_csv(results_dict, csv_path)`** : Sauvegarde chaque configuration et ses métriques dans un fichier CSV centralisé `results.csv` avec les colonnes : `timestamp`, `features`, `type`, `depth`, `lookback`, `horizon`, `nb_windows`, `model`, `metric_name`, `value`, `value_std`, `n_trials`.

16. **`feature_name_path(results_csv_path, feature_cols)`** : Gère les IDs de configuration de features pour éviter les collisions dans l'arborescence de sortie.

---

## 4. Hyperparamètres configurables

| Paramètre | Valeurs | Description |
|---|---|---|
| `DEPTHS` | `[0.1, 0.3, 0.5]` | Profondeurs du capteur (m) |
| `LOOKBACK` | `[7, 14]` | Fenêtre d'observation passée (jours) |
| `HORIZONS` | `[7]` | Horizon de prédiction (jours) |
| `NB_WINDOWS` | `[5000, 10000, 15000]` | Nombre de fenêtres d'entraînement |
| `MODELS` | `["lstm", "transformer", "xgboost", "lightgbm"]` | Modèles à entraîner |
| `EPOCHS` | `150` | Nombre maximal d'époques |
| `BATCH_SIZE` | `32` | Taille de batch |
| `SEED` | `8` | Seed aléatoire pour reproductibilité |

---

## 5. Notes techniques

- **Environnement d'exécution** : Conçu pour Google Colab avec montage Google Drive (`/content/gdrive`)
- **Dépendances** : tensorflow, xgboost, lightgbm, scikit-learn, optuna, pandas, numpy, matplotlib, joblib
- **Gestion des NaN** : Plusieurs niveaux de filtrage — interpolation limitée, découpage aux points de rupture NaN, vérification stricte dans `make_supervised()`
- **Arrêt précoce** : `ReduceLROnPlateau` réduit le learning rate d'un facteur 0.5 si la loss de validation stagne (patience=7), et `EarlyStopping` arrête après 20 époques sans amélioration

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ismn.meta import Depth

def read_and_clean_data(sensor):
    df = sensor.read_data()
    df.loc[df['soil_moisture_flag'] != 'G', 'soil_moisture'] = np.nan
    return df

def resample_timeseries(df_temp, freq='D', method='mean', start_date=None, end_date=None, specific_hour=None):
    """
    Rééchantillonne une série temporelle.

    Paramètres:
    -----------
    df : pandas.DataFrame
        DataFrame contenant une colonne 'DateTime' ou ayant un DatetimeIndex.
    freq : str, defaut 'D'
        Fréquence ('D' pour Daily, 'H' pour Hourly, 'W' pour Weekly, etc.).
    method : str, defaut 'mean'
        Méthode d'agrégation ('mean', 'sum', 'max', 'min', 'first', 'last').
    """
    
    # 2. Filtrer la période
    if start_date is not None:
        df_temp = df_temp.loc[start_date:]
    if end_date is not None:
        df_temp = df_temp.loc[:end_date]
        
    # 3. Rééchantillonnage et application de la méthode
    if freq == 'D' and specific_hour is not None:
        # Prendre uniquement l'heure spécifique de chaque jour
        df_temp = df_temp[df_temp.index.hour == specific_hour]
        # Rééchantillonner pour garantir qu'on a bien un pas de temps par jour 
        # (les jours manquants seront remplis par NaN)
        df_resampled = df_temp.resample('D').first()
    else:
        # Appliquer une méthode classique (moyenne, max, etc.) sur la fréquence choisie
        resampler = df_temp.resample(freq)
        
        if method == 'mean':
            df_resampled = resampler.mean(numeric_only=True)
        elif method == 'sum':
            df_resampled = resampler.sum(numeric_only=True)
        elif method == 'max':
            df_resampled = resampler.max(numeric_only=True)
        elif method == 'min':
            df_resampled = resampler.min(numeric_only=True)
        elif method == 'first':
            df_resampled = resampler.first()
        elif method == 'last':
            df_resampled = resampler.last()
        else:
            raise ValueError(f"Méthode '{method}' non reconnue.")
    return df_resampled


def filter_data(ismn_data, var = 'soil_moisture', Climate = ['Cfb'], land_cover = [10], frequency = 'H', depth_from = 0., depth_to = 0.05):
    filtered_sensors = []

    if depth_from is not None and depth_to is not None :
        for _, _, sensor in ismn_data.collection \
            .iter_sensors(variable=var,
                        depth=Depth(depth_from,depth_to),
                        filter_meta_dict={'lc_2010': land_cover,
                                            'climate_KG':Climate}):
            freq = detect_time_frequency(sensor.read_data())
            if freq != frequency:
                continue
            filtered_sensors.append(sensor)
    else :
        for _, _, sensor in ismn_data.collection \
            .iter_sensors(variable=var,
                        # depth=Depth(depth_from,depth_to),
                        filter_meta_dict={'lc_2010': land_cover,
                                            'climate_KG':Climate}):
            freq = detect_time_frequency(sensor.read_data())
            if freq != frequency:
                continue
            filtered_sensors.append(sensor)
    return filtered_sensors

def interpolate_timeseries(df, col='soil_moisture', n=1, method='linear'):
    """
    Interpole les valeurs manquantes d'une série temporelle pour les valeurs isolées.
    Seules les valeurs manquantes entourées de données valides seront interpolées.

    Paramètres:
    -----------
    df : pandas.DataFrame
        DataFrame contenant une colonne 'Value' avec des valeurs manquantes.
    col : str, defaut 'soil_moisture'
        Nom de la colonne à interpoler.
    method : str, defaut 'linear'
        Méthode d'interpolation ('linear', 'polynomial', 'spline', etc.).
    n : longueur maximale des séquences de NaN à interpoler (par défaut 1, pour n'interpoler que les valeurs isolées).
    Retour:
    --------
    pandas.DataFrame
        DataFrame avec les valeurs manquantes interpolées.
    """

    df_interpolated = df.copy()
    df_interpolated[col] = df_interpolated[col].interpolate(method=method, limit=n)
    return df_interpolated


def cut_timeseries(df, col='soil_moisture', min_length=0):
    """
    Découpe une série temporelle (DataFrame) en séquences (DataFrames) sans NaN pour LSTM.
    min_length permet d'ignorer les séquences qui sont trop courtes.
    """
    sequences = []
    
    # Identifier les valeurs valides
    mask = df[col].notna()
    
    # Créer un identifiant de groupe qui s'incrémente à chaque présence de NaN
    groups = (~mask).cumsum()
    
    # Grouper les données valides par l'identifiant et ajouter chaque sous-dataframe
    for _, group in df[mask].groupby(groups):
        if not group.empty and len(group) >= min_length:
            sequences.append(group)
    
    return sequences

def create_clusters_dict(sensor_list, method='soil_type', verbose=False):

    features_list = []
    Indices = []
    if method == 'soil_type':

        for idx, sensor in enumerate(sensor_list):
            aux = sensor.metadata.to_pd()
            
            # Utilisation de .get() sur le MultiIndex. 
            # Si la variable n'existe pas, retourne np.nan pour éviter l'erreur.
            clay_fraction = aux.get(('clay_fraction', 'val'), np.nan)
            silt_fraction = aux.get(('silt_fraction', 'val'), np.nan)
            sand_fraction = aux.get(('sand_fraction', 'val'), np.nan)
            if pd.isna(clay_fraction) and pd.isna(silt_fraction) and pd.isna(sand_fraction):
                print(f"Station: {sensor} | Propriétés du sol manquantes. Valeurs utilisées : Clay={clay_fraction}, Silt={silt_fraction}, Sand={sand_fraction}"
                      )
                continue
            features_list.append([clay_fraction, silt_fraction, sand_fraction])
            Indices.append(idx) 
        df_features = pd.DataFrame(features_list, columns=[ 'Clay', 'Silt', 'Sand'])

    elif method == 'statistical':

        for idx, sensor in enumerate(sensor_list):
            ts = sensor.read_data()
            # Selon la version de ismn, read_data retourne un ISMNTimeSeries ou un DataFrame (ici déjà DataFrame)
            data = ts.data if hasattr(ts, 'data') else ts
            # Gestion des valeurs manquantes
            data.loc[data['soil_moisture_flag'] != 'G', 'soil_moisture'] = np.nan

            valid_values = data['soil_moisture'].dropna()

            # 1. Extraction des caractéristiques
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            min_val = valid_values.min()
            max_val = valid_values.max()
            median_val = valid_values.median()
            skew_val = valid_values.skew()  # Asymétrie de la courbe
            
            Indices.append(idx)
            features_list.append([mean_val, std_val, min_val, max_val, median_val, skew_val])

        # 2. Création d'un DataFrame de caractéristiques
        df_features = pd.DataFrame(features_list, columns=['Mean', 'Std', 'Min', 'Max', 'Median', 'Skew'])

    df_features.index = Indices

    # 4. Normalisation : Très important pour que KMeans ne favorise pas les métriques aux grandes valeurs
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_features)

    # 5. Application de K-Means
    n_clusters = 4 # Vous pouvez ajuster ce nombre
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_features['Cluster'] = kmeans.fit_predict(features_scaled)

    if verbose :
        print("\n--- Distribution des clusters ---")
        print(df_features['Cluster'].value_counts())

        # Affichage des statistiques moyennes pour chaque Cluster pour aider à les interpréter
    
        print("\n--- Profil moyen des clusters ---")
        print(df_features.groupby('Cluster').mean())

    # Création d'un dictionnaire pour lister facilement quelles stations vont dans quel groupe
    # clusters_dict[0] retournera la liste de toutes les stations du cluster 0
    clusters_dict = {i: df_features[df_features['Cluster'] == i].index.tolist() for i in range(n_clusters)}

    return clusters_dict, sensor_list, df_features


def find_abnormal_sequences(df, column='Value', variance_threshold=1e-5, min_constant_consecutive=100):
    """
    Identifie si une séquence entière est anormale.
    Une séquence est considérée anormale si une trop grande portion de celle-ci 
    est bloquée sur des valeurs constantes ou presque constantes.
    
    Paramètres:
    -----------
    df : pandas.DataFrame
        DataFrame contenant une colonne 'Value' avec des valeurs numériques.
    column : str, defaut 'Value'
        Le nom de la colonne à analyser.
    variance_threshold : float, defaut 1e-5
        Seuil de variance (écart-type) en dessous duquel on considère la série "constante".
    min_constant_consecutive : int, defaut 100
        Nombre minimal d'étapes consécutives où la variance doit être inférieure au seuil pour être considérée constante.
    percent_threshold : float, defaut 0.5
        Pourcentage (de 0 à 1) de points "constants" au-delà duquel toute la courbe est jugée anormale.
        
    Retour:
    --------
    bool
        True si la courbe est anormale (et doit être rejetée), False si elle est saine.
    """
    if len(df) < min_constant_consecutive:
        return False
        
    rolling_std = df[column].rolling(window=min_constant_consecutive, min_periods=min_constant_consecutive).std()
    abnormal_mask = rolling_std < variance_threshold
    
    # Pourcentage de la séquence qui est considéré comme anormal (constant)
    percent_constant = abnormal_mask.sum() / len(df)
    
    return percent_constant


def filter_outliers(df, column='Value', min_threshold=0.0, max_threshold=1.0):
    """
    Filtre les valeurs aberrantes (outliers) d'un DataFrame.
    Les valeurs considérées comme aberrantes sont remplacées par des NaN.
    
    Paramètres:
    -----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    column : str, defaut 'Value'
        Le nom de la colonne sur laquelle appliquer le filtre.
    min_threshold, max_threshold : float
        Seuils stricts. Toute valeur <= min_threshold ou >= max_threshold devient NaN.
        
    Retour:
    --------
    tuple : (pandas.DataFrame, float)
        DataFrame avec les outliers remplacés par NaN, et le pourcentage d'outliers filtrés.
    """
    df_filtered = df.copy()
    
    # Nombre de valeurs valides initialement
    initial_valid = df_filtered[column].notna().sum()
    
    # 1. Filtres Min et Max
    if min_threshold is not None:
        df_filtered.loc[df_filtered[column] <= min_threshold, column] = np.nan
    if max_threshold is not None:
        df_filtered.loc[df_filtered[column] >= max_threshold, column] = np.nan
        
    # Calcul du nombre et du pourcentage d'outliers
    final_valid = df_filtered[column].notna().sum()
    outliers_removed = initial_valid - final_valid
    percent_outliers = (outliers_removed / initial_valid * 100) if initial_valid > 0 else 0
    
    return df_filtered, percent_outliers



def save_to_csv(sequences, base_path):
    """
    Enregistre une liste de DataFrames (séquences) en fichiers CSV à un chemin spécifique.
    Si base_path est 'dossier/sequence.csv', les fichiers seront nommés 
    'dossier/sequence_0.csv', 'dossier/sequence_1.csv', etc.
    
    Paramètres:
    -----------
    sequences : list of pandas.DataFrame
        Liste de DataFrames à enregistrer.
    base_path : str
        Chemin de base pour l'enregistrement (ex: 'data/ma_station_sequence.csv').
    """
    import os
    
    # Créer le dossier s'il n'existe pas
    directory = os.path.dirname(base_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    # # Séparer l'extension pour ajouter l'index
    # base, ext = os.path.splitext(base_path)
    # if not ext:
    #     ext = '.csv'
        
    saved_paths = []
    for i, df in enumerate(sequences):
        file_path = f"{base_path}_{i}{'.csv'}"
        df.to_csv(file_path, header=True)
        saved_paths.append(file_path)
        
    print(f"{len(sequences)} séquences enregistrées sous : {base_path}_X{'.csv'}")
    return saved_paths

########################################################################################################################
########################################################################################################################
########################################            Données Météo        ###############################################
########################################################################################################################
########################################################################################################################

def get_meteo_data(lat, lon, start_date, end_date):
    """
    Récupère les données météorologiques journalières de la réanalyse ERA5-Land (via Google Earth Engine) 
    pour une localisation et une période données.
    
    Colonnes retournées : IRRAD, TMIN, TMAX, WIND, RAIN, VAP
    
    Pré-requis:
    -----------
    Avoir installé earthengine-api (pip install earthengine-api)
    S'être authentifié au moins une fois (dans un terminal : earthengine authenticate)
    """
    import ee
    
    try:
        # Initialisation de l'API GEE
        ee.Authenticate()
        ee.Initialize()
    except Exception as e:
        print(e)
        return pd.DataFrame()
        
    start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # Point d'intérêt
    point = ee.Geometry.Point([lon, lat])
    
    # Collection ERA5-Land Daily Aggregated
    collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                .filterBounds(point) \
                .filterDate(start_str, end_str) \
                .select([
                    'temperature_2m_max', 
                    'temperature_2m_min', 
                    'total_precipitation_sum',
                    'surface_solar_radiation_downwards_sum',
                    'u_component_of_wind_10m_max',
                    'v_component_of_wind_10m_max',
                    'dewpoint_temperature_2m_min'
                    ])

    # Extraire les métadonnées de la collection pour ce point sous forme de FeatureCollection
    # Scale dépend de la résolution : ERA5-Land = 11132m
    def get_data_for_point(image):
        reduced = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132
        )
        # Assigner la date de l'image (prise à partir des propriétés temporelles)
        return ee.Feature(None, reduced).set('system:time_start', image.get('system:time_start'))
        
    features = collection.map(get_data_for_point).getInfo()['features']
    
    if not features:
        return pd.DataFrame()
    
    # Transformer en DataFrame Pandas
    data_list = []
    for f in features:
        props = f['properties']
        dt = pd.to_datetime(props['system:time_start'], unit='ms')
        data_list.append({
            'DateTime': dt,
            'TMAX': props.get('temperature_2m_max', np.nan),
            'TMIN': props.get('temperature_2m_min', np.nan),
            'RAIN_ERA': props.get('total_precipitation_sum', np.nan),
            'IRRAD_ERA': props.get('surface_solar_radiation_downwards_sum', np.nan),
            'U10_max': props.get('u_component_of_wind_10m_max', np.nan),
            'V10_max': props.get('v_component_of_wind_10m_max', np.nan),
            'TDEW': props.get('dewpoint_temperature_2m_min', np.nan)
        })
        
    df_meteo = pd.DataFrame(data_list)
    df_meteo.set_index('DateTime', inplace=True)
    
    # Conversions des unités pour coller au standard habituel
    # Températures K -> °C
    df_meteo['TMAX'] = df_meteo['TMAX'] - 273.15
    df_meteo['TMIN'] = df_meteo['TMIN'] - 273.15
    df_meteo['TDEW'] = df_meteo['TDEW'] - 273.15
    
    # Pluie (m -> mm)
    df_meteo['RAIN'] = df_meteo['RAIN_ERA'] * 1000 
    
    # Irradiance (J/m² -> MJ/m²)
    df_meteo['IRRAD'] = df_meteo['IRRAD_ERA'] / 1e6 
    
    # Vitesse du vent (magnitude à partir des vecteurs u et v max, en m/s)
    df_meteo['WIND'] = np.sqrt(df_meteo['U10_max']**2 + df_meteo['V10_max']**2)
    
    # Afin de comparer IRRAD et WIND on doit accorder les unités
    # Il semble que la station mesure IRRAD en kJ/m2 (ERA5 est en MJ/m2)
    # Et WIND en km/h (ERA5 a été calculé en m/s)
    df_meteo['IRRAD'] = df_meteo['IRRAD'] * 1000 
    df_meteo['WIND'] = df_meteo['WIND'] * 3.6
    # - FAO 10m→2m 
    df_meteo['WIND'] = df_meteo['WIND'] * 0.748
    # Vapor Pressure (VAP) à partir du Dew Point: Formule de Tetens pour VAP actuel
    # VAP en kilopascals (kPa)
    df_meteo['VAP'] = 0.6108 * np.exp((17.27 * df_meteo['TDEW']) / (df_meteo['TDEW'] + 237.3))
    
    # Nettoyage
    df_meteo = df_meteo[['IRRAD', 'TMIN', 'TMAX', 'WIND', 'RAIN', 'VAP']]
    
    return df_meteo






def get_meteo_data_hourly(lat, lon, start_date, end_date):
    import ee

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    point = ee.Geometry.Point([lon, lat])

    dfs = []
    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(months=6), end)
        cs = current.strftime('%Y-%m-%d')
        ce = chunk_end.strftime('%Y-%m-%d')

        collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
                       .filterBounds(point) \
                       .filterDate(cs, ce) \
                       .select([
                           'temperature_2m',
                           'dewpoint_temperature_2m',
                           'total_precipitation',
                           'surface_solar_radiation_downwards',
                           'u_component_of_wind_10m',
                           'v_component_of_wind_10m'
                       ])

        def get_data_for_point(image):
            reduced = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=11132
            )
            return ee.Feature(None, reduced).set(
                'system:time_start', image.get('system:time_start')
            )

        features = collection.map(get_data_for_point).getInfo()['features']
        if features:
            for f in features:
                props = f['properties']
                dt = pd.to_datetime(props['system:time_start'], unit='ms')
                dfs.append({
                    'DateTime': dt,
                    'T2M': props.get('temperature_2m', np.nan),
                    'TDEW': props.get('dewpoint_temperature_2m', np.nan),
                    'RAIN': props.get('total_precipitation', np.nan),
                    'IRRAD_ERA': props.get('surface_solar_radiation_downwards', np.nan),
                    'U10': props.get('u_component_of_wind_10m', np.nan),
                    'V10': props.get('v_component_of_wind_10m', np.nan),
                })

        current = chunk_end

    if not dfs:
        return pd.DataFrame()

    df = pd.DataFrame(dfs).set_index('DateTime').sort_index()

    # K -> °C
    df['T2M'] -= 273.15
    df['TDEW'] -= 273.15
    # m -> mm
    df['RAIN'] *= 1000
    # J/m² -> kJ/m²
    df['IRRAD'] = df['IRRAD_ERA'] / 1000
    # m/s -> km/h
    df['WIND'] = np.sqrt(df['U10']**2 + df['V10']**2) * 3.6 * 0.748  # + FAO 10m→2m

    # Pression de vapeur
    df['VAP'] = 0.6108 * np.exp((17.27 * df['TDEW']) / (df['TDEW'] + 237.3))

    return df[['T2M', 'WIND', 'RAIN', 'IRRAD', 'VAP']]




def detect_time_frequency(data):
    """
    Détecte la fréquence (journalière ou horaire) d'une série temporelle.
    
    Paramètres:
    -----------
    df : pandas.DataFrame
        DataFrame avec un DatetimeIndex ou une colonne 'DateTime'.
        
    Retour:
    --------
    str
        'D' pour Daily (journalier), 'H' pour Hourly (horaire), ou 'Unknown' si non reconnu.
    """

    if 'soil_moisture_flag' in data.columns:
        valid_data = data[data['soil_moisture_flag'] == 'G']
    else:
        valid_data = data.dropna(subset=['soil_moisture'])

    # Calcul de la fréquence en prenant le mode de la différence entre timestamps valides
    if len(valid_data) > 1:
        time_diffs = valid_data.index.to_series().diff()
        mode_diff = time_diffs.mode()
        
        if not mode_diff.empty:
            freq = mode_diff[0]
            if freq == pd.Timedelta(hours=1):
                return 'H'
            elif freq == pd.Timedelta(days=1):
                return 'D'
            else:
                return f"Unknown frequency: {str(freq)}"
    else:
        return f"invalid data length: {len(valid_data)}"


########################################################################################################################
########################################################################################################################
##############################            Données Topographiques         ###############################################
########################################################################################################################
########################################################################################################################
import requests
import rasterio
import numpy as np
import tempfile
import time
import re
import os
import pandas as pd
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from whitebox import WhiteboxTools


zenodo_list = ["10.5281/zenodo.14920387", "10.5281/zenodo.3935359"]
BASE_DEFAULT = "http://s3.eu-central-1.wasabisys.com/stac/openlandmap"
TYPE_OPTIONS = {"silt", "clay", "sand", "bulk", "dem", "ksat"}
STAT_OPTIONS = {"m", "p16", "p84"}
RESOLUTION_OPTIONS = {"30m", "120m"}
TYPE_KEYWORDS = {
    "silt": {"silt", "silty", "limon", "limoneux"},
    "clay": {"clay", "clayey", "argile", "argileux"},
    "sand": {"sand", "sandy", "sable", "sableux"},
    "bulk": {"bulk"},
    "dem": {"dem"},
    "ksat": {"ksat", "Ksat"},
}

def parse_soil_types(soil_type):
    if isinstance(soil_type, (list, tuple, set)):
        values = [str(v).strip().lower() for v in soil_type if str(v).strip()]
    else:
        raw = str(soil_type).strip().lower()
        if not raw:
            values = []
        else:
            values = [v.strip() for v in re.split(r"[,;|+]", raw) if v.strip()]

    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values

def normalize_bbox(box_zone):
    if len(box_zone) != 4:
        raise ValueError("box_zone doit contenir 4 valeurs: [west, south, east, north]")

    west, south, east, north = box_zone
    return [min(west, east), min(south, north), max(west, east), max(south, north)]

def normalize_choice(value, allowed_values, label):
    normalized = str(value).strip().lower()
    if normalized not in allowed_values:
        raise ValueError(
            f"{label} invalide: {value!r}. Valeurs autorisees: {sorted(allowed_values)}"
        )
    return normalized

def configure_inputs(
    soil_type="silt",
    stat="m",
    resolution="30m",
    box_zone=None,
    base=BASE_DEFAULT,
    strict=False,
):
    if box_zone is None:
        box_zone = [2.59, 49.44, 2.63, 49.48]

    soil_type = parse_soil_types(soil_type)
    if not soil_type:
        raise ValueError("type vide: fournis au moins un type (ex: 'silt' ou ['silt','clay'])")

    if strict:
        for one_type in soil_type:
            normalize_choice(one_type, TYPE_OPTIONS, "type")
        stat = normalize_choice(stat, STAT_OPTIONS, "moyenne")
        resolution = normalize_choice(resolution, RESOLUTION_OPTIONS, "resolution")
    else:
        stat = str(stat).strip().lower()
        resolution = str(resolution).strip().lower()

    return {
        "base": base,
        "type": soil_type,
        "stat": stat,
        "resolution": resolution,
        "box_zone": normalize_bbox(box_zone),
    }

def text_contains_any_keyword(text, keywords):
    for keyword in keywords:
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def discover_collections(base, soil_type):
    catalog = requests.get(base + "/catalog.json").json()
    links = [link for link in catalog.get("links", []) if link.get("rel") == "child"]
    soil_types = parse_soil_types(soil_type)
    filtered = []
    for link in links:
        text = (link.get("title", "") + " " + link.get("href", "")).lower()
        if any(text_contains_any_keyword(text, TYPE_KEYWORDS.get(one_type, {one_type})) for one_type in soil_types):
            filtered.append(link)
    return filtered


def normalize_collection_href(base_href, href):
    if href.startswith("./"):
        return f"{base_href.rstrip('/')}/{href[2:]}"
    return href


def normalize_child_href(parent_href, href):
    if href.startswith("./"):
        return f"{parent_href.rsplit('/', 1)[0]}/{href[2:]}"
    return href



def intersects_bbox(item_bbox, bbox):
    if not item_bbox:
        return False

    return not (
        item_bbox[2] < bbox[0]
        or item_bbox[0] > bbox[2]
        or item_bbox[3] < bbox[1]
        or item_bbox[1] > bbox[3]
    )


def infer_mode_token(text):
    normalized_text = str(text).lower()
    for token in ("p16", "p84", "m"):
        if f"_{token}_" in normalized_text or normalized_text.endswith(f"_{token}"):
            return token
    return "m"

def extract_mode(href_asset):
    normalized_text = str(href_asset).lower()
    
    if ".wpct" in normalized_text:
        after_wpct = normalized_text.split(".wpct", 1)[-1]
        cleaned = after_wpct.lstrip("/_")
        first_segment = cleaned.split("/", 1)[0]
        if not first_segment:
            return None
        
        candidate = first_segment.split("_", 1)[0].split("?", 1)[0]
        if candidate in {"m", "p16", "p84"}:
            return candidate
        
    elif ".cm3" in normalized_text:
        after_cm3 = normalized_text.split(".cm3", 1)[-1]
        cleaned = after_cm3.lstrip("/_")
        first_segment = cleaned.split("/", 1)[0]
        if not first_segment:
            return None
         
        candidate = first_segment.split("_", 1)[0].split("?", 1)[0]
        if candidate in {"m", "p16", "p84"}:
            return candidate

    return infer_mode_token(normalized_text)



def asset_matches_resolution(href_asset, resolution):
    return resolution in href_asset.lower()


def get_assets_before_and_after_filters(params, collections, verbose=False):
    modes_autorises = {params["stat"]}
    all_assets = []
    valid_assets = []

    for lien in collections:
        collection_href = normalize_collection_href(params["base"], lien["href"])
        collection = requests.get(collection_href).json()
        collection_title = lien.get("title", "Sans titre")
        items = [link for link in collection.get("links", []) if link.get("rel") == "item"]

        if verbose:
            print(f"COLLECTION : {collection_title}")
            print(f"ITEMS      : {len(items)}")

        for item_lien in items:
            item_href = normalize_child_href(collection_href, item_lien["href"])
            item_data = requests.get(item_href).json()
            item_bbox = item_data.get("bbox")
            if not intersects_bbox(item_bbox, params["box_zone"]):
                continue

            for asset_name, asset in item_data.get("assets", {}).items():
                href_asset = asset.get("href", "")
                if not href_asset.lower().endswith((".tif", ".tiff")):
                    continue

                mode = extract_mode(href_asset)
                asset_record = {
                    "collection": collection_title,
                    "item": item_data.get("id"),
                    "asset": asset_name,
                    "mode": mode,
                    "resolution": params["resolution"],
                    "href": href_asset,
                }
                all_assets.append(asset_record)

                # if not asset_matches_type(asset_name, href_asset, params["type"]):
                #     continue
                if not asset_matches_resolution(href_asset, params["resolution"]):
                    continue
                if mode not in modes_autorises:
                    continue

                valid_assets.append(asset_record)

    return all_assets, valid_assets


def get_valid_assets(params, collections, verbose=False):
    _, valid_assets = get_assets_before_and_after_filters(params, collections, verbose=verbose)
    return valid_assets



def asset_matches_type(asset_name, href_asset, soil_type):
    text = f"{asset_name} {href_asset}".lower()
    soil_types = parse_soil_types(soil_type)
    return any(
        text_contains_any_keyword(text, TYPE_KEYWORDS.get(one_type, {one_type}))
        for one_type in soil_types
    )


def parse_zenodo_record_id(zenodo_doi):
    raw = str(zenodo_doi or "").strip()
    if not raw:
        return None

    normalized = raw.lower()
    if "zenodo.org/records/" in normalized:
        return raw.rstrip("/").split("/")[-1]
    if "doi.org/" in normalized and "zenodo." in normalized:
        return raw.split("zenodo.")[-1].split("/")[0].strip()
    if "10.5281/zenodo." in normalized:
        return raw.split("zenodo.")[-1].split("/")[0].strip()
    if normalized.startswith("doi:") and "zenodo." in normalized:
        return raw.split("zenodo.")[-1].split("/")[0].strip()

    # Accept direct numeric record IDs.
    if raw.isdigit():
        return raw

    # Last fallback: keep only trailing digits if present.
    trailing_digits = re.search(r"(\d+)$", raw)
    if trailing_digits:
        return trailing_digits.group(1)

    return None

def fetch_zenodo_geotiffs(zenodo_doi, params,verbose=False):
    """
    Fetch GeoTIFF files from a Zenodo record via its DOI.
    Returns a list of dicts with keys: collection, item, asset, href.
    """
    zenodo_assets = []
    try:
        record_id = parse_zenodo_record_id(zenodo_doi)
        if not record_id:
            if verbose:
                print(f"[Zenodo] Invalid DOI format: {zenodo_doi}")
            return zenodo_assets
        
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        files = data.get("files", [])
        record_title = data.get("title", "Zenodo Dataset")
        
        if verbose:
            print(f"[Zenodo] Found {len(files)} file(s) in record {record_id}")
        
        for file_info in files:
            file_key = file_info.get("key", "")
            if file_key.lower().endswith((".tif", ".tiff")):
                links = file_info.get("links", {})
                href = links.get("download") or links.get("self") or ""
                if href:
                    if verbose:
                        print(params["type"], "|", file_key, "|", href)
                    if not asset_matches_type(file_key, href, params["type"]):
                        continue
                    
                    zenodo_assets.append({
                        "collection": f"Zenodo: {record_title}",
                        "item": record_id,
                        "asset": file_key,
                        "mode": "zenodo",
                        "resolution": "unknown",
                        "href": href,
                    })
        
        if verbose:
            print(f"[Zenodo] Selected {len(zenodo_assets)} GeoTIFF(s)")
    
    except Exception as e:
        if verbose:
            print(f"[Zenodo] Error fetching DOI {zenodo_doi}: {e}")
    
    return zenodo_assets


def infer_short_type(text):
    normalized_text = str(text).lower()
    for canonical_type, keywords in TYPE_KEYWORDS.items():
        if text_contains_any_keyword(normalized_text, keywords):
            return canonical_type
    return "asset"


def infer_resolution_token(text):
    normalized_text = str(text).lower()
    for token in ("30m", "120m", "250m", "1km", "500m"):
        if token in normalized_text:
            return token
    return "res"


def infer_depth_token(text):
    normalized_text = str(text).lower()
    range_match = re.search(r"b(\d+cm(?:\.\.?\d+cm)?)", normalized_text)
    if range_match:
        return range_match.group(1).replace("..", "_")

    single_match = re.search(r"b(\d+cm)", normalized_text)
    if single_match:
        return single_match.group(1)

    plain_match = re.search(r"(\d+cm(?:\.\.?\d+cm)?)", normalized_text)
    if plain_match:
        return plain_match.group(1).replace("..", "_")

    return "depth"

def build_compact_output_stem(asset):
    source_text = " ".join(
        [
            asset.get("collection", ""),
            asset.get("item", ""),
            asset.get("asset", ""),
            asset.get("href", ""),
        ]
    )

    short_type = infer_short_type(source_text)
    mode = extract_mode(source_text)
    resolution = infer_resolution_token(source_text)
    depth = infer_depth_token(source_text)
    return f"{short_type}_{mode}_{resolution}_{depth}"


def extract_point_values(lon, lat, valid_assets, verbose=False):
    """
    Extrait la valeur du pixel le plus proche pour une coordonnée (lon, lat)
    pour chaque asset valide en utilisant /vsicurl/.
    """
    results = {}
    
    for idx, asset in enumerate(valid_assets):
        href = asset["href"]
        output_stem = build_compact_output_stem(asset)
        
        url_candidates = [href, f"/vsicurl/{href}"]
        pixel_value = None
        
        for remote_url in url_candidates:
            try:
                with rasterio.open(remote_url) as src:
                    gen = src.sample([(lon, lat)])
                    pixel_value = next(gen)[0]
                    nodata = src.nodata if src.nodata is not None else src.profile.get('nodata')
                    if nodata is not None and pixel_value == nodata:
                        pixel_value = None
                    break
            except Exception:
                continue
                
        if pixel_value is not None:
            results[output_stem] = pixel_value
            if verbose:
                print(f" ✓ {output_stem} : {pixel_value}")
        else:
            results[output_stem] = None
            if verbose:
                print(f" ❌ {output_stem} : Échec de la lecture")
                
    return results


def compute_and_save_terrain_attributes(dem_path, verbose=False):
    """
    Calculate terrain attributes (slope, aspect, TWI) from a DEM file.
    Returns a list of dicts with paths to computed attributes.
    """
    output_records = []
    try:
        if verbose:
            print(f"    Calcul des attributs de terrain pour {os.path.basename(dem_path)}...")


        # Use absolute paths to avoid working-directory ambiguity
        abs_dem = os.path.abspath(dem_path)
        base_path = os.path.splitext(abs_dem)[0]
        slope_path = f"{base_path}_slope.tif"
        aspect_path = f"{base_path}_aspect.tif"
        twi_path = f"{base_path}_twi.tif"

        wbt = WhiteboxTools()
        wbt.verbose = False

        # Simple helper to run a WhiteboxTools function with retries
        def run_wbt_tool(func, *args, retries=3, wait=0.5):
            last_exc = None
            for attempt in range(retries):
                try:
                    func(*args)
                    time.sleep(wait)
                    return True
                except Exception as ex:
                    last_exc = ex
                    time.sleep(wait)
            raise last_exc

        # Run slope and aspect and ensure outputs exist
        wbt.slope(dem=abs_dem, output=slope_path, units='degrees')
        if not os.path.exists(slope_path):
            raise FileNotFoundError(slope_path)

        wbt.aspect(dem=abs_dem, output=aspect_path)
        if not os.path.exists(aspect_path):
            raise FileNotFoundError(aspect_path)

        # Compute SCA and TWI in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            sca_path = os.path.join(tmp_dir, "sca.tif")
            wbt.d8_flow_accumulation(i=abs_dem, output=sca_path, out_type='specific contributing area')
            if not os.path.exists(sca_path):
                raise FileNotFoundError(sca_path)

            wbt.wetness_index(sca=sca_path, slope=slope_path, output=twi_path)
            if not os.path.exists(twi_path):
                raise FileNotFoundError(twi_path)

        for attr_name, output_path in [('slope', slope_path), ('aspect', aspect_path), ('twi', twi_path)]:
            output_records.append({
                "local_path": output_path,
                "collection": os.path.basename(dem_path),
                "item": "derived",
                "asset": f"terrain_{attr_name}",
                "output_stem": f"{os.path.basename(base_path)}_{attr_name}",
            })

            if verbose:
                mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"      ✓ {attr_name.upper()}: {output_path} ({mb:.2f} MB)")
        
        return output_records
    
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Erreur lors du calcul des attributs de terrain: {e}")
        return []

def extract_dem_terrain_attributes_in_memory(lon, lat, dem_asset, verbose=False):
    """
    Télécharge un extrait de DEM autour du point dans un dossier temporaire, 
    calcule slope, aspect, et twi, extrait la valeur au point donné et supprime les fichiers temporaires.
    """
    buffer_deg = 0.02
    bbox_wgs84 = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    results = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        dem_path = os.path.join(tmp_dir, "dem.tif")
        href = dem_asset["href"]
        url_candidates = [href, f"/vsicurl/{href}"]
        clipped_ok = False
        
        for remote_url in url_candidates:
            try:
                with rasterio.open(remote_url) as src:
                    if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
                        bbox_src = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                    else:
                        bbox_src = tuple(bbox_wgs84)

                    left = max(src.bounds.left, bbox_src[0])
                    bottom = max(src.bounds.bottom, bbox_src[1])
                    right = min(src.bounds.right, bbox_src[2])
                    top = min(src.bounds.top, bbox_src[3])

                    if left >= right or bottom >= top:
                        raise ValueError("bbox hors emprise")

                    crop_window = from_bounds(left, bottom, right, top, src.transform)
                    crop_window = crop_window.round_offsets().round_lengths()
                    data = src.read(window=crop_window)
                    profile = src.profile.copy()
                    profile.update(
                        height=data.shape[1],
                        width=data.shape[2],
                        transform=window_transform(crop_window, src.transform),
                    )

                with rasterio.open(dem_path, "w", **profile) as dst:
                    dst.write(data)
                clipped_ok = True
                break
            except Exception:
                pass
                
        if not clipped_ok:
            if verbose:
                print(" ❌ Échec téléchargement clip DEM pour calcul topographique.")
            return {}
            
        output_records = compute_and_save_terrain_attributes(dem_path, verbose=False)
        for rec in output_records:
            attr_path = rec["local_path"]
            attr_name = rec["output_stem"]
            try:
                with rasterio.open(attr_path) as src:
                    gen = src.sample([(lon, lat)])
                    val = next(gen)[0]
                    results[attr_name] = val
                    if verbose:
                        print(f" ✓ {attr_name} : {val}")
            except Exception:
                results[attr_name] = None
    return results


def is_dem_asset(asset):
    source_text = " ".join(
        [
            asset.get("collection", ""),
            asset.get("item", ""),
            asset.get("asset", ""),
            asset.get("href", ""),
        ]
    )
    return infer_short_type(source_text) == "dem"


def get_site_soil_properties_as_dataframe(
    site_id,
    longitude,
    latitude,
    soil_types=None,
    stat="m",
    resolution="30m",
    base=BASE_DEFAULT,
    verbose=False
):
    delta = 0.001
    bbox = [longitude - delta, latitude - delta, longitude + delta, latitude + delta]

    params = configure_inputs(
        soil_type=soil_types,
        stat=stat,
        resolution=resolution,
        box_zone=bbox,
        base=base,
        strict=False,
    )

    collections = discover_collections(params["base"], params["type"])
    valid_assets = get_valid_assets(params, collections, verbose=False)
    
    for zenodo in zenodo_list:
        valid_assets += fetch_zenodo_geotiffs(zenodo, params, verbose=False)

    if verbose:
        print(f"\nExtraction in-memory pour le site: {site_id} ({latitude}, {longitude})")
        print(f"Nombre d'assets trouvés : {len(valid_assets)}")

    point_results = extract_point_values(longitude, latitude, valid_assets, verbose=verbose)
    
    dem_assets = [a for a in valid_assets if is_dem_asset(a)]
    if dem_assets:
        topo_results = extract_dem_terrain_attributes_in_memory(longitude, latitude, dem_assets[0], verbose=verbose)
        point_results.update(topo_results)
    
    point_results['site_id'] = site_id
    point_results['longitude'] = longitude
    point_results['latitude'] = latitude
    
    df = pd.DataFrame([point_results])
    # Réorganiser pour avoir les identifiants en premier
    cols = ['site_id', 'longitude', 'latitude'] + [c for c in df.columns if c not in ['site_id', 'longitude', 'latitude']]
    return df[cols]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

import glob

def get_topo_data(BASE_DEST_DIR, master_path, coordonnees):
    # Recherche des fichiers CSV générés précédemment

    """Parcourt depth_X/station_Y/ et retourne tous les *soil_moisture*.csv"""
    file_paths = []
    for depth_dir in os.listdir(BASE_DEST_DIR):
        depth_path = os.path.join(BASE_DEST_DIR, depth_dir)
        if os.path.isdir(depth_path):
            for station_dir in os.listdir(depth_path):
                station_path = os.path.join(depth_path, station_dir)
                if os.path.isdir(station_path):
                    for fname in os.listdir(station_path):
                        if fname.endswith('.csv') and 'soil_moisture' in fname:
                            file_paths.append(os.path.join(station_path, fname))

    soil_properties_cache = {}
    all_results = []
    compteur_fichiers = 0

    if file_paths:
        print(f"{len(file_paths)} fichiers CSV trouvés. Début de l'extraction des propriétés du sol...\n")
        
        for csv_file in file_paths:
            try:
                df_test = pd.read_csv(csv_file, nrows=1) # On ne lit que la première ligne pour être plus rapide
                
                # Extraction des coordonnées
                if coordonnees == "Grandvillers":
                    lat=49.4727
                    lon=2.6203
                else:
                    lat = df_test['Latitude'].iloc[0]
                    lon = df_test['Longitude'].iloc[0]
                
                site_id = os.path.basename(csv_file).split('_')[0]
                
                # Arrondir pour gérer les imprécisions des flottants
                coords_key = (round(lon, 4), round(lat, 4))
                
                if coords_key in soil_properties_cache:
                    print(f"[{site_id}] Coordonnées {coords_key} connues -> Récupération depuis le cache.")
                    df_props = soil_properties_cache[coords_key].copy()
                    df_props['site_id'] = site_id
                else:
                    print(f"[{site_id}] Nouvelles coordonnées {coords_key} -> Téléchargement & calcul...")
                    df_props = get_site_soil_properties_as_dataframe(
                        site_id=site_id,
                        longitude=lon,
                        latitude=lat,
                        soil_types=["bulk","silt","clay","sand","Ksat","dem"],
                        verbose=False # Désactivé pour ne pas polluer l'écran lors du parcours de la boucle
                    )
                    soil_properties_cache[coords_key] = df_props
                
                all_results.append(df_props)
                
                print(f"[{compteur_fichiers+1}/{len(file_paths)}] {site_id}  ")
                compteur_fichiers += 1

            except Exception as e:
                print(f"⚠️ Erreur lors du traitement du fichier {csv_file} : {e}")

        # Assemblage de tous les résultats dans un DataFrame final unique et export
        if all_results:
            df_all_sites_soil = pd.concat(all_results, ignore_index=True)
            # Supprimer les potentiels doublons parfaits (si le même site_id apparaît plusieurs fois avec les mêmes params)
            df_all_sites_soil = df_all_sites_soil.drop_duplicates()
            
            # Sauvegarde d'un fichier maître des propriétés du sol
            master_output = os.path.join(master_path)
            df_all_sites_soil.to_csv(master_output, index=False)
            print(f"\nExtraction terminée ! Fichier maître sauvegardé ici : {master_output}")
    else:
        print("Aucun fichier CSV n'a été trouvé. Exécutez d'abord la cellule précédente.")

def update_local_csv_with_master(csv_file, df_master, cols, coordonnees = None, verbose=True):

    df_test = pd.read_csv(csv_file)

    if coordonnees == "Grandvillers":
        lat_local=49.4727
        lon_local=2.6203
    else : 
            # 1. Extraction des coordonnées dans le fichier local courant
        lat_local = df_test['Latitude'].iloc[0]
        lon_local = df_test['Longitude'].iloc[0]
    
    # Recherche de la ligne associée dans df_master
    # (On utilise une tolérance car des arrondis de flottants ont parfois lieu lors des sauvegardes to_csv)
    tol = 1e-4
    match = df_master[
        (np.abs(df_master['latitude'] - lat_local) < tol) & 
        (np.abs(df_master['longitude'] - lon_local) < tol)
    ]
    
    if match.empty:
        print(f" Aucune correspondance trouvée dans le Master CSV pour ({lat_local}, {lon_local})")
        return None

    row_master = match.iloc[0]
    if verbose:
        print(f"=== Fichier local : {os.path.basename(csv_file)} ===")
    

    ########################################################################
    # ---  Comparaison des données redondantes ---
    
    # Clay fraction (Local: ISMN) vs (Master: OpenLandMap 0-30cm par exemple)
    if not coordonnees == "Grandvillers":
        clay_local = df_test['Clay_fraction'].iloc[0]
        clay_cols = [c for c in df_master.columns if 'clay' in c.lower()]
        clay_master = row_master[clay_cols[0]] if clay_cols else np.nan
        
        if verbose:
            print(f"[Clay_fraction]  Local (ISMN) = {clay_local} | Master (OLM) = {clay_master}")
        
        # Elevation (Local: ISMN) vs DEM (Master)
        elev_local = df_test['Elevation'].iloc[0]
        dem_master =  row_master.get('dem_m_30m_depth', np.nan)
        
        if verbose and pd.notna(elev_local) and pd.notna(dem_master):
            diff_elev = elev_local - dem_master
            print(f"[Elevation/DEM]  Local (ISMN) = {elev_local:.2f}m | Master = {dem_master:.2f}m -> Différence: {diff_elev:.2f} m")
        elif verbose:
            print(f"[Elevation/DEM]  Local (ISMN) = {elev_local}m | Master = {dem_master}m")

    #########################################################################
    
    # --- Étape 2 : Ajout des colonnes manquantes ---
    # rename des colonnes de df_test pour correspondre à celles du master (ex: 'clay_fraction' -> 'clay_m_30m_0cm_30cm', 'elevation' -> 'dem_m_30m_depth'
    df_test = df_test.rename(columns={'Clay_fraction': 'clay_m_30m_0cm_30cm', 
                                      'Silt_fraction': 'silt_m_30m_0cm_30cm', 
                                      'Sand_fraction': 'sand_m_30m_0cm_30cm', 
                                      'Elevation': 'dem_m_30m_depth',
                                      })


    for c in cols:
        if c not in df_test.columns:
            df_test[c] = row_master[c]
        elif pd.isna(df_test[c].iloc[0]) and pd.notna(row_master[c]):
            df_test[c] = row_master[c]
        else :
            df_test[c] = df_test[c].fillna(row_master[c])
    
    return df_test

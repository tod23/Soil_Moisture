import numpy as np
import pandas as pd
from config import DATE_COL

def engineer_features(df, feat_cfg):
    df = df.copy()

    FEATURE_COLS = feat_cfg["dense"] + feat_cfg["soil"] + feat_cfg["sparse"]

    # ── doy_sin / doy_cos ──
    if 'doy_sin' in FEATURE_COLS or 'doy_cos' in FEATURE_COLS:
        day_of_year = df[DATE_COL].dt.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # ── VPD ──
    if 'VPD' in FEATURE_COLS and all(c in df.columns for c in ['TMAX', 'TMIN', 'VAP']):
        t_mean = (df['TMAX'] + df['TMIN']) / 2
        es = 0.6108 * np.exp(17.27 * t_mean / (t_mean + 237.3))
        df['VPD'] = es - df['VAP']

    # ── T_RANGE ──
    if 'T_RANGE' in FEATURE_COLS and all(c in df.columns for c in ['TMAX', 'TMIN']):
        df['T_RANGE'] = df['TMAX'] - df['TMIN']

    # ── Rain accumulation ──
    for window, col in [(3, 'RAIN_CUM_3D'), (7, 'RAIN_CUM_7D'), (14, 'RAIN_CUM_14D')]:
        if col in FEATURE_COLS and 'RAIN' in df.columns:
            df[col] = df['RAIN'].rolling(window=window, min_periods=1).sum()

    # ── S2 indices ──
    if 'S2_NDWI' in FEATURE_COLS and all(c in df.columns for c in ['S2_B3', 'S2_B8']):
        num = df['S2_B3'] - df['S2_B8']
        denom = df['S2_B3'] + df['S2_B8']
        df['S2_NDWI'] = np.where(denom != 0, num / denom, np.nan)

    if 'S2_SAVI' in FEATURE_COLS and all(c in df.columns for c in ['S2_B4', 'S2_B8']):
        L = 0.5
        num = df['S2_B8'] - df['S2_B4']
        denom = df['S2_B8'] + df['S2_B4'] + L
        df['S2_SAVI'] = np.where(denom != 0, (num / denom) * (1 + L), np.nan)

    if 'S2_MNDWI' in FEATURE_COLS and all(c in df.columns for c in ['S2_B3', 'S2_B11']):
        num = df['S2_B3'] - df['S2_B11']
        denom = df['S2_B3'] + df['S2_B11']
        df['S2_MNDWI'] = np.where(denom != 0, num / denom, np.nan)

    if 'S2_NBR' in FEATURE_COLS and all(c in df.columns for c in ['S2_B8', 'S2_B12']):
        num = df['S2_B8'] - df['S2_B12']
        denom = df['S2_B8'] + df['S2_B12']
        df['S2_NBR'] = np.where(denom != 0, num / denom, np.nan)

    # ── S1 ratios ──
    if all(c in df.columns for c in ['S1_VV', 'S1_VH']):
        if 'S1_VV_over_VH' in FEATURE_COLS:
            df['S1_VV_over_VH'] = np.where(df['S1_VH'] != 0, df['S1_VV'] / df['S1_VH'], np.nan)
        if 'S1_VH_over_VV' in FEATURE_COLS:
            df['S1_VH_over_VV'] = np.where(df['S1_VV'] != 0, df['S1_VH'] / df['S1_VV'], np.nan)



    # ── ET₀ FAO Penman-Monteith ──
    if 'ET0' in FEATURE_COLS and all(c in df.columns for c in ['TMAX', 'TMIN', 'WIND', 'IRRAD']):
        # Altitude
        elev = df['dem'].iloc[0] if 'dem' in df.columns else None
        if elev is None:
            elev = 0
        
        # Constantes
        t_mean = (df['TMAX'] + df['TMIN']) / 2
        
        # Pression atmosphérique (kPa)
        P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
        
        # Constante psychrométrique (kPa/°C)
        gamma = 0.665e-3 * P
        
        # Pente de la courbe de saturation (kPa/°C)
        es_tmean = 0.6108 * np.exp(17.27 * t_mean / (t_mean + 237.3))
        Delta = 4098 * es_tmean / (t_mean + 237.3) ** 2
        
        # VPD (kPa) — déjà calculé ou à recalculer
        if 'VPD' in df.columns:
            vpd = df['VPD']
        else:
            es_max = 0.6108 * np.exp(17.27 * df['TMAX'] / (df['TMAX'] + 237.3))
            es_min = 0.6108 * np.exp(17.27 * df['TMIN'] / (df['TMIN'] + 237.3))
            es = (es_max + es_min) / 2
            vap = df.get('VAP', es * 0.5)
            vpd = es - vap
        
        # Rayonnement net Rn ≈ 0.77 * Rs (où Rs = IRRAD en MJ/m²/j)
        # IRRAD est déjà en kJ/m², converti plus tôt
        Rn = 0.77 * df['IRRAD'] / 1000  # Approximation simple (MJ/m²/j)
        
        # Termes PM
        wind_ms = df['WIND'] / 3.6
        numer = 0.408 * Delta * Rn + gamma * (900 / (t_mean + 273)) * wind_ms * vpd
        denom = Delta + gamma * (1 + 0.34 * wind_ms)
        df['ET0'] = np.where(denom != 0, numer / denom, np.nan)

    return df


def update_soil_property(df, df_master, cols, depth):
    
    df_master = df_master.copy()
    
    lat_local = df['Latitude'].iloc[0]
    lon_local = df['Longitude'].iloc[0]
    
    tol = 1e-4
    match = df_master[
        (np.abs(df_master['latitude'] - lat_local) < tol) & 
        (np.abs(df_master['longitude'] - lon_local) < tol)
    ]
    
    if match.empty:
        print(f" Aucune correspondance trouvée dans le Master CSV pour ({lat_local}, {lon_local})")
        return None

    df = df.rename(columns={'Clay_fraction': 'clay', 'Silt_fraction': 'silt',
                            'Sand_fraction': 'sand', 'Elevation': 'dem'})
    
    idx = match.index[0]                    # (1) index AVANT rename

    if depth <= 0.3:
        df_master = df_master.rename(columns={
            'clay_m_30m_0cm_30cm': 'clay', 'silt_m_30m_0cm_30cm': 'silt',
            'sand_m_30m_0cm_30cm': 'sand', 'bulk_m_30m_0cm_30cm': 'bulk',
            'dem_m_30m_depth': 'dem', 'ksat_m_1km_0cm': 'ksat_m_1km',
        })
    elif depth <= 0.6:
        df_master = df_master.rename(columns={
            'clay_m_30m_30cm_60cm': 'clay', 'silt_m_30m_30cm_60cm': 'silt',
            'sand_m_30m_30cm_60cm': 'sand', 'bulk_m_30m_30cm_60cm': 'bulk',
            'dem_m_30m_depth': 'dem', 'ksat_m_1km_30cm': 'ksat_m_1km',
        })
    elif depth <= 1.0:
        df_master = df_master.rename(columns={
            'clay_m_30m_60cm_100cm': 'clay', 'silt_m_30m_60cm_100cm': 'silt',
            'sand_m_30m_60cm_100cm': 'sand', 'bulk_m_30m_60cm_100cm': 'bulk',
            'dem_m_30m_depth': 'dem', 'ksat_m_1km_60cm': 'ksat_m_1km',
        })
    else:
        return df
    
    row_master = df_master.loc[idx]          # (2) depuis le df renommé

    for c in cols:
        if c not in df.columns:
            if c in row_master.index and pd.notna(row_master[c]):
                df[c] = row_master[c]
            else:
                df[c] = np.nan
        elif c in row_master.index:
            df[c] = df[c].fillna(row_master[c])

    return df

def update_soil_name(df, depth):
   
    if depth <= 0.3:
        df = df.rename(columns={
            'clay_m_30m_0cm_30cm': 'clay', 'silt_m_30m_0cm_30cm': 'silt',
            'sand_m_30m_0cm_30cm': 'sand', 'bulk_m_30m_0cm_30cm': 'bulk',
            'dem_m_30m_depth': 'dem', 'ksat_m_1km_0cm': 'ksat_m_1km',
        })
    elif depth <= 0.6:
        df = df.rename(columns={
            'clay_m_30m_30cm_60cm': 'clay', 'silt_m_30m_30cm_60cm': 'silt',
            'sand_m_30m_30cm_60cm': 'sand', 'bulk_m_30m_30cm_60cm': 'bulk',
            'dem_m_30m_depth': 'dem', 'ksat_m_1km_30cm': 'ksat_m_1km',
        })
    elif depth <= 1.0:
        df = df.rename(columns={
            'clay_m_30m_60cm_100cm': 'clay', 'silt_m_30m_60cm_100cm': 'silt',
            'sand_m_30m_60cm_100cm': 'sand', 'bulk_m_30m_60cm_100cm': 'bulk',
            'dem_m_30m_depth': 'dem', 'ksat_m_1km_60cm': 'ksat_m_1km',
        })
    else:
        return df
    return df


def resample_timeseries(df_temp, freq='D', method='mean', start_date=None, end_date=None, specific_hour=None):
    if start_date is not None:
        df_temp = df_temp.loc[start_date:]
    if end_date is not None:
        df_temp = df_temp.loc[:end_date]

    numeric_cols = df_temp.select_dtypes(include='number').columns
    df_temp = df_temp[numeric_cols]

    if freq == 'D' and specific_hour is not None:
        df_temp = df_temp[df_temp.index.hour == specific_hour]
        df_resampled = df_temp.resample('D').first()
    else:
        resampler = df_temp.resample(freq)
        if method == 'mean':
            df_resampled = resampler.mean()
        elif method == 'sum':
            df_resampled = resampler.sum()
        else:
            raise ValueError(f"Méthode '{method}' non reconnue.")
    return df_resampled

def interpolate_timeseries(df, col='soil_moisture', n=1, method='linear'):
    """
    Interpole les valeurs manquantes d'une série temporelle pour les valeurs isolées.
    Seules les valeurs manquantes entourées de données valides seront interpolées.
    """

    df_interpolated = df.copy()
    df_interpolated[col] = df_interpolated[col].interpolate(method=method, limit=n)
    return df_interpolated
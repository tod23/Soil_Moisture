import requests
import folium
import rasterio
import numpy as np
import tempfile
import time
import re
import os
from collections import Counter
from folium.plugins import MiniMap
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


def text_contains_any_keyword(text, keywords):
    for keyword in keywords:
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


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


def create_bbox_around_point(lon, lat, buffer_km=5):
    """Create a square bbox around a point using a simple km-to-degree approximation."""
    delta_deg = buffer_km / 111.0
    return [lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg]


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


def intersects_bbox(item_bbox, bbox):
    if not item_bbox:
        return False

    return not (
        item_bbox[2] < bbox[0]
        or item_bbox[0] > bbox[2]
        or item_bbox[3] < bbox[1]
        or item_bbox[1] > bbox[3]
    )


def normalize_collection_href(base_href, href):
    if href.startswith("./"):
        return f"{base_href.rstrip('/')}/{href[2:]}"
    return href


def normalize_child_href(parent_href, href):
    if href.startswith("./"):
        return f"{parent_href.rsplit('/', 1)[0]}/{href[2:]}"
    return href


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

def asset_matches_resolution(href_asset, resolution):
    return resolution in href_asset.lower()


def asset_matches_type(asset_name, href_asset, soil_type):
    text = f"{asset_name} {href_asset}".lower()
    soil_types = parse_soil_types(soil_type)
    return any(
        text_contains_any_keyword(text, TYPE_KEYWORDS.get(one_type, {one_type}))
        for one_type in soil_types
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


def get_assets_before_and_after_filters(params, collections, verbose=True):
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


def get_valid_assets(params, collections, verbose=True):
    _, valid_assets = get_assets_before_and_after_filters(params, collections, verbose=verbose)
    return valid_assets



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


def fetch_zenodo_geotiffs(zenodo_doi, params,verbose=True):
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



def show_selected_assets(valid_assets, max_rows=15):
    print("Apercu des assets valides :")
    if not valid_assets:
        print("Aucun asset retenu avec les filtres actuels.")
        return

    for asset in valid_assets[:max_rows]:
        print(
            f"- {asset['collection']} | {asset['item']} | {asset['asset']} "
            f"| mode={asset['mode']} | res={asset['resolution']}"
        )

    if len(valid_assets) > max_rows:
        print(f"... {len(valid_assets) - max_rows} asset(s) supplementaire(s)")

    by_item = Counter(a["item"] for a in valid_assets)
    by_asset = Counter(a["asset"] for a in valid_assets)

    print("\nRepartition par item:")
    for item_id, count in by_item.items():
        print(f"- {item_id}: {count}")

    print("\nRepartition par asset:")
    for asset_name, count in by_asset.items():
        print(f"- {asset_name}: {count}")


def colorize_normalized_image(img, color_rgb):

    safe_img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    safe_img = np.clip(safe_img, 0.0, 1.0)

    r, g, b = color_rgb
    rgba = np.zeros((safe_img.shape[0], safe_img.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = (safe_img * r * 255).astype(np.uint8)
    rgba[..., 1] = (safe_img * g * 255).astype(np.uint8)
    rgba[..., 2] = (safe_img * b * 255).astype(np.uint8)
    rgba[..., 3] = (safe_img * 255).astype(np.uint8)
    return rgba


def build_map(
    valid_assets,
    bbox,
    max_layers_to_display=20,
    opacity=0.55,
    random_colors=True,
    verbose=True,
):
    bbox = [
        min(bbox[0], bbox[2]),
        min(bbox[1], bbox[3]),
        max(bbox[0], bbox[2]),
        max(bbox[1], bbox[3]),
    ]

    assets_to_plot = valid_assets[:max_layers_to_display]

    if verbose:
        print(f"Preview mode: {len(assets_to_plot)}/{len(valid_assets)} couche(s) seront affichees")

    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        width="80%",
        height="520px",
    )
    map_obj.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

    minimap = MiniMap(
        tile_layer="OpenStreetMap",
        position="bottomleft",
        width=190,
        height=120,
        zoom_level_offset=-5,
        toggle_display=True,
        minimized=False,
        title="Vue monde",
        title_minimized="Vue monde",
    )
    minimap.add_to(map_obj)

    folium.Rectangle(
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        color="red",
        weight=2,
        fill=False,
        tooltip="bbox cible",
    ).add_to(map_obj)

    rng = np.random.default_rng()
    for asset in assets_to_plot:
        url = asset["href"]

        try:
            try:
                src_ctx = rasterio.open(url)
                tmp_path = None
            except Exception:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                tmp.write(response.content)
                tmp.flush()
                tmp.close()
                tmp_path = tmp.name
                src_ctx = rasterio.open(tmp_path)

            with src_ctx as src:
                if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
                    bbox_src = transform_bounds("EPSG:4326", src.crs, *bbox)
                else:
                    bbox_src = tuple(bbox)

                src_left, src_bottom, src_right, src_top = src.bounds
                left = max(src_left, bbox_src[0])
                bottom = max(src_bottom, bbox_src[1])
                right = min(src_right, bbox_src[2])
                top = min(src_top, bbox_src[3])

                if left >= right or bottom >= top:
                    continue

                crop_window = from_bounds(left, bottom, right, top, src.transform)
                crop_window = crop_window.round_offsets().round_lengths()

                img = src.read(1, window=crop_window).astype(np.float32)

                img_min = np.nanmin(img)
                img_max = np.nanmax(img)
                img = (img - img_min) / (img_max - img_min + 1e-9)

                if random_colors:

                    color_rgb = rng.uniform(0.35, 1.0, size=3)
                    img_for_overlay = colorize_normalized_image(img, color_rgb)
                else:
                    img_for_overlay = img

                if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
                    lon_left, lat_bottom, lon_right, lat_top = transform_bounds(src.crs, "EPSG:4326", left, bottom, right, top)


                else:
                    lon_left, lat_bottom, lon_right, lat_top = left, bottom, right, top

                folium.raster_layers.ImageOverlay(
                    image=img_for_overlay,
                    bounds=[[lat_bottom, lon_left], [lat_top, lon_right]],
                    opacity=opacity,
                    name=f"{asset['item']} | {asset['asset']}",
                ).add_to(map_obj)

            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Overlay skipped for {asset['item']} | {asset['asset']}: {e}")

    folium.LayerControl(collapsed=False).add_to(map_obj)
    return map_obj


def compute_and_save_terrain_attributes(dem_path, verbose=True):
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


def prepare_gee_clip_export(valid_assets, output_dir, bbox, verbose=True, force_download=True):
    """Download and clip each asset to the provided bbox."""

    bbox_wgs84 = normalize_bbox(bbox)
    os.makedirs(output_dir, exist_ok=True)

    local_files = []
    dem_to_process = []
    for idx, asset in enumerate(valid_assets):
        href = asset["href"]
        collection = asset.get("collection", "collection")
        item = asset.get("item", "item")
        asset_name = asset.get("asset", f"asset_{idx}")
        output_stem = build_compact_output_stem(asset)
        local_path = os.path.join(output_dir, f"{output_stem}.tif")

        if os.path.exists(local_path) and not force_download:
            if verbose:
                print(f"  ↷ Déjà présent, ignoré: {local_path}")
            continue

        clip_errors = []
        url_candidates = [href, f"/vsicurl/{href}"]
        clipped_ok = False

        for remote_url in url_candidates:
            try:
                if verbose:
                    print(f"  Clip distant via: {remote_url[:110]}...")

                with rasterio.open(remote_url) as src:
                    if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
                        bbox_src = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                    else:
                        bbox_src = tuple(bbox_wgs84)

                    src_left, src_bottom, src_right, src_top = src.bounds
                    left = max(src_left, bbox_src[0])
                    bottom = max(src_bottom, bbox_src[1])
                    right = min(src_right, bbox_src[2])
                    top = min(src_top, bbox_src[3])

                    if left >= right or bottom >= top:
                        raise ValueError("bbox hors emprise du raster")

                    crop_window = from_bounds(left, bottom, right, top, src.transform)
                    crop_window = crop_window.round_offsets().round_lengths()

                    data = src.read(window=crop_window)
                    profile = src.profile.copy()
                    profile.update(
                        height=data.shape[1],
                        width=data.shape[2],
                        transform=window_transform(crop_window, src.transform),
                    )

                with rasterio.open(local_path, "w", **profile) as dst:
                    dst.write(data)

                clipped_ok = True
                if verbose:
                    mb = os.path.getsize(local_path) / (1024 * 1024)
                    print(f"  ✓ Clip bbox sauvegardé: {local_path} ({mb:.2f} MB)")
                break
            except Exception as e:
                clip_errors.append(str(e))

        if not clipped_ok:
            raise RuntimeError("Clip distant impossible. " + f"Erreurs: {' | '.join(clip_errors[:2])}")

        local_files.append(
            {
                "local_path": local_path,
                "collection": collection,
                "item": item,
                "asset": asset_name,
                "output_stem": output_stem,
            }
        )
        if is_dem_asset(asset):
            # Defer DEM processing until after downloads to avoid race conditions
            dem_to_process.append({
                "local_path": local_path,
                "collection": collection,
                "item": item,
                "asset": asset_name,
                "output_stem": output_stem,
            })
            if verbose:
                print(f"  DEM détecté, marquage pour calcul des attributs de terrain...")

        if verbose:
            print(f"  ✓ Sauvegardé: {local_path}")

        
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ {len(local_files)} fichier(s) téléchargé(s)")
        print(f"{'='*60}")
        print(f"Dossier de sortie : {output_dir}")

    # Now process all deferred DEMs in a second pass
    if dem_to_process:
        if verbose:
            print("\nTraitement des DEMs pour calcul des attributs de terrain...")
        for dem_rec in dem_to_process:
            try:
                if verbose:
                    print(f"  Calcul des attributs pour {dem_rec['local_path']}...")
                terrain_records = compute_and_save_terrain_attributes(dem_rec['local_path'], verbose=verbose)
                local_files.extend(terrain_records)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Erreur lors du calcul des attributs pour {dem_rec['local_path']}: {e}")

    return {
        "mode": "local_clip_download_only",
        "local_files": local_files,
        "gee_commands": [],
        "output_dir": output_dir,
    }


def get_field_bbox_from_landcover(
    lon,
    lat,
    field_buffer_km=2,
    search_radius_km=5,
    base_image_collection="ESA/WorldCover/v200",
    cropland_class=40,
    verbose=False,
):
    """Return a field-sized bbox only if the site is inside cropland."""
    try:
        import ee

        try:
            ee.Initialize(project="projet-hrms")
        except Exception:
            pass

        worldcover = ee.ImageCollection(base_image_collection).first().select("Map")
        point = ee.Geometry.Point([lon, lat])
        search_roi = point.buffer(search_radius_km * 1000)

        histogram = worldcover.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=search_roi,
            scale=10,
            bestEffort=True,
            maxPixels=1e8,
        ).getInfo()

        class_counts = histogram.get("Map", {}) or {}
        normalized_counts = {}
        for key, value in class_counts.items():
            try:
                normalized_counts[int(key)] = int(value)
            except Exception:
                continue

        total_pixels = sum(normalized_counts.values())
        cropland_pixels = normalized_counts.get(cropland_class, 0)
        if total_pixels == 0 or cropland_pixels == 0:
            return None

        confidence = round((cropland_pixels / total_pixels) * 100, 1)
        bbox = create_bbox_around_point(lon, lat, buffer_km=field_buffer_km)

        if verbose:
            print(f"  Cropland detected: {confidence}% of sampled pixels in a {search_radius_km} km search window")
            print(f"  Field bbox buffer: {field_buffer_km} km")

        return {
            "bbox": bbox,
            "field_buffer_km": field_buffer_km,
            "search_radius_km": search_radius_km,
            "landcover_class": cropland_class,
            "landcover_name": "Cropland",
            "confidence": confidence,
        }

    except Exception as e:
        if verbose:
            print(f"⚠️  Landcover check failed: {e}")
        return None


def download_site_datasets(
    site_id,
    longitude,
    latitude,
    output_base_dir,
    soil_types=None,
    stat="m",
    resolution="30m",
    field_buffer_km=2,
    search_radius_km=5,
    base=BASE_DEFAULT,
    require_cropland=True,
    verbose=True,
    **prepare_gee_kwargs,
):
    if soil_types is None:
        soil_types = ["clay", "silt"]

    field_info = get_field_bbox_from_landcover(
        longitude,
        latitude,
        field_buffer_km=field_buffer_km,
        search_radius_km=search_radius_km,
        verbose=verbose,
    )
    if field_info is None:
        if require_cropland:
            raise ValueError("Site not in cropland according to WorldCover")
        bbox = create_bbox_around_point(longitude, latitude, buffer_km=field_buffer_km)
        field_info = {"bbox": bbox, "confidence": 0.0, "landcover_name": "Unknown", "field_buffer_km": field_buffer_km}
    else:
        bbox = field_info["bbox"]

    params = configure_inputs(
        soil_type=soil_types,
        stat=stat,
        resolution=resolution,
        box_zone=bbox,
        base=base,
        strict=False,
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"SITE: {site_id}")
        print(f"  Position: ({latitude:.4f}, {longitude:.4f})")
        print(f"  Landcover: {field_info.get('landcover_name', 'Unknown')}")
        print(f"  Confidence: {field_info.get('confidence', 0.0)}%")
        print(f"  bbox: {[round(x, 4) for x in bbox]}")
        print(f"{'='*70}")

    collections = discover_collections(params["base"], params["type"])
    if verbose:
        print(f"Collections trouvées: {len(collections)}")
    if not collections:
        raise ValueError("Aucune collection trouvée pour les filtres donnés")

    valid_assets = get_valid_assets(params, collections, verbose=verbose)
    for zenodo in zenodo_list :
        valid_assets += fetch_zenodo_geotiffs(zenodo, params, verbose=verbose)
    if verbose:
        print(f"Assets valides: {len(valid_assets)}")
        show_selected_assets(valid_assets, max_rows=5)
    if not valid_assets:
        raise ValueError("Aucun asset ne correspond aux critères")
    site_output_dir = os.path.join(output_base_dir, str(site_id))
    os.makedirs(site_output_dir, exist_ok=True)

    result = prepare_gee_clip_export(
        valid_assets,
        output_dir=site_output_dir,
        bbox=bbox,
        verbose=verbose,
        **prepare_gee_kwargs,
    )

    return {
        "site_id": site_id,
        "status": "success",
        "output_dir": site_output_dir,
        "downloaded_files": result.get("local_files", []),
        "num_files": len(result.get("local_files", [])),
        "landcover_name": field_info.get("landcover_name", "Unknown"),
        "field_buffer_km": field_info.get("field_buffer_km", field_buffer_km),
        "confidence": field_info.get("confidence", 0.0),
        "error": None,
    }


def batch_download_ismn_sites(
    ismn_sites_df,
    output_base_dir="./gee_export/sites",
    soil_types=None,
    stat="m",
    resolution="30m",
    field_buffer_km=2,
    search_radius_km=5,
    base=BASE_DEFAULT,
    require_cropland=True,
    verbose=True,
    metadata_output_path="./gee_export/ISMN_metadata.csv",
    log_output_path="./gee_export/download_log.csv",
):
    import pandas as pd
    import os

    if soil_types is None:
        soil_types = ["clay", "silt"]

    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_output_path), exist_ok=True)

    results = []
    failed_sites = []
    total_sites = len(ismn_sites_df)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# BATCH DOWNLOAD - {total_sites} sites ISMN")
        print(f"# Output dir: {output_base_dir}")
        print(f"# Field buffer: {field_buffer_km}km, Search radius: {search_radius_km}km, Stat: {stat}, Resolution: {resolution}")
        print(f"{'#'*70}\n")

    for idx, row in ismn_sites_df.iterrows():
        site_id = row.get("ID") or f"site_{idx}"
        lon = row.get("Longitude")
        lat = row.get("Latitude")

        if lon is None or lat is None:
            if verbose:
                print(f"⚠️  [{idx+1}/{total_sites}] Site {site_id} sans coordonnées, ignoré")
            failed_sites.append(site_id)
            continue

        if verbose:
            print(f"[{idx+1}/{total_sites}] Traitement {site_id}...")

        result = download_site_datasets(
            site_id=site_id,
            longitude=lon,
            latitude=lat,
            output_base_dir=output_base_dir,
            soil_types=soil_types,
            stat=stat,
            resolution=resolution,
            field_buffer_km=field_buffer_km,
            search_radius_km=search_radius_km,
            base=base,
            require_cropland=require_cropland,
            verbose=verbose,
        )

        results.append(result)
        if result["status"] == "failed":
            failed_sites.append(site_id)

        if verbose:
            status_symbol = "✓" if result["status"] == "success" else "❌"
            print(f"{status_symbol} Résultat: {result['status']}\n")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    if verbose:
        print(f"\n{'='*70}")
        print("RÉSUMÉ FINAL")
        print(f"{'='*70}")
        print(f"Total sites: {total_sites}")
        print(f"✓ Succès: {successful}")
        print(f"❌ Échecs: {failed}")
        print(f"{'='*70}\n")

    if results:
        log_df = pd.DataFrame(
            [
                {
                    "site_id": r["site_id"],
                    "status": r["status"],
                    "num_files": r["num_files"],
                    "output_dir": r["output_dir"],
                    "landcover_name": r.get("landcover_name", ""),
                    "field_buffer_km": r.get("field_buffer_km", field_buffer_km),
                    "confidence": r.get("confidence", 0.0),
                    "error": r.get("error", ""),
                }
                for r in results
            ]
        )
        log_df.to_csv(log_output_path, index=False)
        if verbose:
            print(f"Log sauvegardé: {log_output_path}")

    if metadata_output_path:
        metadata_df = ismn_sites_df.copy()
        metadata_df["bbox"] = metadata_df.apply(
            lambda row: create_bbox_around_point(
                row.get("Longitude"),
                row.get("Latitude"),
                buffer_km=field_buffer_km,
            ),
            axis=1,
        )
        metadata_df["download_status"] = metadata_df["ID"].map({r["site_id"]: r["status"] for r in results})
        metadata_df.to_csv(metadata_output_path, index=False)
        if verbose:
            print(f"Métadonnées sauvegardées: {metadata_output_path}")

    return {
        "total_sites": total_sites,
        "successful": successful,
        "failed": failed,
        "results": results,
        "failed_sites": failed_sites,
    }

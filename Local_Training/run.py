import argparse
import os


MODES = ["ismn_train", "osiris_train", "osiris_fine_tuning", "osiris_eval"]


def _parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lance un training du pipeline Soil_Moisture en surchargeant "
                    "la config par la ligne de commande (aucune édition de config.py).")
    p.add_argument("--mode", choices=MODES,
                   help="quel grand mode lancer (ignoré si --list)")
    p.add_argument("--config", nargs="+", default=["all"],
                   help="nom(s) de config dans FEATURE_CONFIGS, ou 'all'")
    p.add_argument("--output", default=None,
                   help="OUTPUT_NAME -> drive_dir = <home>/Documents/<output>/outputs")
    p.add_argument("--drive-dir", default=None,
                   help="surcharge complète de DRIVE_DIR (écrase --output)")
    p.add_argument("--root-dir", default=None, help="surcharge de ROOT_DIR")
    p.add_argument("--osiris-dir", default=None,
                   help="surcharge de OSIRIS_DIR (dataset unifié)")
    p.add_argument("--exp-name", default=None,
                   help="renomme l'expérience dans results.csv (colonne experience_name)")
    p.add_argument("--depth", type=_parse_float_list, default=None,
                   help="liste de profondeurs, ex: 0.1,0.3,0.5")
    p.add_argument("--model", type=_parse_str_list, default=None,
                   help="liste de modèles, ex: lstm,lightgbm")
    p.add_argument("--lookback", type=_parse_int_list, default=None,
                   help="liste de lookbacks, ex: 1,7,14")
    p.add_argument("--horizon", type=_parse_int_list, default=None,
                   help="liste d'horizons, ex: 7,14")
    p.add_argument("--nb-windows", type=_parse_int_list, default=None,
                   help="liste de fenêtres, ex: 50000,100000")
    p.add_argument("--list", action="store_true",
                   help="affiche les configs disponibles et quitte")
    p.add_argument("--dry-run", action="store_true",
                   help="affiche le plan (chemins + configs résolues) sans lancer")
    return p


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    c = dict(cfg)
    if args.exp_name:
        c["name"] = args.exp_name
    if args.depth:
        c["depths"] = args.depth
    if args.model:
        c["models"] = args.model
    if args.lookback:
        c["lookbacks"] = args.lookback
    if args.horizon:
        c["horizons"] = args.horizon
    if args.nb_windows:
        c["nb_windows"] = args.nb_windows
    return c


def _select_configs(feature_configs, names):
    registry = {}
    for cfg in feature_configs:
        registry.setdefault(cfg["name"], []).append(cfg)
    dups = {n for n, cfgs in registry.items() if len(cfgs) > 1}
    if dups:
        raise SystemExit(f"Noms de config en double dans FEATURE_CONFIGS: {sorted(dups)}")

    if "all" in names:
        return list(feature_configs)
    selected = []
    for name in names:
        if name not in registry:
            raise SystemExit(f"Config inconnue: {name!r} (voir --list)")
        selected.append(registry[name][0])
    return selected


def _summary(cfg: dict) -> str:
    n_dense, n_soil, n_sparse = map(len, (cfg["dense"], cfg["soil"], cfg["sparse"]))
    return (f"{cfg['name']}: dense={n_dense}, soil={n_soil}, sparse={n_sparse} | "
            f"LB={cfg['lookbacks']} H={cfg['horizons']} depth={cfg['depths']} "
            f"NB={cfg['nb_windows']} models={cfg['models']}")


def main() -> None:
    args = _parser().parse_args()

    if args.output:
        os.environ["OUTPUT_NAME"] = args.output
    if args.drive_dir:
        os.environ["DRIVE_DIR"] = args.drive_dir
    if args.root_dir:
        os.environ["ROOT_DIR"] = args.root_dir
    if args.osiris_dir:
        os.environ["OSIRIS_DIR"] = args.osiris_dir

    from config import (ROOT_DIR, FEATURE_CONFIGS, RESULTS_CSV_PATH, drive_dir,
                        base_path, OSIRIS_DIR, MONTHS)
    from Fcn_Training import (full_training, osiris_train, osiris_fine_tuning,
                              full_eval_osiris, get_osiris_data)

    if args.list:
        print("Configs disponibles: ")
        for cfg in FEATURE_CONFIGS:
            print("  - " + _summary(cfg))
        print(f"  ROOT_DIR -> {ROOT_DIR}")
        print(f"  OSIRIS_DIR -> {OSIRIS_DIR}")
        print(f"  drive_dir -> {drive_dir}")
        print(f"  base_path -> {base_path}")
        return

    if args.mode is None:
        raise SystemExit("--mode est requis (ou --list)")

    cfgs = [
        _apply_overrides(cfg, args) for cfg in _select_configs(FEATURE_CONFIGS, args.config)
    ]

    print(f"Mode: {args.mode}")
    print(f"  ROOT_DIR = {ROOT_DIR}")
    print(f"  OSIRIS_DIR = {OSIRIS_DIR}")
    print(f"  drive_dir = {drive_dir}")
    print(f"  base_path = {base_path}")
    print(f"  results.csv = {RESULTS_CSV_PATH}")
    for cfg in cfgs:
        print("  * " + _summary(cfg))

    if args.dry_run:
        print("\n  [dry-run] aucun training lancé.")
        return

    all_dfs = None
    if args.mode != "ismn_train":
        all_dfs = get_osiris_data(OSIRIS_DIR)
        print(f"  {len(all_dfs)} sondes Osiris chargées")

    for cfg in cfgs:
        feature_cols = cfg["dense"] + cfg["soil"] + cfg["sparse"]
        print(f"\n========== FEATURE SET: {cfg['name']} ==========")
        print(f"  Features ({len(feature_cols)}): {feature_cols}")

        if args.mode == "ismn_train":
            full_training(cfg, base_path, drive_dir, MONTHS)
        elif args.mode == "osiris_train":
            osiris_train(cfg, drive_dir, all_dfs)
        elif args.mode == "osiris_fine_tuning":
            osiris_fine_tuning(cfg, drive_dir, all_dfs)
        elif args.mode == "osiris_eval":
            full_eval_osiris(cfg, drive_dir, all_dfs)


if __name__ == "__main__":
    main()
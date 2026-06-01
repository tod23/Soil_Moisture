# Soil Moisture Project

Ce dépôt contient des scripts Python et des notebooks Jupyter dédiés à l'acquisition, au traitement, à la modélisation et à la visualisation interactive de données d'humidité du sol (Soil Moisture) et de météo.

## Structure du projet

* **Acquisition des données :**
  * Dataset : Données ISMN téléchargeables sur leur site.
* **Acquisition des données :**
  * `GetDatasetFromISMN.ipynb` et `Fcn_GetDatasetFromISMN.py` : Premiers Notebooks pour télécharger et traiter les données topographiques à partir des données ISMN (International Soil Moisture Network).
* **Préparation et Cartographie :**
  * `Dataset_map.ipynb` : Visualisation spatiale / cartographie des jeux de données.
  * `csv_for_hrsm.ipynb` et `Fcn_for_csv.py` : Préparation et formatage des CSV pour les modèles haute résolution (HRSM).
    * Découpage des fichiers ismn
    * Ajout de données météos
    * Ajout de données topographiques
    * Création d'un csv pour l'entrainement de modèles
* **Modélisation et Entraînement :**
  * `Soil_Moisture_Training.ipynb` : Notebook dédié à l'entraînement des modèles d'humidité du sol.
* **Analyse des Résultats :**
  * `Results.ipynb` : Interface interactive utilisant `ipywidgets` (avec intégration Google Drive) pour analyser, filtrer et tracer les résultats finaux.

* Adaptation incomplète d'un github de prédiction d'humidité.
    * `HRSM_Grandvillers.ipynb` : Application spécifique des modèles pour la localisation de Grandvillers.
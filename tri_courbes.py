#!/usr/bin/env python3
"""
Parcourt un dossier de courbes temporelles (CSV) et demande à l'utilisateur
de garder ou rejeter chaque courbe via une interface Tkinter.

Usage : python tri_courbes.py <dossier>
"""

import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd


def scan_csv(dossier):
    fichiers = []
    for root, dirs, files in os.walk(dossier):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            chemin = os.path.join(root, f)
            try:
                df = pd.read_csv(chemin, nrows=0)
                if "soil_moisture" in df.columns and "date_time" in df.columns:
                    fichiers.append(chemin)
            except Exception:
                continue
    return sorted(fichiers)


def lire_csv(chemin):
    df = pd.read_csv(chemin, sep=",")
    temps = pd.to_datetime(df["date_time"])
    valeur = df["soil_moisture"].values
    return temps, valeur


def extraire_infos(chemin):
    parts = chemin.replace(os.sep, "/").split("/")
    try:
        depth_folder = [p for p in parts if p.startswith("depth_")][0]
        station_folder = [p for p in parts if p.startswith("station_")][0]
        station_id = station_folder.split("_")[1]
    except IndexError:
        depth_folder = "?"
        station_id = "?"
    return depth_folder, station_id


def creer_figure(temps, valeur):
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots(1, 1)
    ax.plot(temps, valeur, lw=0.8)
    ax.set_xlabel("Temps")
    ax.set_ylabel("soil_moisture")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95)
    return fig


def main():
    if len(sys.argv) > 1:
        dossier = sys.argv[1]
    else:
        dossier = filedialog.askdirectory(title="Dossier contenant les CSV")
        if not dossier:
            print("Aucun dossier sélectionné.")
            sys.exit(1)

    if not os.path.isdir(dossier):
        print(f"Erreur : {dossier} n'est pas un dossier valide.")
        sys.exit(1)

    fichiers = scan_csv(dossier)
    if not fichiers:
        print("Aucun fichier CSV trouvé.")
        sys.exit(1)

    print(f"{len(fichiers)} fichier(s) trouvé(s).")
    resultats = []

    root = tk.Tk()
    root.title("Tri de courbes")
    root.geometry("900x650")

    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=0)
    root.grid_columnconfigure(0, weight=1)

    label_info = tk.Label(root, text="", font=("Arial", 12))
    label_info.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    frame_plot = tk.Frame(root)
    frame_plot.grid(row=1, column=0, sticky="nsew")
    frame_plot.grid_rowconfigure(0, weight=1)
    frame_plot.grid_columnconfigure(0, weight=1)

    frame_btn = tk.Frame(root)
    frame_btn.grid(row=2, column=0, pady=10)

    canvas = None
    toolbar = None
    canvas_container = None

    def afficher(index):
        nonlocal canvas, toolbar, canvas_container

        if canvas_container is not None:
            canvas_container.destroy()
            canvas_container = None
            canvas = None
            toolbar = None
        plt.close("all")

        if index >= len(fichiers):
            label_info.config(text="Toutes les courbes traitées.")
            return

        chemin = fichiers[index]
        depth_folder, station_id = extraire_infos(chemin)
        label_info.config(
            text=f"{index+1}/{len(fichiers)} – {depth_folder}/{station_id} – {os.path.basename(chemin)}"
        )

        try:
            t, v = lire_csv(chemin)
        except Exception as e:
            messagebox.showerror("Erreur", f"{os.path.basename(chemin)} : {e}")
            resultats.append(False)
            root.after(10, lambda: afficher(index + 1))
            return

        fig = creer_figure(t, v)

        canvas_container = tk.Frame(frame_plot)
        canvas_container.grid(row=0, column=0, sticky="nsew")
        canvas_container.grid_rowconfigure(1, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(fig, master=canvas_container)
        toolbar = NavigationToolbar2Tk(canvas, canvas_container)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ew")

        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        frame_btn.index = index

    def repondre(garder):
        resultats.append(garder)
        root.after(10, lambda: afficher(frame_btn.index + 1))

    tk.Button(frame_btn, text="Garder", bg="lightgreen", width=12,
              command=lambda: repondre(True)).pack(side=tk.LEFT, padx=10)
    tk.Button(frame_btn, text="Rejeter", bg="lightcoral", width=12,
              command=lambda: repondre(False)).pack(side=tk.LEFT, padx=10)
    tk.Button(frame_btn, text="Quitter", width=10,
              command=root.destroy).pack(side=tk.LEFT, padx=10)

    root.after(50, lambda: afficher(0))
    root.mainloop()

    chemin_sortie = os.path.join(dossier, "resultats.txt")
    with open(chemin_sortie, "w") as f:
        for garde in resultats:
            f.write(f"{str(garde).lower()}\n")

    print(f"\nRésultats sauvegardés dans {chemin_sortie}")
    for f, g in zip(fichiers, resultats):
        print(f"  {'✅' if g else '❌'} {f}")
    print(f"-> {sum(resultats)}/{len(resultats)} gardé(s)")


if __name__ == "__main__":
    main()
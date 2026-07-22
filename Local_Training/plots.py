import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_loss(history: tf.keras.callbacks.History, out_png: str, title: str) -> None:
    """Plot train/validation loss curves (single figure)."""
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE on scaled y)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# %%
def plot_test_true_vs_pred(dates, y_true, y_pred_local, y_pred_era5=None, out_png: str="", title: str="") -> None:
    """Plot test series: actual vs prediction (handles both Local and optionally ERA5 logic)."""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual", color='black', alpha=0.7)
    plt.plot(dates, y_pred_local, label="Prediction (Local)", color='tab:blue')
    
    if y_pred_era5 is not None:
        plt.plot(dates, y_pred_era5, label="Prediction (ERA5)", color='tab:orange', linestyle='--')
        
    plt.xlabel("Date")
    plt.ylabel("Soil Moisture (m3/m3)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def inverse_y(scaler_y: StandardScaler, y_scaled_2d: np.ndarray) -> np.ndarray:
    """Inverse-transform a 2D array (N, horizon) back to original units."""
    flat = y_scaled_2d.reshape(-1, 1)
    inv = scaler_y.inverse_transform(flat).reshape(y_scaled_2d.shape)
    return inv


def compute_horizon_metrics(y_true, y_pred, horizon):
    assert y_true.shape == y_pred.shape, "y_true et y_pred doivent avoir la même forme"
    assert y_true.shape[1] == horizon, f"Le nombre de colonnes de y_true doit être égal à horizon ({horizon})"

    rmse_h = np.array([
        np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
        for j in range(horizon)
    ])

    smape_h = np.array([
        float(np.mean(
            np.abs(y_pred[:, j] - y_true[:, j]) /
            np.maximum((np.abs(y_true[:, j]) + np.abs(y_pred[:, j])) / 2, 1e-3)
        )) * 100.0
        for j in range(horizon)
    ])

    r2_h = np.array([
        float(1 - np.sum((y_true[:, j] - y_pred[:, j])**2) /
              (np.sum((y_true[:, j] - np.mean(y_true[:, j]))**2) + 1e-3))
        for j in range(horizon)
    ])

    ubrmse_h = np.zeros(horizon)
    kge_h = np.zeros(horizon)
    corr_h = np.zeros(horizon)

    for j in range(horizon):
        mask = ~np.isnan(y_true[:, j]) & ~np.isnan(y_pred[:, j])
        yt, yp = y_true[mask, j], y_pred[mask, j]

        rmse_j = np.sqrt(np.mean((yt - yp)**2))
        bias_j = np.mean(yp) - np.mean(yt)
        ubrmse_h[j] = np.sqrt(rmse_j**2 - bias_j**2)

        corr_h[j] = np.corrcoef(yt, yp)[0, 1]
        alpha = np.std(yp) / np.std(yt) if np.std(yt) > 0 else 1
        beta = np.mean(yp) / np.mean(yt) if np.mean(yt) > 0 else 1
        kge_h[j] = 1 - np.sqrt((corr_h[j] - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return rmse_h, ubrmse_h, smape_h, r2_h, kge_h, corr_h
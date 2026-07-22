import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb

from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanSquaredError as MSE
from tensorflow.keras.regularizers import l2
from sklearn.multioutput import MultiOutputRegressor
from config import SEED
from typing import Dict

def build_xgboost(horizon: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,          # Augmenter
        max_depth=7,
        learning_rate=0.05,        # Diminuer
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,       # Ajouter
        gamma=0.1,                 # Ajouter
        random_state=SEED,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=20,
        multi_strategy="multi_output_tree",  # Pour multi-horizon
    )

def build_random_forest(horizon: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )

def build_lightgbm(horizon: int) -> MultiOutputRegressor:
    base = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return MultiOutputRegressor(base)

# %%
def build_convlstm(lookback: int, width: int, horizon: int) -> tf.keras.Model:
    """
    ConvLSTM backbone (inspired by the reference repo's ConvLSTM forecasting idea).
    """
    inp = layers.Input(shape=(lookback, 1, width, 1))

    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(1, 3),
        padding="same",
        return_sequences=True,
        activation="tanh",
        kernel_regularizer=l2(1e-4),
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(1, 3),
        padding="same",
        return_sequences=False,
        activation="tanh",
        kernel_regularizer=l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    out = layers.Dense(horizon, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=tf.keras.losses.Huber(delta=1.0))
    return model




def build_lstm(lookback: int, width: int, horizon: int) -> tf.keras.Model:
    inp = layers.Input(shape=(lookback, width))

    x = layers.LSTM(
        64,
        return_sequences=True,
        kernel_regularizer=l2(1e-4),
    )(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(
        32,
        return_sequences=False,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(horizon, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=tf.keras.losses.Huber(delta=1.0))
    return model

def build_gru(lookback: int, width: int, horizon: int) -> tf.keras.Model:
    inp = layers.Input(shape=(lookback, width))
    x = layers.GRU(128, return_sequences=True, kernel_regularizer=l2(1e-4))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(64, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(32, return_sequences=False, kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(horizon, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=tf.keras.losses.Huber(delta=1.0))
    return model

def build_tcn(lookback: int, width: int, horizon: int) -> tf.keras.Model:
    inp = layers.Input(shape=(lookback, width))
    x = layers.Conv1D(32, kernel_size=3, padding="causal", dilation_rate=1, activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", dilation_rate=2, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(16, kernel_size=3, padding="causal", dilation_rate=4, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(horizon, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


@register_keras_serializable(package="Custom", name="PositionalEncoding")
class PositionalEncoding(layers.Layer):
    def __init__(self, max_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embed_dim = embed_dim

        # Calcul des encodages sin/cos SANS modifier un tenseur existant
        position = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]  # Shape: (max_length, 1)
        div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / embed_dim))  # Shape: (embed_dim // 2)

        # Calcul de sin et cos en une seule opération
        sin_encoding = tf.sin(position * div_term)  # Shape: (max_length, embed_dim // 2)
        cos_encoding = tf.cos(position * div_term)  # Shape: (max_length, embed_dim // 2)

        # Intercaler sin et cos pour obtenir (max_length, embed_dim)
        pe = tf.stack([sin_encoding, cos_encoding], axis=2)  # Shape: (max_length, embed_dim//2, 2)
        pe = tf.reshape(pe, [max_length, embed_dim])  # Shape: (max_length, embed_dim)

        # Ajouter une dimension batch (1, max_length, embed_dim)
        self.pe = tf.reshape(pe, [1, max_length, embed_dim])

    def call(self, inputs):
        """Ajoute l'encodage positionnel aux inputs."""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]  # Broadcast sur batch_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "embed_dim": self.embed_dim,
        })
        return config
    
# ============================================================
# 2. Transformer Encoder Layer (amélioré)
# ============================================================
def transformer_encoder(
    inputs,
    head_size,
    num_heads,
    ff_dim,
    dropout=0.1,
    **kwargs
):
    """
    Implémentation d'un bloc Transformer Encoder avec:
    - Multi-Head Attention
    - Feed-Forward Network (GELU)
    - Layer Normalization + Residual Connections
    """
    # --- Self-Attention ---
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout,
        **kwargs
    )(inputs, inputs)  # Self-attention: queries = keys = values
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)  # Residual + Norm

    # --- Feed-Forward Network ---
    ffn = layers.Dense(ff_dim, activation="gelu")(out1)  # GELU > ReLU pour les Transformers
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(head_size)(ffn)  # Retour à la dimension d'entrée
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)  # Residual + Norm

# ============================================================
# 3. Modèle Transformer complet (optimisé pour ton cas)
# ============================================================
def build_transformer(
    lookback: int,
    width: int,
    horizon: int,
    head_size: int = 256,
    num_heads: int = 4,
    ff_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Transformer Encoder pour la prédiction de séries temporelles.
    """
    inp = layers.Input(shape=(lookback, width))

    x = layers.Dense(head_size)(inp)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = PositionalEncoding(max_length=lookback, embed_dim=head_size)(x)

    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon, activation="linear", name="output")(x)

    model = models.Model(inp, out)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_models(choose_model: str, lookback: int, horizon: int, feat_cfg: Dict) -> tf.keras.Model:
    
    n_features = len(feat_cfg["dense"]) + len(feat_cfg["soil"]) + len(feat_cfg["sparse"])
    if choose_model == "convlstm":
        model = build_convlstm(lookback, n_features, horizon=horizon)
    elif choose_model == "lstm":
        model = build_lstm(lookback, n_features, horizon=horizon)
    elif choose_model == "xgboost":
        model = build_xgboost(horizon=horizon)
    elif choose_model == "gru":
        model = build_gru(lookback, n_features, horizon=horizon)
    elif choose_model == "tcn":
        model = build_tcn(lookback, n_features, horizon=horizon)
    elif choose_model == "lightgbm":
        model = build_lightgbm(horizon=horizon)
    elif choose_model == "transformer":
        model = build_transformer(
            lookback=lookback,
            width=n_features,
            horizon=horizon,
            head_size=256,
            num_heads=4,
            ff_dim=512,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-4,
        )
    elif choose_model == "random_forest":
        model = build_random_forest(horizon=horizon)
    return model
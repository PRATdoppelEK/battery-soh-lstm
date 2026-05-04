"""
models.py
---------
LSTM, RNN, and baseline model definitions for battery SOH prediction.

All PyTorch models follow the same interface:
    forward(x) → predictions of shape (batch,)

Author: Prateek Gaur
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import logging

logger = logging.getLogger(__name__)


# ── LSTM Model ────────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    Two-layer LSTM for SOH sequence prediction.

    Architecture:
        Input  → LSTM (layer 1) → LSTM (layer 2) → Dropout → Linear → SOH

    Args:
        input_size  : number of input features per timestep
        hidden_size : number of LSTM hidden units per layer
        num_layers  : number of stacked LSTM layers
        dropout     : dropout probability between layers (0 = disabled)
        output_size : prediction dimension (1 for SOH regression)
    """

    def __init__(self,
                 input_size:  int   = 7,
                 hidden_size: int   = 128,
                 num_layers:  int   = 2,
                 dropout:     float = 0.2,
                 output_size: int   = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch,) — predicted SOH values
        """
        # h0, c0 default to zeros
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, hidden)
        last_step   = lstm_out[:, -1, :]        # take last timestep
        out         = self.dropout(last_step)
        out         = self.fc(out).squeeze(-1)  # (batch,)
        return out


# ── Vanilla RNN Model ─────────────────────────────────────────────────────────

class RNNPredictor(nn.Module):
    """
    Vanilla (Elman) RNN for SOH prediction.
    Simpler than LSTM — useful as intermediate baseline above linear models.

    Args:
        input_size  : number of input features
        hidden_size : RNN hidden state size
        num_layers  : stacked RNN layers
        dropout     : dropout probability
    """

    def __init__(self,
                 input_size:  int   = 7,
                 hidden_size: int   = 64,
                 num_layers:  int   = 2,
                 dropout:     float = 0.1,
                 output_size: int   = 1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            nonlinearity= "tanh",
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        last_step  = rnn_out[:, -1, :]
        out        = self.dropout(last_step)
        return self.fc(out).squeeze(-1)


# ── GRU Model (bonus — often outperforms vanilla LSTM) ───────────────────────

class GRUPredictor(nn.Module):
    """
    Gated Recurrent Unit model for SOH prediction.
    Fewer parameters than LSTM, often comparable performance.
    """

    def __init__(self,
                 input_size:  int   = 7,
                 hidden_size: int   = 128,
                 num_layers:  int   = 2,
                 dropout:     float = 0.2,
                 output_size: int   = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step  = gru_out[:, -1, :]
        out        = self.dropout(last_step)
        return self.fc(out).squeeze(-1)


# ── Sklearn Baselines ─────────────────────────────────────────────────────────

class RandomForestBaseline:
    """
    Random Forest regressor baseline.
    Flattens the sequence window into a single feature vector.
    """

    def __init__(self, n_estimators: int = 200,
                 max_depth: int = 10,
                 random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            random_state = random_state,
            n_jobs       = -1,
        )

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """(N, seq_len, features) → (N, seq_len * features)"""
        return X.reshape(X.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._flatten(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X))


class SVMBaseline:
    """
    Support Vector Regressor baseline with RBF kernel.
    Uses only the last timestep of each sequence (most recent cycle features).
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf", epsilon: float = 0.01):
        self.model = SVR(C=C, kernel=kernel, epsilon=epsilon)

    def _last_step(self, X: np.ndarray) -> np.ndarray:
        """Use only the most recent cycle from each sequence window."""
        return X[:, -1, :]   # (N, features)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._last_step(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._last_step(X))


# ── Model factory ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "lstm": LSTMPredictor,
    "rnn":  RNNPredictor,
    "gru":  GRUPredictor,
}

BASELINE_REGISTRY = {
    "random_forest": RandomForestBaseline,
    "svm":           SVMBaseline,
}


def get_model(name: str, **kwargs):
    """
    Instantiate a model by name.

    Args:
        name   : one of 'lstm', 'rnn', 'gru', 'random_forest', 'svm'
        kwargs : model hyperparameters

    Returns:
        PyTorch nn.Module or sklearn-style model
    """
    name = name.lower()
    if name in MODEL_REGISTRY:
        model = MODEL_REGISTRY[name](**kwargs)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {name.upper()} | Trainable parameters: {n_params:,}")
        return model
    elif name in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[name](**kwargs)
    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {list(MODEL_REGISTRY) + list(BASELINE_REGISTRY)}"
        )


if __name__ == "__main__":
    # Quick sanity check
    batch, seq_len, features = 16, 30, 7
    x = torch.randn(batch, seq_len, features)

    for name in ["lstm", "rnn", "gru"]:
        model = get_model(name, input_size=features)
        out   = model(x)
        print(f"{name.upper():6s} output shape: {out.shape}  ✓")

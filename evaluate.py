"""
evaluate.py
-----------
Evaluation metrics and visualisation for battery SOH prediction models.

Usage:
    python src/evaluate.py --model lstm
    python src/evaluate.py --model all   # compare all saved models

Author: Prateek Gaur
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import get_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")
PLOTS_DIR     = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    model_name: str = "") -> dict:
    """
    Compute MAE, RMSE, R² and log results.

    Args:
        y_true      : ground-truth SOH values
        y_pred      : predicted SOH values
        model_name  : label for logging

    Returns:
        dict with keys 'mae', 'rmse', 'r2'
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    logger.info(
        f"{model_name or 'Model':>15} — "
        f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
    )
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses: list,
                          val_losses:   list,
                          model_name:   str):
    """Plot and save training vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE", color="#2563EB", linewidth=1.8)
    ax.plot(epochs, val_losses,   label="Val MSE",   color="#DC2626", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name.upper()} — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / f"{model_name}_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curve saved to {path}")


def plot_predictions(y_true:     np.ndarray,
                     y_pred:     np.ndarray,
                     model_name: str,
                     n_samples:  int = 200):
    """
    Two-panel plot:
        Left  — Predicted vs True SOH over cycles (time series view)
        Right — Scatter: predicted vs actual with perfect-fit line
    """
    n = min(n_samples, len(y_true))
    idx = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Time-series panel
    ax1.plot(idx, y_true[:n], label="True SOH",  color="#059669", linewidth=1.5)
    ax1.plot(idx, y_pred[:n], label="Predicted", color="#2563EB",
             linewidth=1.5, linestyle="--")
    ax1.axhline(0.8, color="#DC2626", linestyle=":", linewidth=1.2,
                label="EOL threshold (0.80)")
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("State of Health (SOH)")
    ax1.set_title(f"{model_name.upper()} — SOH Prediction")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.55, 1.05)

    # Scatter panel
    ax2.scatter(y_true[:n], y_pred[:n], alpha=0.4, s=15, color="#2563EB")
    lims = [min(y_true.min(), y_pred.min()) - 0.02,
            max(y_true.max(), y_pred.max()) + 0.02]
    ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax2.set_xlabel("True SOH")
    ax2.set_ylabel("Predicted SOH")
    ax2.set_title(f"{model_name.upper()} — Predicted vs Actual")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Battery State-of-Health Prediction — {model_name.upper()}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / f"{model_name}_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Prediction plot saved to {path}")


def plot_model_comparison(all_metrics: dict):
    """Bar chart comparing MAE, RMSE, R² across all models."""
    models = list(all_metrics.keys())
    mae_vals  = [all_metrics[m]["mae"]  for m in models]
    rmse_vals = [all_metrics[m]["rmse"] for m in models]
    r2_vals   = [all_metrics[m]["r2"]   for m in models]

    x = np.arange(len(models))
    width = 0.25
    colors = ["#2563EB", "#DC2626", "#059669"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, vals, label, color in zip(
        axes,
        [mae_vals, rmse_vals, r2_vals],
        ["MAE ↓", "RMSE ↓", "R² ↑"],
        colors,
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.82, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], rotation=20, fontsize=9)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison — Battery SOH Prediction", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison plot saved to {path}")


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_pytorch_model(model_name: str):
    """Load a saved PyTorch checkpoint from results/."""
    ckpt_path = RESULTS_DIR / f"{model_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(
        ckpt["model_name"],
        input_size  = ckpt["input_size"],
        hidden_size = ckpt["hidden_size"],
        num_layers  = ckpt["num_layers"],
        dropout     = ckpt["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_pytorch(model, X: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def evaluate_model(model_name: str, splits: dict) -> dict:
    """Load a saved model and evaluate on the test set."""
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    from models import MODEL_REGISTRY, BASELINE_REGISTRY

    if model_name in MODEL_REGISTRY:
        model = load_pytorch_model(model_name)
        preds = predict_pytorch(model, X_test)
    elif model_name in BASELINE_REGISTRY:
        pkl_path = RESULTS_DIR / f"{model_name}.pkl"
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    metrics = compute_metrics(y_test, preds, model_name=model_name)
    plot_predictions(y_test, preds, model_name)
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate battery SOH models")
    parser.add_argument("--model", type=str, default="lstm",
                        help="Model name or 'all'")
    args = parser.parse_args()

    splits = {
        "X_test": np.load(PROCESSED_DIR / "X_test.npy"),
        "y_test": np.load(PROCESSED_DIR / "y_test.npy"),
    }

    from models import MODEL_REGISTRY, BASELINE_REGISTRY
    all_model_names = list(MODEL_REGISTRY) + list(BASELINE_REGISTRY)

    if args.model == "all":
        all_metrics = {}
        for name in all_model_names:
            try:
                m = evaluate_model(name, splits)
                all_metrics[name] = m
            except FileNotFoundError:
                logger.warning(f"No saved model found for {name} — skipping.")
        if all_metrics:
            plot_model_comparison(all_metrics)
            with open(RESULTS_DIR / "all_metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)
    else:
        evaluate_model(args.model, splits)


if __name__ == "__main__":
    main()

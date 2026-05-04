"""
train.py
--------
Training loop for LSTM/RNN/GRU models and sklearn baselines.
Supports early stopping, learning rate scheduling, and checkpoint saving.

Usage:
    python src/train.py --model lstm --epochs 100 --batch_size 32
    python src/train.py --model random_forest
    python src/train.py --model all   # trains every model for comparison

Author: Prateek Gaur
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from models import get_model, MODEL_REGISTRY, BASELINE_REGISTRY
from evaluate import compute_metrics, plot_training_curves

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits():
    """Load preprocessed numpy arrays from data/processed/."""
    splits = {}
    for split in ["train", "val", "test"]:
        splits[f"X_{split}"] = np.load(PROCESSED_DIR / f"X_{split}.npy")
        splits[f"y_{split}"] = np.load(PROCESSED_DIR / f"y_{split}.npy")
    logger.info(
        f"Data loaded — train: {splits['X_train'].shape}, "
        f"val: {splits['X_val'].shape}, test: {splits['X_test'].shape}"
    )
    return splits


def make_dataloaders(splits: dict, batch_size: int = 32):
    """Convert numpy arrays to PyTorch DataLoaders."""
    def to_tensors(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    loaders = {
        "train": DataLoader(
            to_tensors(splits["X_train"], splits["y_train"]),
            batch_size=batch_size, shuffle=True, drop_last=False,
        ),
        "val": DataLoader(
            to_tensors(splits["X_val"], splits["y_val"]),
            batch_size=batch_size, shuffle=False,
        ),
    }
    return loaders


# ── PyTorch training loop ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            loss    = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def train_pytorch_model(model_name: str,
                         splits: dict,
                         epochs:       int   = 100,
                         batch_size:   int   = 32,
                         lr:           float = 1e-3,
                         patience:     int   = 15,
                         hidden_size:  int   = 128,
                         num_layers:   int   = 2,
                         dropout:      float = 0.2):
    """
    Full training loop with early stopping and LR scheduling.

    Args:
        model_name  : 'lstm', 'rnn', or 'gru'
        splits      : dict of X_train, y_train, X_val, y_val, X_test, y_test
        epochs      : maximum training epochs
        batch_size  : mini-batch size
        lr          : initial learning rate
        patience    : early stopping patience (epochs without improvement)
        hidden_size : LSTM/RNN hidden size
        num_layers  : number of stacked recurrent layers
        dropout     : dropout probability
    """
    input_size = splits["X_train"].shape[2]   # number of features
    model = get_model(model_name,
                      input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      dropout=dropout).to(DEVICE)

    loaders   = make_dataloaders(splits, batch_size)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    train_losses   = []
    val_losses     = []

    logger.info(f"\n{'='*55}")
    logger.info(f"Training {model_name.upper()} on {DEVICE}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    logger.info(f"{'='*55}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, DEVICE)
        val_loss   = validate_epoch(model, loaders["val"], criterion, DEVICE)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    # Save checkpoint
    ckpt_path = RESULTS_DIR / f"{model_name}_best.pt"
    torch.save({"model_state": best_state,
                "model_name":  model_name,
                "input_size":  input_size,
                "hidden_size": hidden_size,
                "num_layers":  num_layers,
                "dropout":     dropout}, ckpt_path)
    logger.info(f"Best model saved to {ckpt_path}")

    # Save training curves
    plot_training_curves(train_losses, val_losses, model_name)

    return model, {"train_losses": train_losses, "val_losses": val_losses}


# ── Sklearn baseline training ─────────────────────────────────────────────────

def train_baseline(model_name: str, splits: dict):
    """
    Train a sklearn baseline model (RandomForest or SVM).
    Saves the fitted model to results/.
    """
    logger.info(f"\nTraining baseline: {model_name.upper()}")
    model = get_model(model_name)

    t0 = time.time()
    model.fit(splits["X_train"], splits["y_train"])
    logger.info(f"Fit completed in {time.time()-t0:.1f}s")

    # Save
    pkl_path = RESULTS_DIR / f"{model_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {pkl_path}")

    return model


# ── Run all models ─────────────────────────────────────────────────────────────

def train_all(splits: dict, epochs: int = 100, batch_size: int = 32):
    """Train every model and collect metrics for comparison."""
    all_metrics = {}

    # Deep learning models
    for name in MODEL_REGISTRY:
        model, history = train_pytorch_model(
            name, splits, epochs=epochs, batch_size=batch_size
        )
        X_test = torch.tensor(splits["X_test"], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = model(X_test).cpu().numpy()
        metrics = compute_metrics(splits["y_test"], preds, model_name=name)
        all_metrics[name] = metrics

    # Baseline models
    for name in BASELINE_REGISTRY:
        model = train_baseline(name, splits)
        preds = model.predict(splits["X_test"])
        metrics = compute_metrics(splits["y_test"], preds, model_name=name)
        all_metrics[name] = metrics

    # Save comparison table
    results_path = RESULTS_DIR / "all_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\n{'='*55}")
    logger.info("FINAL RESULTS COMPARISON")
    logger.info(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    logger.info(f"{'-'*45}")
    for name, m in all_metrics.items():
        logger.info(f"{name:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
    logger.info(f"{'='*55}")

    return all_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train battery SOH prediction models")
    parser.add_argument("--model",       type=str, default="lstm",
                        help="Model to train: lstm | rnn | gru | random_forest | svm | all")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--patience",    type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-run preprocessing if processed data doesn't exist
    if not (PROCESSED_DIR / "X_train.npy").exists():
        logger.info("Processed data not found — running preprocessing first ...")
        from data_loader import run_preprocessing
        run_preprocessing()

    splits = load_splits()

    if args.model == "all":
        train_all(splits, epochs=args.epochs, batch_size=args.batch_size)

    elif args.model in MODEL_REGISTRY:
        model, _ = train_pytorch_model(
            args.model, splits,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            hidden_size = args.hidden_size,
            num_layers  = args.num_layers,
            dropout     = args.dropout,
            patience    = args.patience,
        )

    elif args.model in BASELINE_REGISTRY:
        train_baseline(args.model, splits)

    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()

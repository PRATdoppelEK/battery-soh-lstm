"""
data_loader.py
--------------
Loads NASA PCoE battery dataset (.mat files), extracts cycle-level features,
and generates sliding-window sequences for LSTM/RNN training.

Dataset: NASA Prognostics Center of Excellence Battery Dataset
Download: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Author: Prateek Gaur
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
RATED_CAPACITY = 2.0          # Ah — NASA B0005/B0006/B0007/B0018 batteries
EOL_THRESHOLD  = 0.8          # 80% of rated = end of life
SEQUENCE_LENGTH = 30          # Number of past cycles used as input window
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# test = remaining 0.15


# ── Feature extraction from raw cycle data ───────────────────────────────────

def extract_cycle_features(cycle_data: dict) -> dict:
    """
    Extract scalar features from a single charge/discharge cycle.

    Features:
        - discharge_capacity : total Ah discharged this cycle
        - voltage_mean        : mean terminal voltage during discharge
        - voltage_min         : minimum terminal voltage
        - current_mean        : mean discharge current
        - temperature_mean    : mean temperature (°C)
        - temperature_max     : peak temperature
        - discharge_time      : seconds from start to cutoff
        - coulombic_efficiency: ratio of charge put in vs charge taken out
    """
    try:
        voltage     = np.array(cycle_data["Voltage_measured"]).flatten()
        current     = np.array(cycle_data["Current_measured"]).flatten()
        temperature = np.array(cycle_data["Temperature_measured"]).flatten()
        time        = np.array(cycle_data["Time"]).flatten()
        capacity    = np.array(cycle_data["Capacity"]).flatten()

        discharge_capacity = float(capacity[-1]) if len(capacity) > 0 else np.nan

        return {
            "discharge_capacity":   discharge_capacity,
            "voltage_mean":         float(np.mean(voltage)),
            "voltage_min":          float(np.min(voltage)),
            "current_mean":         float(np.mean(np.abs(current))),
            "temperature_mean":     float(np.mean(temperature)),
            "temperature_max":      float(np.max(temperature)),
            "discharge_time":       float(time[-1] - time[0]) if len(time) > 1 else np.nan,
        }
    except Exception as e:
        logger.warning(f"Feature extraction failed for cycle: {e}")
        return {}


def load_nasa_battery(mat_path: Path) -> pd.DataFrame:
    """
    Load a single NASA battery .mat file and return a DataFrame of cycles.

    Each row = one discharge cycle with extracted features + SOH label.
    """
    logger.info(f"Loading {mat_path.name} ...")
    mat = sio.loadmat(str(mat_path), simplify_cells=True)

    # Navigate the nested MATLAB struct
    battery_key = [k for k in mat.keys() if not k.startswith("_")][0]
    cycles = mat[battery_key]["cycle"]

    records = []
    cycle_num = 0

    for cycle in cycles:
        if cycle["type"] != "discharge":
            continue
        cycle_num += 1
        features = extract_cycle_features(cycle["data"])
        if not features:
            continue
        features["cycle_number"] = cycle_num
        records.append(features)

    df = pd.DataFrame(records).dropna()

    # Compute SOH = discharge_capacity / rated_capacity
    df["soh"] = df["discharge_capacity"] / RATED_CAPACITY
    df["soh"] = df["soh"].clip(0.0, 1.0)

    # Mark end-of-life
    df["is_eol"] = df["soh"] < EOL_THRESHOLD

    logger.info(f"  → {len(df)} discharge cycles extracted, "
                f"final SOH = {df['soh'].iloc[-1]:.3f}")
    return df.reset_index(drop=True)


def load_all_batteries(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load all .mat files found in raw_dir and concatenate into one DataFrame.
    Adds a 'battery_id' column to distinguish batteries.
    """
    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        logger.warning(
            f"No .mat files found in {raw_dir}. "
            "Generating synthetic data for demonstration."
        )
        return generate_synthetic_data()

    all_dfs = []
    for f in mat_files:
        df = load_nasa_battery(f)
        df["battery_id"] = f.stem
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total cycles loaded: {len(combined)} from {len(mat_files)} batteries")
    return combined


# ── Synthetic data (fallback when no real data available) ────────────────────

def generate_synthetic_data(n_batteries: int = 4,
                             cycles_per_battery: int = 200,
                             seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic battery degradation data for testing.
    SOH follows an exponential decay with added Gaussian noise.
    """
    logger.info("Generating synthetic battery dataset for demonstration ...")
    rng = np.random.default_rng(seed)
    records = []

    for b in range(n_batteries):
        for c in range(1, cycles_per_battery + 1):
            # Exponential SOH decay with noise
            soh = 1.0 * np.exp(-0.0015 * c) + rng.normal(0, 0.005)
            soh = float(np.clip(soh, 0.6, 1.0))
            cap = soh * RATED_CAPACITY

            records.append({
                "battery_id":          f"B{b+1:04d}",
                "cycle_number":        c,
                "discharge_capacity":  cap,
                "voltage_mean":        3.8 - 0.001 * c + rng.normal(0, 0.02),
                "voltage_min":         2.7 - 0.0008 * c + rng.normal(0, 0.01),
                "current_mean":        1.5 + rng.normal(0, 0.05),
                "temperature_mean":    25.0 + 0.01 * c + rng.normal(0, 0.5),
                "temperature_max":     35.0 + 0.02 * c + rng.normal(0, 0.5),
                "discharge_time":      3600 * soh + rng.normal(0, 60),
                "soh":                 soh,
                "is_eol":              soh < EOL_THRESHOLD,
            })

    df = pd.DataFrame(records)
    logger.info(f"Synthetic data: {len(df)} cycles across {n_batteries} batteries")
    return df


# ── Sequence generation ───────────────────────────────────────────────────────

FEATURE_COLS = [
    "discharge_capacity",
    "voltage_mean",
    "voltage_min",
    "current_mean",
    "temperature_mean",
    "temperature_max",
    "discharge_time",
]

TARGET_COL = "soh"


def make_sequences(df: pd.DataFrame,
                   seq_len: int = SEQUENCE_LENGTH,
                   feature_cols: list = FEATURE_COLS,
                   target_col: str = TARGET_COL):
    """
    Build sliding-window sequences per battery.

    Returns:
        X : np.ndarray of shape (N, seq_len, n_features)
        y : np.ndarray of shape (N,)  — SOH at cycle t+1
    """
    X_list, y_list = [], []

    for battery_id, group in df.groupby("battery_id"):
        group = group.sort_values("cycle_number").reset_index(drop=True)
        features = group[feature_cols].values  # (T, F)
        targets  = group[target_col].values    # (T,)

        for i in range(seq_len, len(group)):
            X_list.append(features[i - seq_len : i])   # window of seq_len cycles
            y_list.append(targets[i])                   # next-step SOH

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(f"Sequences created: X={X.shape}, y={y.shape}")
    return X, y


def train_val_test_split(X: np.ndarray, y: np.ndarray,
                          train_ratio: float = TRAIN_RATIO,
                          val_ratio:   float = VAL_RATIO):
    """Chronological (non-shuffled) split to avoid data leakage."""
    n = len(X)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    splits = {
        "X_train": X[:t1],  "y_train": y[:t1],
        "X_val":   X[t1:t2],"y_val":   y[t1:t2],
        "X_test":  X[t2:],  "y_test":  y[t2:],
    }
    for k, v in splits.items():
        logger.info(f"  {k}: {v.shape}")
    return splits


def scale_features(splits: dict, save_scaler: bool = True) -> dict:
    """
    Fit MinMaxScaler on training features only. Apply to all splits.
    Scaler is saved to data/processed/scaler.pkl for inference use.
    """
    n_train, seq_len, n_feat = splits["X_train"].shape
    scaler = MinMaxScaler()

    # Fit on flattened training windows
    train_flat = splits["X_train"].reshape(-1, n_feat)
    scaler.fit(train_flat)

    scaled = {}
    for split in ["train", "val", "test"]:
        X = splits[f"X_{split}"]
        flat = X.reshape(-1, n_feat)
        scaled[f"X_{split}"] = scaler.transform(flat).reshape(X.shape).astype(np.float32)
        scaled[f"y_{split}"] = splits[f"y_{split}"]

    if save_scaler:
        scaler_path = PROCESSED_DIR / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")

    return scaled, scaler


# ── Main preprocessing pipeline ───────────────────────────────────────────────

def run_preprocessing():
    """End-to-end: load → sequence → split → scale → save."""
    df = load_all_batteries()

    # Save processed cycle DataFrame
    df.to_csv(PROCESSED_DIR / "cycles.csv", index=False)
    logger.info(f"Cycle data saved to {PROCESSED_DIR / 'cycles.csv'}")

    X, y = make_sequences(df)
    splits = train_val_test_split(X, y)
    scaled, scaler = scale_features(splits)

    # Save arrays
    np.save(PROCESSED_DIR / "X_train.npy", scaled["X_train"])
    np.save(PROCESSED_DIR / "y_train.npy", scaled["y_train"])
    np.save(PROCESSED_DIR / "X_val.npy",   scaled["X_val"])
    np.save(PROCESSED_DIR / "y_val.npy",   scaled["y_val"])
    np.save(PROCESSED_DIR / "X_test.npy",  scaled["X_test"])
    np.save(PROCESSED_DIR / "y_test.npy",  scaled["y_test"])

    logger.info("Preprocessing complete. Arrays saved to data/processed/")
    return scaled, scaler


if __name__ == "__main__":
    run_preprocessing()

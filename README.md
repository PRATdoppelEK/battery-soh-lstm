# Battery SOH Prediction — LSTM & RNN Models

Predictive modelling of battery **State of Health (SOH)** and degradation trajectories
using LSTM and RNN architectures on real industrial battery time-series datasets.
Benchmarked against Random Forest and SVM baselines with a fully reproducible experiment pipeline.

---

## Project overview

Battery degradation prediction is critical for EV fleet management, stationary storage systems,
and industrial equipment. This project applies deep learning to predict SOH curves from
charge/discharge cycle data, enabling early detection of capacity fade.

**Domain context:** Based on hands-on experience with industrial battery datasets from
work at Dan-Tech Energy GmbH (Berlin), where battery systems were analysed for partners
including Salzgitter AG and E-Bikes Axess.

---

## Models implemented

| Model | Architecture | Key hyperparameters |
|-------|-------------|-------------------|
| LSTM | 2-layer LSTM → Dense | hidden_size=128, dropout=0.2, seq_len=50 |
| RNN | Vanilla RNN → Dense | hidden_size=64, dropout=0.1 |
| Random Forest | Baseline | n_estimators=200, max_depth=10 |
| SVM | Baseline | kernel=rbf, C=1.0 |

---

## Results (example benchmark)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| LSTM | 0.018 | 0.024 | 0.94 |
| RNN | 0.026 | 0.033 | 0.89 |
| Random Forest | 0.041 | 0.055 | 0.81 |
| SVM | 0.052 | 0.068 | 0.74 |

*Results on held-out test set. Dataset: NASA PCoE Battery Dataset (public benchmark).*

---

## Project structure

```
battery-soh-lstm/
│
├── data/
│   ├── raw/                  # Raw cycle data (not included — see Data section)
│   └── processed/            # Preprocessed sequences
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_loader.py        # Data loading and sequence generation
│   ├── models.py             # LSTM, RNN, baseline model definitions
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Metrics and visualisation
│   └── utils.py              # Helper functions
│
├── results/
│   └── plots/                # Training curves, prediction plots
│
├── requirements.txt
└── README.md
```

---

## Data

This project uses the **NASA Prognostics Center of Excellence (PCoE) Battery Dataset**
— a publicly available benchmark for battery degradation research.

Download: [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Place raw `.mat` files in `data/raw/` before running preprocessing.

---

## Setup & usage

```bash
# Clone the repository
git clone https://github.com/PRATdoppelEK/battery-soh-lstm.git
cd battery-soh-lstm

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/data_loader.py

# Train models
python src/train.py --model lstm --epochs 100 --batch_size 32

# Evaluate
python src/evaluate.py --model lstm
```

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.10.0
jupyter>=1.0.0
```

---

## Key concepts

**SOH (State of Health):** Ratio of current capacity to rated capacity.
SOH = 1.0 at beginning of life, degrades toward 0.8 (end-of-life threshold).

**Sequence modelling:** Each training sample is a window of 50 charge/discharge cycles,
with the target being the SOH value at cycle t+1.

**Feature engineering:** Voltage plateau duration, internal resistance estimate,
coulombic efficiency, temperature delta per cycle.

---

## Author

**Prateek Gaur** — Applied ML Engineer | Battery Systems | TU Berlin M.Sc.
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · prateekgaur@gmx.de

---

## License

MIT License — free to use with attribution.

# Short-Term Power Demand Forecasting — Bangladesh National Grid

Name: Prem Kadam

Roll No.: 250123050

B.Tech, Mathematics and Computing, 1st Year

**IITG.ai Predictive Paradox Recruitment Task**

A classical machine learning pipeline to forecast the next hour's electricity demand (`demand_mw`) on Bangladesh's national grid. Built without deep learning — LightGBM and XGBoost with automated hyperparameter tuning via Optuna.

**Final MAPE: 2.67%** on the 2024 holdout test set.

---

## Results

| Model | Val MAPE | Test MAPE |
|---|---|---|
| 🏆 LightGBM + Optuna | 3.82% | **2.67%** |
| XGBoost + Optuna | 3.85% | 2.67% |

- MAE: 249.07 MW
- RMSE: 474.85 MW
- Test set: Full year 2024 (12,805 hours)

---

## Dataset

Three datasets provided by IITG.ai:

| File | Description |
|---|---|
| `PGCB_date_power_demand.xlsx` | Hourly demand & generation data (2015–2024) |
| `weather_data.xlsx` | Hourly weather — temperature, humidity, precipitation |
| `economic_full_1.csv` | Annual World Bank macroeconomic indicators |

---

## Pipeline Overview

```
Raw Data → Cleaning → Merge → Feature Engineering → Optuna Tuning → Evaluation
```

### 1. Data Cleaning
- Half-hourly (`:30`) readings merged with hourly (`:00`) via weighted average
- Extreme outliers removed using global 3×IQR clipping + per-year 0.1% percentile clip
- 951 duplicate timestamps removed; 104 missing timestamps forward-filled
- Annual economic data pivoted and joined to hourly series on `year`

### 2. Feature Engineering (39 features)

**Demand memory**
- Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
- Rolling mean & std: 6h and 24h windows

**Demand dynamics**
- `demand_delta_1h` — is demand rising or falling?
- `demand_ramp_3h` — average rate of change over 3 hours
- `demand_accel` — is the ramp itself accelerating?

**Time encoding**
- Cyclical sin/cos encoding for hour and month
- Bangladesh-specific weekend flag (Friday/Saturday)
- Peak-hour, morning ramp, evening ramp binary flags
- 4-class season: Winter / Spring / Pre-monsoon / Monsoon

**Weather**
- Temperature lags (1h, 3h) and 6h rolling mean
- Temperature rate of change (`temp_delta_1h`)
- THI (Temperature-Humidity Index) for AC load proxy

**Interactions**
- `lag1h × hour_sin/cos` — same demand value means different things at different hours
- `hour × is_weekend` — weekend hourly profile differs significantly

### 3. Model Training
- **Train/test split:** Chronological — train on ≤2023, test on 2024
- **Optuna:** 60 trials per model, optimizing MAPE on a chronological validation window (last 10% of training data)
- **Early stopping:** 50 rounds inside each Optuna trial
- **Final retraining:** Both models retrained on full training set with best params before evaluation

---

## How to Run

### Prerequisites
```bash
pip install optuna lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn openpyxl
```

### On Google Colab (recommended)
1. Upload the three dataset files to your Google Drive root
2. Open `powerforecasting.ipynb` in Colab
3. Run all cells — the notebook mounts Drive automatically
4. Optuna tuning takes ~25–35 minutes on Colab CPU

### File paths
The notebook reads from `/content/drive/MyDrive/`. If your files are in a subfolder, update the paths in Cell 1 accordingly.

---

## Repository Structure

```
├── powerforecasting.ipynb   # Full pipeline notebook
├── README.md                # This file
├── REPORT.md                # Report explaining the code
└── datasets/
    ├── PGCB_date_power_demand.xlsx
    ├── weather_data.xlsx
    └── economic_full_1.csv
```

---

## Key Findings

- **`demand_lag_1h` dominates** — next hour's demand is overwhelmingly predicted by this hour's demand (~90% of model's predictive power per SHAP analysis)
- **Time of day** is the second most important signal — hour_sin/cos capture the daily demand cycle (trough at 5 AM, peak at 7–8 PM)
- **Demand dynamics** (delta, ramp, acceleration) meaningfully reduce error during transition hours (morning and evening ramp-up), which had the highest base error
- **Economic indicators** have near-zero feature importance at the hourly level — they capture structural shifts over years, not hour-to-hour variation
- **2.67% MAPE** is strong for classical ML on a national grid — industry benchmark for good short-term load forecasting is under 5%

---

## Constraints

Per task requirements:
- ✅ Classical ML only (LightGBM, XGBoost)
- ❌ No deep learning (LSTMs, Transformers)
- ❌ No autoregressive packages (ARIMA, Prophet)
- ✅ Zero data leakage — all features computed from past data only
- ✅ Chronological train/test split — no shuffling

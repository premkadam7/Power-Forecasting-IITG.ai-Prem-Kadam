# Methodology Report — Bangladesh Power Demand Forecasting

Name: Prem Kadam

Roll No.: 250123050

B.Tech, Mathematics and Computing, 1st Year

## 1. Problem Summary

The goal was to forecast the **next hour's electricity demand** (`demand_mw`) on Bangladesh's national grid using only classical ML — no LSTMs, no ARIMA, no Prophet. Three datasets were provided: hourly power demand (PGCB), hourly weather, and annual World Bank macroeconomic indicators. Primary evaluation metric: **MAPE**.

Final result: **LightGBM + Optuna achieved 2.6712% MAPE** on the 2024 holdout test set.

---

## 2. Data Preparation & Anomaly Handling

### 2.1 Half-Hourly Timestamp Resolution
The raw PGCB data contained a mix of on-the-hour (`:00`) and half-hourly (`:30`) readings. Rather than discarding the `:30` entries, we merged them into the hourly timeline using a **weighted average** — giving the on-hour reading twice the weight of the half-hour reading. This preserves information while producing a clean, uniform hourly series.

### 2.2 Outlier Detection & Removal
Demand values were cleaned in two passes:

- **Global IQR clipping (3×IQR):** Catches massive spikes like the 155,000 MW entries visible in the raw data. The 3× multiplier is deliberately conservative — it only removes values that are statistically impossible, not just unusual.
- **Per-year percentile clipping (0.1th–99.9th percentile):** A secondary pass to catch localised logging errors within each year that the global IQR would miss. The threshold is intentionally tight (0.1%) to avoid clipping legitimate low-demand hours.

Missing timestamps (104 gaps) were forward-filled since power demand is highly autocorrelated — the last known value is a better estimate than any interpolation.

### 2.3 Economic Data Integration
The World Bank data is annual, not hourly. We selected 6 indicators most relevant to electricity demand (GDP growth, manufacturing share, urban population growth, FDI, transmission losses, total population), pivoted the data so each year becomes a row, and joined it to the hourly series on the `year` column. Each hourly row inherits the economic indicators for its calendar year — a clean, leak-free integration.

---

## 3. Feature Engineering

Since tree-based models treat each row independently (no sequential memory), we had to manually encode "time" into the feature set. Every feature was computed using only information available at or before time `t`, with rolling/lag features shifted by at least 1 step to prevent data leakage.

### 3.1 Demand Memory (Lags)
The single most important group of features. We engineered lags at:
`1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h`

The 1-hour lag alone accounts for ~90% of the model's predictive power (confirmed by SHAP analysis) — next hour's demand is strongly correlated with this hour's demand. The 24h and 168h lags capture daily and weekly seasonality respectively.

### 3.2 Demand Dynamics
To help the model understand whether demand is rising or falling *going into* the target hour:

- **`demand_delta_1h`** — difference between lag_1h and lag_2h (is demand increasing?)
- **`demand_ramp_3h`** — average rate of change over the last 3 hours
- **`demand_accel`** — whether the ramp itself is accelerating or decelerating

These features were motivated by error analysis showing the highest MAPE at hours 17 (evening ramp-up) and 6–7 (morning ramp-up) — transition periods where demand changes fastest.

### 3.3 Rolling Statistics
- **Rolling means (6h, 24h):** Recent average demand, capturing short and medium-term trends
- **Rolling std (6h, 24h):** Demand volatility — high std signals an unstable period where single-point lags are less reliable

All rolling features are computed on `demand_mw.shift(1)` to ensure no leakage of the target value.

### 3.4 Cyclical Time Encoding
Hour and month are cyclical (hour 23 is adjacent to hour 0, December is adjacent to January). Treating them as raw integers breaks this continuity. We encode both using sine and cosine transformations:

`hour_sin = sin(2π × hour / 24)`,  `hour_cos = cos(2π × hour / 24)`

This allows the model to understand that midnight and 11 PM are close in time.

### 3.5 Thermal Features
- **Temperature lags (1h, 3h) and 6h rolling mean:** Air conditioning demand responds to heat buildup, not instantaneous temperature
- **`temp_delta_1h`:** Temperature rate of change — motivated by high MAPE in March–May (pre-monsoon), when temperatures shift rapidly day-to-day
- **THI (Temperature-Humidity Index):** `T - (0.55 - 0.0055×RH) × (T - 14.5)` — captures "feels-like" heat stress, which correlates more strongly with cooling load than raw temperature alone

### 3.6 Bangladesh-Specific Features
- **`is_weekend`:** Friday and Saturday are the weekend in Bangladesh (not Saturday/Sunday)
- **`season`:** Custom 4-class encoding — Winter (Dec–Feb), Spring (Mar–Apr), Pre-monsoon/Summer (May–Jun), Monsoon (Jul–Nov). Pre-monsoon is the most volatile season for demand.
- **`is_peak`, `is_morning_ramp`, `is_evening_ramp`:** Binary flags for the three highest-error time windows identified in error analysis (hours 6–8, 16–18, 18–21)

### 3.7 Interaction Features
- **`lag1h_x_hour_sin` / `lag1h_x_hour_cos`:** The same lag_1h value means different things at 3 AM vs 5 PM. These interactions let the model adapt its use of the lag depending on the time of day.
- **`hour_x_weekend`:** Weekend demand has a completely different hourly profile — this feature captures that interaction explicitly.

---

## 4. Model Selection & Training

### 4.1 Why LightGBM + XGBoost?
Both are gradient boosted tree ensembles — well-suited for tabular time-series with engineered lag features. LightGBM uses leaf-wise tree growth and gain-based importance, which tends to handle skewed feature importance distributions (like our lag_1h dominance) more effectively. XGBoost was included as a strong baseline for comparison.

Deep learning architectures (LSTMs, Transformers) and autoregressive packages (ARIMA, Prophet) were explicitly excluded per task constraints.

### 4.2 Chronological Train/Test Split
- **Training:** All data up to and including 2023 (76,128 hours)
- **Test:** 2024 onwards (12,805 hours)

This is the only valid approach for time series — no shuffling, no random splits. Using 2024 as the holdout ensures the model is evaluated on genuinely unseen future data.

### 4.3 Optuna Hyperparameter Tuning
Both models were tuned using **Optuna** with 60 trials each. The objective function was MAPE on the last 10% of the training set (chronological validation window — approximately mid-2022 to end of 2023). Early stopping (50 rounds) was used inside each trial to prevent overfitting.

After tuning, both models were retrained on the **full** training set with their best parameters before evaluation on the 2024 test set.

**Best hyperparameters found (LightGBM):**
- n_estimators: 1869, max_depth: 9, learning_rate: 0.064, num_leaves: 25, min_child_samples: 73, subsample: 0.729, colsample_bytree: 0.869

---

## 5. Results

| Model | Val MAPE (Optuna) | Test MAPE (2024) |
|---|---|---|
| LightGBM | 3.8152% | **2.6712%** |
| XGBoost | 3.8479% | 2.6748% |

**Winner: LightGBM**
- MAE: 249.07 MW
- RMSE: 474.85 MW
- MAPE: **2.6712%**

The gap between validation MAPE (~3.8%) and test MAPE (~2.67%) is expected: the final model trains on 100% of training data (vs 90% in Optuna trials), and 2024 is a more stable year for Bangladesh's grid than the 2022–2023 validation window.

A MAPE below 3% for classical ML on a national grid is considered strong performance in the energy forecasting literature.

---

## 6. Feature Importance Insights

From LightGBM's gain-based importance:

1. **`demand_lag_1h`** dominates — next hour's demand is overwhelmingly predicted by this hour's demand. This is expected for any short-term load forecasting task.
2. **`hour_sin` / `hour_cos`** are the second most important group — time of day is the primary seasonal driver after recent history.
3. **`demand_delta_1h`** and **`demand_ramp_3h`** contribute meaningfully — the direction and speed of demand change adds real signal beyond the lag levels alone.
4. **`demand_rolling_std_6h`** — volatility in the recent window is a useful signal for uncertain hours.
5. **Weather features** (temperature, THI, temp lags) contribute measurably, especially during pre-monsoon months.
6. **Economic indicators** have near-zero importance — they capture long-run structural shifts in demand levels, but at the hourly scale their effect is swamped by the immediate demand history.

---

## 7. Limitations & Future Work

- **Holiday effects:** Major Bangladeshi holidays (Eid al-Fitr, Eid al-Adha, Independence Day) cause demand patterns that differ substantially from regular days. Adding a holiday flag would likely reduce errors on those specific days.
- **Load shedding events:** Some anomalous low-demand readings in the test set (values below 3,000 MW) appear to be data logging errors during load shedding events rather than true demand. Better anomaly detection on the test side could improve evaluation accuracy.
- **Ensemble stacking:** A meta-learner combining LightGBM and XGBoost predictions could squeeze additional MAPE reduction, though the gains would be marginal given how close the two models already are (2.6712% vs 2.6748%).

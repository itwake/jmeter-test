# FX Forecasting POC – Technical Architecture & Justification

## 🧠 1. Model Selection: LightGBM

### ✅ Why LightGBM?
- Fast training speed, optimized for large datasets;
- Native handling of categorical and missing values;
- Provides interpretable feature importance;
- Generally better performance than most tree models.

### 🔄 Alternatives Comparison

| Model       | Pros                      | Cons                        |
|-------------|---------------------------|-----------------------------|
| XGBoost     | High accuracy, stable     | Slightly slower training    |
| CatBoost    | Excellent for categorical | Less mainstream             |
| RandomForest| Simple, robust            | Weak for time-sensitive data|
| LSTM/GRU    | Good for sequences        | Slow, less interpretable    |
| Prophet     | Trend & holiday support   | Not suited for hourly data  |

---

## 🧱 2. Feature Engineering Techniques

### ✅ Lag Features
- Captures short-term trends;
- Easy to interpret;
- Alternative: Let RNNs learn lag patterns.

### ✅ Rolling Statistics (Mean/Std)
- Highlights short-term volatility;
- Alternative: Use EMA (Exponential Moving Average).

### ✅ Time-based Features (hour, day_of_week)
- Captures periodic behavior;
- Alternative: Use time embeddings in deep models.

### ✅ Cyclical Encoding (sin/cos)
- Maintains continuity in cyclical features (e.g., hours);
- Alternative: One-hot (non-cyclical), embeddings.

### ✅ Holiday Features (HK/JP)
- Important for capturing special periods;
- Alternative: Use lead-up-to-holiday windows or cross-country patterns.

---

## 📊 3. Evaluation Metrics

| Metric | Purpose                     | Alternatives              |
|--------|-----------------------------|---------------------------|
| MSE    | Penalizes large errors      | RMSE, MAPE                |
| MAE    | Intuitive average error     | MedAE                     |
| WMAPE  | Business-friendly accuracy  | Add cost-based weighting |

---

## 🔍 4. Stationarity Check: ADF Test

- Used to verify if volume series is stationary;
- p-value < 0.05 indicates stable;
- Alternatives: KPSS test, Hurst Exponent.

---

## 📈 5. Strategy Logic & Business Simulation

### ✅ Peak Volume Rule
- Predict volume > 3× 90-day hourly median;
- Alternative: Use StdDev or IQR method.

### ✅ Margin Simulation
- Assume adding 0.05% doesn’t affect customer behavior;
- Can be further validated via A/B testing.

---

## ✅ Summary Table

| Module     | Current Choice | Reason                     | Are Alternatives Better?         |
|------------|----------------|-----------------------------|----------------------------------|
| Model      | LightGBM       | Fast, interpretable         | Deep models are stronger but complex |
| Features   | Lag + Rolling + Time | Efficient at short-term trend detection | Could improve with macro & behavior features |
| Strategy   | Median threshold + margin raise | Simple & practical | Business-driven optimization possible |


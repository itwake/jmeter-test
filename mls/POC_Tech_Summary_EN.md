
# FX Trading Volume Forecasting POC - Technical Summary

## 1. Modeling & Machine Learning

### ✅ LightGBM Model
- Used to predict hourly trading volume.
- Efficient for structured data; supports feature importance analysis.

---

## 2. Feature Engineering

### 🔹 Lag Features
- Use trading volume from past 1-5 hours to capture short-term trends.

### 🔹 Rolling Statistics
- Recent volume mean and std to identify spikes and anomalies.

### 🔹 Time Features
- Hour, day of week, month, etc. for periodic trading patterns.

### 🔹 Fourier Encoding
- Sin/cos encoding to model time periodicity in nonlinear ways.

### 🔹 Holiday Features
- Mark holidays in HK and JP to account for market behavior shifts.

---

## 3. Model Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **WMAPE**: Weighted Mean Absolute Percentage Error
  - Buy-side WMAPE ≈ 9.83%
  - Sell-side WMAPE ≈ 17.48%

---

## 4. Time Series Stationarity Check

### ✅ ADF Test
- ADF = -13.13, p-value ≈ 1.47e-24
- Conclusion: Time series is stationary and suitable for modeling.

---

## 5. Peak Detection & Margin Strategy

### ✅ Peak Rule
- A time slot is "peak" if predicted volume > 3x or 4x historical median.

### ✅ Margin Simulation
- Increase margin by 0.05% during peaks (assumes no user drop).
- Simulated potential revenue increase.

---

## 6. A/B Testing Design
- Test group: Adjust margins based on model.
- Control group: Use business heuristics.
- Compare actual volume and profits.

---

## 7. Feature Importance
- LightGBM's built-in importance scores.
- Rolling mean and lag features are most critical.

---

## 8. Technical Workflow Diagram

```
Data Collection → Feature Engineering → Modeling → Evaluation
                   ↓
      Peak Detection + Margin Simulation + Revenue Analysis
```

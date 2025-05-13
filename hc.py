import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

# 1. 数据加载
df = pd.read_csv("hsi_history.csv",
                 parse_dates=["dateTime"],
                 dtype={"status": str})

# 保留可交易状态
df = df[df["status"] == "T"].reset_index(drop=True)

# 2. 类型转换 & 原始特征
df["price"]    = df["current"].astype(float)
df["change"]   = df["change"].astype(float)
df["percent"]  = df["percent"].astype(float)  # 已经是百分数形式，如 -1.5
df["high_o"]   = df["high"].astype(float)
df["low_o"]    = df["low"].astype(float)
df["turnover"] = df["turnover"].astype(float)

# 3. 滑动窗口特征
window = 60  # 60秒窗口
df["mean_60"]     = df["price"].rolling(window).mean()
df["std_60"]      = df["price"].rolling(window).std()
df["max_60"]      = df["price"].rolling(window).max()
df["min_60"]      = df["price"].rolling(window).min()
df["momentum_60"] = df["price"] - df["price"].shift(window)

# 4. 技术比率特征
df["zscore"]   = (df["price"] - df["mean_60"]) / df["std_60"]
df["hl_ratio"] = (df["price"] - df["min_60"]) / (df["max_60"] - df["min_60"] + 1e-9)

# 5. 日内位置特征
df["pos_high"] = (df["price"] - df["low_o"]) / (df["high_o"] - df["low_o"] + 1e-9)
df["vol_ratio"]= df["turnover"] / (df["turnover"].rolling(window).mean() + 1e-9)

# 6. 丢弃空值
df = df.dropna().reset_index(drop=True)

# 7. 标签：未来 ΔT 秒内跌超 0.2% 标为 “Sell”
delta_t = 10
theta   = 0.002
df["future_min"] = df["price"].rolling(delta_t, min_periods=1).min().shift(-delta_t)
df["label"] = np.where(
    (df["future_min"] - df["price"]) / df["price"] <= -theta,
    1, 0
)
df = df.dropna(subset=["label"]).reset_index(drop=True)

# 8. 准备训练与测试
features = [
    "price", "change", "percent",
    "high_o", "low_o", "turnover",
    "mean_60", "std_60", "max_60", "min_60",
    "momentum_60", "zscore", "hl_ratio",
    "pos_high", "vol_ratio"
]
X = df[features]
y = df["label"]

tscv = TimeSeriesSplit(n_splits=5)
reports = []
for tr_idx, te_idx in tscv.split(X):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_eval  = lgb.Dataset(X_te, label=y_te, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_eval],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    y_pred = (gbm.predict(X_te, num_iteration=gbm.best_iteration) > 0.5).astype(int)
    reports.append(classification_report(y_te, y_pred, output_dict=True))

# 输出各折 F1-score
df_report = pd.DataFrame(reports)[["0", "1"]].applymap(lambda m: m["f1-score"])
print("各折 Hold/Sell F1-score：")
print(df_report)

# 9. 在线预测示例
def should_sell(latest: pd.DataFrame, model: lgb.Booster) -> bool:
    # latest: 最近 window 秒的原始 DataFrame，含 current, change, percent, high, low, turnover
    latest = latest.copy()
    latest["price"]    = latest["current"].astype(float)
    latest["change"]   = latest["change"].astype(float)
    latest["percent"]  = latest["percent"].astype(float)
    latest["high_o"]   = latest["high"].astype(float)
    latest["low_o"]    = latest["low"].astype(float)
    latest["turnover"] = latest["turnover"].astype(float)

    latest["mean_60"]     = latest["price"].rolling(window).mean()
    latest["std_60"]      = latest["price"].rolling(window).std()
    latest["max_60"]      = latest["price"].rolling(window).max()
    latest["min_60"]      = latest["price"].rolling(window).min()
    latest["momentum_60"] = latest["price"] - latest["price"].shift(window)

    latest["zscore"]   = (latest["price"] - latest["mean_60"]) / latest["std_60"]
    latest["hl_ratio"] = (latest["price"] - latest["min_60"]) / (latest["max_60"] - latest["min_60"] + 1e-9)
    latest["pos_high"] = (latest["price"] - latest["low_o"]) / (latest["high_o"] - latest["low_o"] + 1e-9)
    latest["vol_ratio"]= latest["turnover"] / (latest["turnover"].rolling(window).mean() + 1e-9)

    X_new = latest.dropna().iloc[[-1]][features]
    prob = model.predict(X_new)[0]
    return prob > 0.5

# 保存模型
gbm.save_model("hsi_sell_model.txt")
# 加载模型示例
# model = lgb.Booster(model_file="hsi_sell_model.txt")

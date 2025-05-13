import logging
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from ligh:contentReference[oaicite:3]{index=3}uation
from sklearn.model_:contentReference[oaicite:4]{index=4} i:contentReference[oaicite:5]{index=5}quared_error

# 1) 加载数据
df = pd.read_csv("hsi_history.csv",
                 parse_dates=["dateTime"],
                 dtype={"status": str})
df = df[df["status"] == "T"].reset_index(drop=True)

# 2) 转换数据类型
df["price"]    = df["current"].astype(float)
df["change"]   = df["change"].astype(float)
df["percent"]  = df["percent"].astype(float)
df["high_o"]   = df["high"].astype(float)
df["low_o"]    = df["low"].astype(float)
df["turnover"] = df["turnover"].astype(float)

# 3) 构造滑动窗口特征
window = 60
df["mean_60"]     = df["price"].rolling(window).mean()
df["std_60"]      = df["price"].rolling(window).std()
df["max_60"]      = df["price"].rolling(window).max()
df["min_60"]      = df["price"].rolling(window).min()

# 4) 衍生比率指标
df["zscore"]   = (df["price"] - df["mean_60"]) / (df["std_60"] + 1e-9)
df["hl_ratio"] = (df["price"] - df["min_60"]) / ((df["max_60"] - df["min_60"]) + 1e-9)
df["pos_high"] = (df["price"] - df["low_o"]) / ((df["high_o"] - df["low_o"]) + 1e-9)
df["vol_ratio"]= df["turnover"] / (df["turnover"].rolling(window).mean() + 1e-9)

# 5) 丢弃空值
df = df.dropna().reset_index(drop=True)

# 6) 构造回归目标：预测 ΔT 秒后的价格
delta_t = 10  # 往后预测 10 秒
df["future_price"] = df["price"].shift(-delta_t)
df = df.dropna(subset=["future_price"]).reset_index(drop=True)

# 7) 划分训练/测试
features = [
    "price", "change", "percent",
    "high_o", "low_o", "turnover",
    "mean_60", "std_60", "max_60", "min_60",
    "zscore", "hl_ratio", "pos_high", "vol_ratio"
]
X = df[features]
y = df["future_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# 8) 训练回归模型（带早停 & 静默）
model = LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.05,
    n_estimators=500,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)

callbacks = [
    early_stopping(stopping_rounds=20, first_metric_only=True, verbose=False),
    log_evaluation(period=0)
]

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks
)

# 9) 评估
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 10) 保存模型
with open("hsi_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to hsi_price_model.pkl")


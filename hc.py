import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

# 1. 数据加载
# 假设 df.csv 包含字段：
# dateTime,status,current,change,percent,high,low,turnover
df = pd.read_csv("hsi_history.csv",
                 parse_dates=["dateTime"],
                 dtype={"status": str})

# 只保留可交易状态
df = df[df["status"] == "T"].reset_index(drop=True)

# 2. 特征工程
window = 60  # 60秒窗口
df["price"] = df["current"].astype(float)

# 短期特征
df["mean_60"] = df["price"].rolling(window).mean()
df["std_60"]  = df["price"].rolling(window).std()
df["max_60"]  = df["price"].rolling(window).max()
df["min_60"]  = df["price"].rolling(window).min()

# 相对位置
df["zscore"] = (df["price"] - df["mean_60"]) / df["std_60"]
df["hl_ratio"] = (df["price"] - df["min_60"]) / (df["max_60"] - df["min_60"] + 1e-9)

# 动量：当前价与 window 秒前价之差
df["momentum_60"] = df["price"] - df["price"].shift(window)

# 删除空值行
df = df.dropna().reset_index(drop=True)

# 3. 标签设计：未来 ΔT 秒内下跌超过 0.2% 即标记为 “Sell”
delta_t = 10  # 秒
theta = 0.002  # 0.2%
# 先构造未来价格列
df["future_min"] = df["price"].rolling(delta_t, min_periods=1).min().shift(-delta_t)
df["label"] = np.where(
    (df["future_min"] - df["price"]) / df["price"] <= -theta,
    1,  # Sell
    0   # Hold
)
df = df.dropna(subset=["label"]).reset_index(drop=True)

# 4. 准备训练集与测试集（时间序列切分）
features = ["price", "mean_60", "std_60", "max_60", "min_60",
            "zscore", "hl_ratio", "momentum_60"]
X = df[features]
y = df["label"]

tscv = TimeSeriesSplit(n_splits=5)
reports = []
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

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

    # 预测与评估
    y_pred = (gbm.predict(X_te, num_iteration=gbm.best_iteration) > 0.5).astype(int)
    report = classification_report(y_te, y_pred, output_dict=True)
    reports.append(report)

# 汇总各折 F1-score
df_report = pd.DataFrame(reports)[["0", "1"]].applymap(lambda x: x["f1-score"])
print("各折 Sell 类别(F1) 与 Hold 类别(F1):")
print(df_report)

# 5. 在线预测接口示例
def should_sell(latest_batch: pd.DataFrame, model: lgb.Booster) -> bool:
    """
    :param latest_batch: 包含最近 window 秒的原始数据 DataFrame，
                         字段需与训练时一致
    :param model: 训练好的 LightGBM 模型
    :return: True=Sell, False=Hold
    """
    # 在此重新计算特征
    X_new = pd.DataFrame({
        "price": latest_batch["current"].astype(float),
        "mean_60": latest_batch["current"].astype(float).rolling(window).mean(),
        "std_60":  latest_batch["current"].astype(float).rolling(window).std(),
        "max_60":  latest_batch["current"].astype(float).rolling(window).max(),
        "min_60":  latest_batch["current"].astype(float).rolling(window).min(),
    })
    X_new["zscore"] = (X_new["price"] - X_new["mean_60"]) / X_new["std_60"]
    X_new["hl_ratio"] = (X_new["price"] - X_new["min_60"]) / (X_new["max_60"] - X_new["min_60"] + 1e-9)
    X_new["momentum_60"] = X_new["price"] - X_new["price"].shift(window)
    X_new = X_new.dropna().iloc[[-1]]  # 取最后一行

    prob = model.predict(X_new)[0]
    return prob > 0.5  # 阈值可调

# 保存 & 加载模型
gbm.save_model("hsi_sell_model.txt")
# later...
# model = lgb.Booster(model_file="hsi_sell_model.txt")

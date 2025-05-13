import logging
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 1. 加载并预处理数据
logger.info("1. Loading HSI history data")
df = pd.read_csv(
    "hsi_history.csv",
    parse_dates=["dateTime"],
    dtype={"status": str}
)
df = df[df["status"] == "T"].reset_index(drop=True)
df["price"]   = df["current"].astype(float)
df["change"]  = df["change"].astype(float)
df["percent"] = df["percent"].astype(float)

# 2. 特征工程：滑动窗口统计
window = 60
logger.info("2. Computing rolling-window features (window=%d)", window)
df["mean_60"]     = df["price"].rolling(window).mean()
df["std_60"]      = df["price"].rolling(window).std()
df["max_60"]      = df["price"].rolling(window).max()
df["min_60"]      = df["price"].rolling(window).min()
df["momentum_60"] = df["price"] - df["price"].shift(window)
df["zscore"]      = (df["price"] - df["mean_60"]) / (df["std_60"] + 1e-9)
df = df.dropna().reset_index(drop=True)

# 3. 构造回归目标
delta_t = 10
df["future_price"] = df["price"].shift(-delta_t)
df = df.dropna(subset=["future_price"]).reset_index(drop=True)
logger.info("3. Constructed future_price target (ΔT=%d), samples: %d", delta_t, len(df))

# 4. 划分训练集 / 测试集
features = [
    "price", "change", "percent",
    "mean_60", "std_60", "max_60", "min_60",
    "momentum_60", "zscore"
]
X = df[features]
y = df["future_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
logger.info("4. Train/Test split: %d / %d", len(X_train), len(X_test))

# 5. 训练模型
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
logger.info("5. Training model with early stopping")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks
)
logger.info("  → Training completed, best iteration: %d", model.best_iteration_)

# 6. 评估
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
logger.info("6. Evaluation — MAE: %.4f, RMSE: %.4f", mae, rmse)

# 7. 保存模型
with open("hsi_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
logger.info("7. Model saved to hsi_price_model.pkl")

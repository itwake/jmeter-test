import logging
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 2. 加载并预处理数据
logger.info("Loading data from hsi_history.csv")
df = pd.read_csv("hsi_history.csv", parse_dates=["dateTime"], dtype={"status": str})
df = df[df["status"] == "T"].reset_index(drop=True)
df["price"]    = df["current"].astype(float)
df["change"]   = df["change"].astype(float)
df["percent"]  = df["percent"].astype(float)
df["high_o"]   = df["high"].astype(float)
df["low_o"]    = df["low"].astype(float)
df["turnover"] = df["turnover"].astype(float)

# 3. 特征工程
window = 60
logger.info("Computing rolling features with window=%d", window)
df["mean_60"]     = df["price"].rolling(window).mean()
df["std_60"]      = df["price"].rolling(window).std()
df["max_60"]      = df["price"].rolling(window).max()
df["min_60"]      = df["price"].rolling(window).min()
df["momentum_60"] = df["price"] - df["price"].shift(window)
df["zscore"]      = (df["price"] - df["mean_60"]) / df["std_60"]
df["hl_ratio"]    = (df["price"] - df["min_60"]) / (df["max_60"] - df["min_60"] + 1e-9)
df["pos_high"]    = (df["price"] - df["low_o"]) / (df["high_o"] - df["low_o"] + 1e-9)
df["vol_ratio"]   = df["turnover"] / (df["turnover"].rolling(window).mean() + 1e-9)

df = df.dropna().reset_index(drop=True)
logger.info("After feature engineering, dataset shape: %s", df.shape)

# 4. 构造标签
delta_t = 10
theta   = 0.002
logger.info("Generating labels with delta_t=%ds and threshold=%.3f", delta_t, theta)
df["future_min"] = df["price"].rolling(delta_t, min_periods=1).min().shift(-delta_t)
df["label"] = np.where(
    (df["future_min"] - df["price"]) / df["price"] <= -theta,
    1, 0
)
df = df.dropna(subset=["label"]).reset_index(drop=True)
logger.info("Final dataset for modeling: %d samples", len(df))

# 5. 划分训练/测试集
features = [
    "price", "change", "percent", "high_o", "low_o", "turnover",
    "mean_60", "std_60", "max_60", "min_60",
    "momentum_60", "zscore", "hl_ratio",
    "pos_high", "vol_ratio"
]
X = df[features]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
logger.info("Train/Test split: %d / %d", len(X_train), len(X_test))

# 6. 模型定义
model = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    learning_rate=0.05,
    n_estimators=200,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)

# 7. 训练：使用 callbacks 启用早停和静默输出 :contentReference[oaicite:0]{index=0}
callbacks = [
    early_stopping(stopping_rounds=20, first_metric_only=True, verbose=False),
    log_evaluation(period=0)
]
logger.info("Begin training with early stopping (callbacks)")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks
)
logger.info("Training completed. Best iteration: %d", model.best_iteration_)

# 8. 最终验证
logger.info("Running final evaluation on test set")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
logger.info("Classification Report:\n%s", report)
logger.info("Confusion Matrix:\n%s", cm)

# 9. 保存模型
model.booster_.save_model("hsi_sell_model.txt")
logger.info("Model saved to hsi_sell_model.txt")

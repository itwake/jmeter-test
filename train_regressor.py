# train_and_inspect.py

import logging
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ——————————————————————
# 日志配置
# ——————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ——————————————————————
# 1. 加载并预处理数据
# ——————————————————————
logger.info("Loading data")
df = pd.read_csv("hsi_history.csv", parse_dates=["dateTime"], dtype={"status": str})
df = df[df["status"]=="T"].reset_index(drop=True)
df["price"]   = df["current"].astype(float)
df["change"]  = df["change"].astype(float)
df["percent"] = df["percent"].astype(float)

# ——————————————————————
# 2. 构造多周期滚动特征与滞后特征
# ——————————————————————
logger.info("Computing multi-horizon rolling features")
# 定义窗口长度（秒）
w1m  = 60
w30m = 30*60
w1h  = 60*60
w2h  = 2*60*60

# 滚动均值与标准差
df["mean_1m"]  = df["price"].rolling(w1m).mean()
df["std_1m"]   = df["price"].rolling(w1m).std()
df["mean_30m"] = df["price"].rolling(w30m).mean()
df["std_30m"]  = df["price"].rolling(w30m).std()
df["mean_1h"]  = df["price"].rolling(w1h).mean()
df["std_1h"]   = df["price"].rolling(w1h).std()
df["mean_2h"]  = df["price"].rolling(w2h).mean()
df["std_2h"]   = df["price"].rolling(w2h).std()

# 滞后价格
df["price_1m_ago"]  = df["price"].shift(w1m)
df["price_30m_ago"] = df["price"].shift(w30m)
df["price_1h_ago"]  = df["price"].shift(w1h)
df["price_2h_ago"]  = df["price"].shift(w2h)

# 丢弃空值
df = df.dropna().reset_index(drop=True)
logger.info("After feature engineering: %d rows", len(df))

# ——————————————————————
# 3. 构造回归目标
# ——————————————————————
delta_t = 10  # 预测未来 10 秒
df["future_price"] = df["price"].shift(-delta_t)
df = df.dropna(subset=["future_price"]).reset_index(drop=True)
logger.info("Built target, samples left: %d", len(df))

# ——————————————————————
# 4. 划分训练/测试
# ——————————————————————
features = [
    "price", "change", "percent",
    "mean_1m", "std_1m",
    "mean_30m", "std_30m",
    "mean_1h", "std_1h",
    "mean_2h", "std_2h",
    "price_1m_ago", "price_30m_ago",
    "price_1h_ago", "price_2h_ago"
]
X = df[features]
y = df["future_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
logger.info("Train/Test split: %d / %d", len(X_train), len(X_test))

# ——————————————————————
# 5. 训练 LightGBM 回归模型
# ——————————————————————
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

logger.info("Training model")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks
)
logger.info("Training complete, best_iteration=%d", model.best_iteration_)

# ——————————————————————
# 6. 模型评估
# ——————————————————————
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
logger.info("Evaluation: MAE=%.4f, RMSE=%.4f", mae, rmse)

# ——————————————————————
# 7. 特征重要性输出
# ——————————————————————
importances = model.feature_importances_
feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
logger.info("Feature importances (desc):")
for feat, imp in feat_imp:
    logger.info("  %s: %d", feat, imp)

# ——————————————————————
# 8. 保存模型
# ——————————————————————
with open("hsi_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
logger.info("Model saved")

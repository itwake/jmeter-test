
# train_multi_return.py
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 超参数
REQUEST_INTERVAL = 2  # 秒
HORIZONS = {"2s": 1, "6s": 3, "10s": 5}
FEAT_WINDOWS = {"10s": 10, "20s": 20, "30s": 30}
WINDOW_COUNTS = {k: int(v / REQUEST_INTERVAL) for k, v in FEAT_WINDOWS.items()}
MODEL_PATH = "multi_horizon_return_model.pkl"


def main():
    # 1. 加载与预处理
    logger.info("加载历史数据并预处理")
    df = pd.read_csv(
        "hsi_history.csv",
        parse_dates=["dateTime"],
        dtype={"status": str}
    )
    df = df[df["status"] == "T"].reset_index(drop=True)
    df["price"] = df["current"].astype(float)
    df["change"] = df["change"].astype(float)
    df["percent"] = df["percent"].astype(float)

    # 2. 构造特征
    logger.info("构造窗口特征: %s 样本点", WINDOW_COUNTS)
    for name, cnt in WINDOW_COUNTS.items():
        df[f"mean_{name}"] = df["price"].rolling(cnt).mean()
        df[f"std_{name}"] = df["price"].rolling(cnt).std()
        df[f"price_{name}_ago"] = df["price"].shift(cnt)
    df = df.dropna().reset_index(drop=True)
    logger.info("特征构造后样本数: %d", len(df))

    # 3. 构造相对收益标签
    logger.info("构造相对收益标签: %s", HORIZONS)
    for label, step in HORIZONS.items():
        df[f"ret_{label}"] = (df["price"].shift(-step) - df["price"]) / df["price"]
    df = df.dropna().reset_index(drop=True)
    logger.info("标签构造后样本数: %d", len(df))

    # 4. 划分数据集
    feature_cols = ["price", "change", "percent"] + \
                   [f"mean_{w}" for w in FEAT_WINDOWS] + \
                   [f"std_{w}" for w in FEAT_WINDOWS] + \
                   [f"price_{w}_ago" for w in FEAT_WINDOWS]
    target_cols = [f"ret_{h}" for h in HORIZONS]
    X = df[feature_cols]
    Y = df[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )
    logger.info("训练/测试集: %d / %d", len(X_train), len(X_test))

    # 5. 训练模型
    base = LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5
    )
    model = MultiOutputRegressor(base)
    logger.info("训练多输出回归模型")
    model.fit(X_train, y_train)
    logger.info("训练完成")

    # 6. 评估
    y_pred = pd.DataFrame(model.predict(X_test), columns=target_cols)
        time.sleep(strat.request_interval)

import logging
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ——————————————————
# 日志配置
# ——————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ——————————————————
# 参数设置
# ——————————————————
request_interval = 2  # 采样间隔（秒）
windows_secs = {"10s": 10, "20s": 20, "30s": 30}
# 转换为样本数量
window_counts = {k: int(v / request_interval) for k, v in windows_secs.items()}
delta_t = 10  # 预测未来 10 秒后的价格

# ——————————————————
# 主流程
# ——————————————————
def main():
    # 1. 数据加载与预处理
    logger.info("加载 HSI 历史数据")
    df = pd.read_csv(
        "hsi_history.csv",
        parse_dates=["dateTime"],
        dtype={"status": str}
    )
    df = df[df["status"] == "T"].reset_index(drop=True)
    df["price"] = df["current"].astype(float)
    df["change"] = df["change"].astype(float)
    df["percent"] = df["percent"].astype(float)
    logger.info("状态=T样本数: %d", len(df))

    # 2. 构造滚动窗口特征与滞后特征
    logger.info("计算窗口特征：%s 样本点", windows_secs)
    for name, cnt in window_counts.items():
        df[f"mean_{name}"] = df["price"].rolling(cnt).mean()
        df[f"std_{name}"]  = df["price"].rolling(cnt).std()
        df[f"price_{name}_ago"] = df["price"].shift(cnt)
    df = df.dropna().reset_index(drop=True)
    logger.info("特征构造后样本数: %d", len(df))

    # 3. 构造回归目标
    df["future_price"] = df["price"].shift(-delta_t)
    df = df.dropna(subset=["future_price"]).reset_index(drop=True)
    logger.info("目标构造ΔT=%d后样本: %d", delta_t, len(df))

    # 4. 划分训练/测试集
    features = ["price", "change", "percent"] + \
               [f"mean_{w}" for w in windows_secs] + \
               [f"std_{w}" for w in windows_secs] + \
               [f"price_{w}_ago" for w in windows_secs]
    X = df[features]
    y = df["future_price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    logger.info("划分训练/测试: %d / %d", len(X_train), len(X_test))

    # 5. 训练 LightGBM 回归模型
    model = LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5
    )
    callbacks = [
        early_stopping(stopping_rounds=20, first_metric_only=True, verbose=False),
        log_evaluation(period=0)
    ]
    logger.info("开始训练模型")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=callbacks
    )
    logger.info("训练完成，best_iteration=%d", model.best_iteration_)

    # 6. 模型评估
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("评估结果MAE=%.4f, RMSE=%.4f", mae, rmse)

    # 7. 特征重要性
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    logger.info("特征重要性(降序):")
    for feat, imp in feat_imp:
        logger.info("  %s: %d", feat, imp)

    # 8. 保存模型
    with open("hsi_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("模型已保存至 hsi_price_model.pkl")

if __name__ == "__main__":
    main()

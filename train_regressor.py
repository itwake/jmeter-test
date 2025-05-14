# train_regressor.py
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
windows_secs = {"2s": 2, "4s": 4, "10s": 10}
# 转换为样本数量
window_counts = {k: int(v / request_interval) for k, v in windows_secs.items()}
delta_t = 1  # 预测未来 2 秒后的价格（1 个采样点）

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
    df["price"]   = df["current"].astype(float)
    df["change"]  = df["change"].astype(float)
    df["percent"] = df["percent"].astype(float)
    logger.info("状态=T样本数: %d", len(df))

    # 2. 构造滚动窗口特征与滞后特征
    logger.info("计算窗口特征：%s 对应样本数", windows_secs)
    for name, cnt in window_counts.items():
        df[f"mean_{name}"]      = df["price"].rolling(cnt).mean()
        df[f"std_{name}"]       = df["price"].rolling(cnt).std()
        df[f"price_{name}_ago"] = df["price"].shift(cnt)
    df = df.dropna().reset_index(drop=True)
    logger.info("特征构造后样本数: %d", len(df))

    # 3. 构造回归目标
    df["future_price"] = df["price"].shift(-delta_t)
    df = df.dropna(subset=["future_price"]).reset_index(drop=True)
    logger.info("目标构造 ΔT=%.0f秒 后样本: %d", delta_t*request_interval, len(df))

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
    logger.info("训练/测试集划分: %d / %d", len(X_train), len(X_test))

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
    logger.info("评估结果 MAE=%.4f, RMSE=%.4f", mae, rmse)

    # 7. 特征重要性
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    logger.info("特征重要性 (降序):")
    for feat, imp in feat_imp:
        logger.info("  %s: %d", feat, imp)

    # 8. 保存模型
    with open("hsi_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("模型已保存至 hsi_price_model.pkl")

if __name__ == "__main__":
    main()


# strategy_with_regression.py
import logging
import datetime
import time
import requests
import pickle
import pandas as pd
import numpy as np
from collections import deque

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class SellStrategy:
    def __init__(
        self,
        total_shares=100,
        window_size=30,         # 滑动窗口长度（秒），用于特征重建，最大30秒
        batch_count=5,
        request_interval=2,
        model_path="hsi_price_model.pkl",
        buffer_minutes=1        # 提前结束交易的缓冲时间（分钟）
    ):
        # 卖出参数
        self.total_shares     = total_shares
        self.shares_remaining = total_shares
        self.batch_count      = batch_count

        # 交易时长与清仓缓冲
        self.trading_duration   = datetime.timedelta(minutes=10)
        self.clear_buffer       = datetime.timedelta(minutes=buffer_minutes)
        self.effective_duration = self.trading_duration - self.clear_buffer
        self.batch_duration     = self.effective_duration / batch_count
        self.request_interval   = request_interval

        # 时间跟踪
        self.start_dt = None
        self.last_dt  = None

        # 滑动窗口，用于多周期特征
        self.price_window = deque(maxlen=window_size)

        # 加载模型
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Loaded model from %s", model_path)

    def get_index_status(self):
        """
        获取最新行情，返回包含 dateTime, status, current, change, percent
        """
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        """
        构造特征：当前值，以及最近10s/20s/30s的均值、标准差和相应的滞后价格
        """
        price   = float(info["current"])
        change  = float(info["change"])
        percent = float(info["percent"])

        # 更新窗口
        self.price_window.append(price)
        if len(self.price_window) < self.price_window.maxlen:
            return None

        arr = np.array(self.price_window)
        windows = {"10s": 10, "20s": 20, "30s": 30}

        feats = {"price": price, "change": change, "percent": percent}
        for name, w in windows.items():
            vals = arr[-w:]
            feats[f"mean_{name}"]      = float(vals.mean())
            feats[f"std_{name}"]       = float(vals.std())
            feats[f"price_{name}_ago"] = float(arr[-w])

        return pd.DataFrame([feats])

    def sell_strategy(self):
        """
        卖出策略：获取行情 -> 特征 -> 预测 -> 卖出决策
        返回 ["sell", vol] 或 ["hold", 0]
        """
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(
            info["dateTime"], "%Y%m%d%H%M%S%f")

        # 去重
        if self.last_dt and current_dt <= self.last_dt:
            return ["hold", 0]
        self.last_dt = current_dt

        # 仅在交易开放期
        if info["status"] != "T":
            return ["hold", 0]

        # 记录开始时间
        if self.start_dt is None:
            self.start_dt = current_dt

        elapsed = current_dt - self.start_dt
        current_price = float(info["current"])
        sold = self.total_shares - self.shares_remaining

        # 基线卖出量：按时间进度线性分配
        t_sec = min(elapsed, self.effective_duration).total_seconds()
        eff_sec = self.effective_duration.total_seconds()
        req_cum = int(self.total_shares * t_sec / eff_sec)
        unsold_req = max(req_cum - sold, 0)

        # 构造特征
        feat = self.make_features(info)
        if feat is None:
            # 到缓冲结束 -> 清仓
            if elapsed >= self.effective_duration:
                vol = self.shares_remaining
                self.shares_remaining = 0
                logger.info(
                    "%s → FINAL CLEAR %d at %.2f",
                    current_dt.strftime("%H:%M:%S"), vol, current_price
                )
                return ["sell", vol]
            # 基线卖出
            if unsold_req > 0:
                vol = min(unsold_req, self.shares_remaining)
                self.shares_remaining -= vol
                logger.info(
                    "%s → BASELINE SELL %d at %.2f",
                    current_dt.strftime("%H:%M:%S"), vol, current_price
                )
                return ["sell", vol]
            return ["hold", 0]

        # 预测回归
        pred_price = self.model.predict(feat)[0]
        pred_ret = (pred_price - current_price) / current_price

        # 决策：基线 + 预测触发
        to_sell = unsold_req
        if pred_ret <= 0:
            phase_idx = min(int(elapsed / self.batch_duration), self.batch_count - 1)
            phase_target = int(self.total_shares * (phase_idx + 1) / self.batch_count)
            unsold_ph = max(phase_target - sold, 0)
            to_sell = max(to_sell, unsold_ph)

        # 到缓冲结束 -> 清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining

        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            logger.info(
                "%s → SELL %d at %.2f (pred_ret=%.4f, base=%d)",
                current_dt.strftime("%H:%M:%S"), vol, current_price, pred_ret, unsold_req
            )
            return ["sell", vol]
        return ["hold", 0]

if __name__ == "__main__":
    strat = SellStrategy()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)

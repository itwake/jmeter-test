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
request_interval = 2  # 秒
# 多时点预测步数（2s、6s、10s）
horizons = {"2s": 1, "6s": 3, "10s": 5}
# 特征窗口长度（秒）
feat_windows = {"10s": 10, "20s": 20, "30s": 30}
window_counts = {k: int(v / request_interval) for k, v in feat_windows.items()}

def main():
    # 1. 数据加载
    logger.info("加载历史数据并预处理")
    df = pd.read_csv("hsi_history.csv", parse_dates=["dateTime"], dtype={"status": str})
    df = df[df["status"] == "T"].reset_index(drop=True)
    df["price"] = df["current"].astype(float)
    df["change"] = df["change"].astype(float)
    df["percent"] = df["percent"].astype(float)

    # 2. 构造特征
    logger.info("构造窗口特征: %s 样本点", window_counts)
    for name, cnt in window_counts.items():
        df[f"mean_{name}"] = df["price"].rolling(cnt).mean()
        df[f"std_{name}"] = df["price"].rolling(cnt).std()
        df[f"price_{name}_ago"] = df["price"].shift(cnt)
    df = df.dropna().reset_index(drop=True)
    logger.info("特征构造后样本数: %d", len(df))

    # 3. 构造相对收益标签
    logger.info("构造相对收益标签: %s", horizons)
    for label, step in horizons.items():
        df[f"ret_{label}"] = (df["price"].shift(-step) - df["price"]) / df["price"]
    df = df.dropna().reset_index(drop=True)
    logger.info("标签构造后样本数: %d", len(df))

    # 4. 划分数据集
    feature_cols = ["price", "change", "percent"] + \
                   [f"mean_{w}" for w in feat_windows] + \
                   [f"std_{w}" for w in feat_windows] + \
                   [f"price_{w}_ago" for w in feat_windows]
    target_cols = [f"ret_{h}" for h in horizons]
    X = df[feature_cols]; Y = df[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    logger.info("训练/测试集: %d / %d", len(X_train), len(X_test))

    # 5. 训练多输出回归模型
    base = LGBMRegressor(
        boosting_type="gbdt", objective="regression",
        learning_rate=0.05, n_estimators=300,
        num_leaves=31, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5
    )
    model = MultiOutputRegressor(base)
    logger.info("训练多输出回归模型")
    model.fit(X_train, y_train)
    logger.info("训练完成")

    # 6. 模型评估
    y_pred = pd.DataFrame(model.predict(X_test), columns=target_cols)
    for col in target_cols:
        mae = mean_absolute_error(y_test[col], y_pred[col])
        rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[col]))
        logger.info("%s -> MAE=%.4f, RMSE=%.4f", col, mae, rmse)

    # 7. 保存模型
    with open("multi_horizon_return_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("模型已保存至 multi_horizon_return_model.pkl")

if __name__ == '__main__':
    main()


# strategy_multi_return.py (高位清仓触发)
import logging
import datetime
import time
import requests
import pickle
import numpy as np
from collections import deque
import pandas as pd

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class MultiHorizonReturnSell:
    def __init__(
        self, total_shares=100, window_size=30,
        batch_count=5, request_interval=2,
        model_path="multi_horizon_return_model.pkl",
        buffer_minutes=1
    ):
        # 参数与时间管理
        self.total_shares = total_shares
        self.shares_remaining = total_shares
        self.batch_count = batch_count
        self.request_interval = request_interval
        self.trading_duration = datetime.timedelta(minutes=10)
        self.clear_buffer = datetime.timedelta(minutes=buffer_minutes)
        self.effective_duration = self.trading_duration - self.clear_buffer
        self.batch_duration = self.effective_duration / batch_count
        # 初始跟踪
        self.start_dt = None
        self.last_dt = None
        self.segment_start = None
        self.segment_end = None
        self.segment_max = -np.inf
        self.segment_min = np.inf
        self.price_window = deque(maxlen=window_size)
        # 预测高点记录
        self.horizons_sec = np.array([2, 6, 10])
        self.pred_high_time = None
        # 模型加载
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info("加载模型 %s", model_path)

    def get_index_status(self):
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        price = float(info['current'])
        change = float(info['change'])
        percent = float(info['percent'])
        self.price_window.append(price)
        if len(self.price_window) < self.price_window.maxlen:
            return None
        arr = np.array(self.price_window)
        windows = {'10s':10, '20s':20, '30s':30}
        feats = {'price':price, 'change':change, 'percent':percent}
        for name, w in windows.items():
            vals = arr[-w:]
            feats[f'mean_{name}'] = float(vals.mean())
            feats[f'std_{name}'] = float(vals.std())
            feats[f'price_{name}_ago'] = float(arr[-w])
        return pd.DataFrame([feats])

    def sell_strategy(self):
        info = self.get_index_status()
        now = datetime.datetime.strptime(info['dateTime'], '%Y%m%d%H%M%S%f')
        price = float(info['current'])
        # 去重与非交易
        if (self.last_dt and now <= self.last_dt) or info['status'] != 'T':
            return ['hold', 0]
        self.last_dt = now
        # 初始化分段与高低点
        if self.start_dt is None:
            self.start_dt = now
            self.segment_start = now
            self.segment_end = now + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None
        elapsed = now - self.start_dt
        sold = self.total_shares - self.shares_remaining
        # 更新段内极值
        self.segment_max = max(self.segment_max, price)
        self.segment_min = min(self.segment_min, price)
        # 基线量
        t_sec = min(elapsed, self.effective_duration).total_seconds()
        base_cum = int(self.total_shares * t_sec / self.effective_duration.total_seconds())
        unsold_base = max(base_cum - sold, 0)
        # 首次预测并记录高点时间
        feat = self.make_features(info)
        if feat is not None and self.pred_high_time is None:
            preds = self.model.predict(feat)[0]
            predicted_prices = price * (1 + preds)
            idx = int(np.argmax(predicted_prices))
            offset = self.horizons_sec[idx]
            self.pred_high_time = now + datetime.timedelta(seconds=int(offset))
            logger.info("预测高点时间：%s", self.pred_high_time.strftime('%H:%M:%S'))
        to_sell = 0
        # 在预测高点时间到来时，一次性清仓
        if self.pred_high_time and now >= self.pred_high_time:
            to_sell = self.shares_remaining
            logger.info("到达预测高点，清仓剩余 %d 股", to_sell)
        # 段尾补基线
        elif now > self.segment_end:
            to_sell = max(to_sell, unsold_base)
            self.segment_start = self.segment_end
            self.segment_end = self.segment_start + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None
        # 缓冲结束清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining
        # 执行
        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            logger.info(
                "%s → SELL %d @%.2f (base=%d)",
                now.strftime('%H:%M:%S'), vol, price, unsold_base
            )
            return ['sell', vol]
        return ['hold', 0]

if __name__ == '__main__':
    strat = MultiHorizonReturnSell()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)

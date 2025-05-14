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
    for col in target_cols:
        mae = mean_absolute_error(y_test[col], y_pred[col])
        rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[col]))
        logger.info("%s -> MAE=%.4f, RMSE=%.4f", col, mae, rmse)

    # 7. 保存模型
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("模型已保存至 %s", MODEL_PATH)


if __name__ == '__main__':
    main()


# strategy_multi_return.py
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
        self,
        total_shares=100,
        window_size=30,
        batch_count=5,
        request_interval=2,
        model_path=MODEL_PATH,
        buffer_minutes=1,
        sell_multiplier=1.5
    ):
        self.total_shares = total_shares
        self.shares_remaining = total_shares
        self.batch_count = batch_count
        self.request_interval = request_interval
        self.trading_duration = datetime.timedelta(minutes=10)
        self.clear_buffer = datetime.timedelta(minutes=buffer_minutes)
        self.effective_duration = self.trading_duration - self.clear_buffer
        self.batch_duration = self.effective_duration / batch_count
        self.sell_multiplier = sell_multiplier
        self.start_dt = None
        self.last_dt = None
        self.segment_start = None
        self.segment_end = None
        self.segment_max = -np.inf
        self.segment_min = np.inf
        self.price_window = deque(maxlen=window_size)
        self.horizons_sec = np.array([2, 6, 10])
        self.pred_high_time = None
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
            feats[f"mean_{name}"] = float(vals.mean())
            feats[f"std_{name}"] = float(vals.std())
            feats[f"price_{name}_ago"] = float(arr[-w])
        return pd.DataFrame([feats])

    def sell_strategy(self):
        info = self.get_index_status()
        now = datetime.datetime.strptime(info['dateTime'], '%Y%m%d%H%M%S%f')
        price = float(info['current'])

        if (self.last_dt and now <= self.last_dt) or info['status'] != 'T':
            return ['hold', 0]
        self.last_dt = now

        if self.start_dt is None:
            self.start_dt = now
            self.segment_start = now
            self.segment_end = now + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None

        elapsed = now - self.start_dt
        sold = self.total_shares - self.shares_remaining

        self.segment_max = max(self.segment_max, price)
        self.segment_min = min(self.segment_min, price)

        # 基线与加速因子
        t_sec = min(elapsed, self.effective_duration).total_seconds()
        base_cum = int(self.total_shares * t_sec / self.effective_duration.total_seconds())
        unsold_base = max(base_cum - sold, 0)
        base_to_sell = int(unsold_base * self.sell_multiplier)
        to_sell = base_to_sell

        # 窗口高位加码
        win_arr = np.array(self.price_window)
        if unsold_base > 0 and win_arr.size:
            thresh = np.percentile(win_arr, 80)
            if price >= thresh:
                strength = (price - thresh) / (self.segment_max - thresh + 1e-9)
                extra = int(strength * unsold_base)
                to_sell = max(to_sell, base_to_sell + extra)
            elif price >= self.segment_max:
                to_sell = max(to_sell, base_to_sell)

        # 段尾补基线
        if now > self.segment_end:
            to_sell = max(to_sell, base_to_sell)
            self.segment_start = self.segment_end
            self.segment_end = self.segment_start + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None

        # 缓冲清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining

        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            logger.info(
                "%s → SELL %d @%.2f (base*mult=%d)" ,
                now.strftime('%H:%M:%S'), vol, price, base_to_sell
            )
            return ['sell', vol]
        return ['hold', 0]

if __name__ == '__main__':
    strat = MultiHorizonReturnSell()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)

import logging
import datetime
import time
import requests
import pickle
import pandas as pd
from collections import deque

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class SellStrategy:
    def __init__(self,
                 total_shares=100,
                 window_size=60,
                 batch_count=5,
                 request_interval=2,
                 model_path="hsi_price_model.pkl"):
        # 初始化交易参数
        self.total_shares     = total_shares
        self.shares_remaining = total_shares
        self.batch_count      = batch_count
        self.trading_duration = datetime.timedelta(minutes=10)
        self.batch_duration   = self.trading_duration / batch_count
        self.start_dt         = None
        self.last_dt          = None
        self.request_interval = request_interval

        # 滑动窗口，用于特征重建
        self.price_window   = deque(maxlen=window_size)
        self.change_window  = deque(maxlen=window_size)
        self.percent_window = deque(maxlen=window_size)

        # 加载训练好的回归模型
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Loaded regression model from %s", model_path)

    def get_index_status(self):
        # TODO: 替换为真实数据接口
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        price   = float(info["current"])
        change  = float(info["change"])
        percent = float(info["percent"])

        # 更新滑动窗口
        self.price_window.append(price)
        self.change_window.append(change)
        self.percent_window.append(percent)

        # 窗口未满无法预测
        if len(self.price_window) < self.price_window.maxlen:
            return None

        s = pd.Series(self.price_window)
        mean_60    = s.mean()
        std_60     = s.std()
        max_60     = s.max()
        min_60     = s.min()
        momentum60 = price - self.price_window[0]
        zscore     = (price - mean_60) / (std_60 + 1e-9)

        feat = {
            "price": price,
            "change": change,
            "percent": percent,
            "mean_60": mean_60,
            "std_60": std_60,
            "max_60": max_60,
            "min_60": min_60,
            "momentum_60": momentum60,
            "zscore": zscore
        }
        return pd.DataFrame([feat])

    def sell_strategy(self):
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(
            info["dateTime"], "%Y%m%d%H%M%S%f")
        if self.last_dt and current_dt <= self.last_dt:
            return ["hold", 0]
        self.last_dt = current_dt
        if info["status"] != "T":
            return ["hold", 0]
        if self.start_dt is None:
            self.start_dt = current_dt

        elapsed = current_dt - self.start_dt
        phase   = min(int(elapsed / self.batch_duration), self.batch_count - 1)
        target_cum = int(self.total_shares * (phase+1) / self.batch_count)
        sold       = self.total_shares - self.shares_remaining
        unsold     = target_cum - sold

        feat = self.make_features(info)
        if feat is None:
            if elapsed >= self.trading_duration:
                vol = self.shares_remaining
                self.shares_remaining = 0
                return ["sell", vol]
            return ["hold", 0]

        pred_price    = self.model.predict(feat)[0]
        current_price = float(info["current"])
        pred_ret      = (pred_price - current_price) / current_price

        to_sell = 0
        if pred_ret <= 0 and unsold > 0:
            to_sell = unsold
        elif elapsed >= (phase+1)*self.batch_duration and unsold > 0:
            to_sell = unsold
        elif elapsed >= self.trading_duration and self.shares_remaining > 0:
            to_sell = self.shares_remaining

        if to_sell > 0:
            to_sell = min(to_sell, self.shares_remaining)
            self.shares_remaining -= to_sell
            logger.info(
                "%s → SELL %d shares at %.2f (pred_ret=%.4f)",
                current_dt.strftime("%H:%M:%S"),
                to_sell, current_price, pred_ret
            )
            return ["sell", to_sell]
        else:
            return ["hold", 0]

if __name__ == "__main__":
    strategy = SellStrategy()
    while strategy.shares_remaining > 0:
        action, vol = strategy.sell_strategy()
        time.sleep(strategy.request_interval)

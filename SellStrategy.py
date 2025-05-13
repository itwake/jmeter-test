# strategy_with_regression.py

import datetime
import time
import requests
import pickle
import pandas as pd
from collections import deque

class SellStrategy:
    def __init__(self,
                 total_shares=100,
                 window_size=60,      # 和训练时一致
                 batch_count=5,
                 request_interval=2,
                 model_path="hsi_price_model.pkl"):
        # 交易参数
        self.total_shares = total_shares
        self.shares_remaining = total_shares
        self.batch_count = batch_count
        self.trading_duration = datetime.timedelta(minutes=10)
        self.start_trading_dt = None
        self.last_datetime = None
        self.request_interval = request_interval

        # 分批时间
        self.batch_duration = self.trading_duration / batch_count

        # 用于重建特征的滑动窗口
        self.price_window = deque(maxlen=window_size)
        self.change_window = deque(maxlen=window_size)
        self.percent_window= deque(maxlen=window_size)
        self.high_window   = deque(maxlen=window_size)
        self.low_window    = deque(maxlen=window_size)
        self.turn_window   = deque(maxlen=window_size)

        # 加载回归模型
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def get_index_status(self):
        # TODO: 替换成真实接口
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        """基于滑动窗口 + 最新 info 构造一行特征"""
        price   = float(info["current"])
        change  = float(info["change"])
        percent = float(info["percent"])
        high_o  = float(info["high"])
        low_o   = float(info["low"])
        turn    = float(info["turnover"])

        # 更新各窗口
        self.price_window.append(price)
        self.change_window.append(change)
        self.percent_window.append(percent)
        self.high_window.append(high_o)
        self.low_window.append(low_o)
        self.turn_window.append(turn)

        # 如果窗口未满，暂无法预测
        if len(self.price_window) < self.price_window.maxlen:
            return None

        arr = {
            "price": price,
            "change": change,
            "percent": percent,
            "high_o": high_o,
            "low_o": low_o,
            "turnover": turn,
            # 滑动窗滚动特征
            "mean_60": pd.Series(self.price_window).mean(),
            "std_60":  pd.Series(self.price_window).std(),
            "max_60":  max(self.price_window),
            "min_60":  min(self.price_window),
            # 衍生比率
            "zscore":   (price - pd.Series(self.price_window).mean()) /
                        (pd.Series(self.price_window).std() + 1e-9),
            "hl_ratio": (price - min(self.price_window)) /
                        ((max(self.price_window) - min(self.price_window)) + 1e-9),
            "pos_high": (price - low_o) / ((high_o - low_o) + 1e-9),
            "vol_ratio": turn / (pd.Series(self.turn_window).mean() + 1e-9)
        }
        return pd.DataFrame([arr])

    def sell_strategy(self):
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(
            info["dateTime"], "%Y%m%d%H%M%S%f")
        # 去重
        if self.last_datetime and current_dt <= self.last_datetime:
            return ["hold", 0]
        self.last_datetime = current_dt

        # 未到 T 阶段不交易
        if info["status"] != "T":
            return ["hold", 0]

        # 记录开始
        if self.start_trading_dt is None:
            self.start_trading_dt = current_dt

        elapsed = current_dt - self.start_trading_dt
        phase = min(int(elapsed / self.batch_duration),
                    self.batch_count - 1)

        # 按均匀权重算本阶段累计目标
        target_cum = int(self.total_shares * (phase+1) / self.batch_count)
        sold       = self.total_shares - self.shares_remaining
        unsold     = target_cum - sold

        # 构造特征并预测未来价格
        feat = self.make_features(info)
        if feat is None:
            # 窗口未满，或先不卖
            if elapsed >= self.trading_duration:
                # 最后 1 分钟清仓
                vol = self.shares_remaining
                self.shares_remaining = 0
                return ["sell", vol]
            return ["hold", 0]

        pred_price = self.model.predict(feat)[0]
        current_price = float(info["current"])
        pred_ret = (pred_price - current_price) / current_price

        to_sell = 0
        # 如果模型预测未来下跌（pred_ret ≤ 0）且还有未卖份额
        if pred_ret <= 0 and unsold > 0:
            to_sell = unsold
        # 或到本阶段末尾仍未卖够
        elif elapsed >= (phase+1)*self.batch_duration and unsold > 0:
            to_sell = unsold
        # 或 10 分钟到了，清仓
        elif elapsed >= self.trading_duration and self.shares_remaining > 0:
            to_sell = self.shares_remaining

        if to_sell > 0:
            to_sell = min(to_sell, self.shares_remaining)
            self.shares_remaining -= to_sell
            return ["sell", to_sell]
        else:
            return ["hold", 0]

# 使用示例
if __name__ == "__main__":
    strategy = SellStrategy()
    while strategy.shares_remaining > 0:
        action, vol = strategy.sell_strategy()
        print(f"{strategy.last_datetime} → {action} {vol}手，剩余 {strategy.shares_remaining}手")
        time.sleep(strategy.request_interval)

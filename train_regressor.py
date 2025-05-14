strategy_multi_return.py (多次高位卖出改进)
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
        # 加载模型
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
        # 1. 去重 & 非交易
        if (self.last_dt and now <= self.last_dt) or info['status'] != 'T':
            return ['hold', 0]
        self.last_dt = now
        # 2. 初始化段与极值
        if self.start_dt is None:
            self.start_dt = now
            self.segment_start = now
            self.segment_end = now + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None
        elapsed = now - self.start_dt
        sold = self.total_shares - self.shares_remaining
        # 更新段内最值
        self.segment_max = max(self.segment_max, price)
        self.segment_min = min(self.segment_min, price)
        # 3. 基线量
        t_sec = min(elapsed, self.effective_duration).total_seconds()
        base_cum = int(self.total_shares * t_sec / self.effective_duration.total_seconds())
        unsold_base = max(base_cum - sold, 0)
        to_sell = 0
        # 4. 预测高点记录
        feat = self.make_features(info)
        if feat is not None and self.pred_high_time is None:
            preds = self.model.predict(feat)[0]
            predicted_prices = price * (1 + preds)
            idx = int(np.argmax(predicted_prices))
            offset = self.horizons_sec[idx]
            self.pred_high_time = now + datetime.timedelta(seconds=int(offset))
            logger.info("预测高点时间：%s", self.pred_high_time.strftime('%H:%M:%S'))
        # 5. 在高点处分多次卖出：
        #    每次到达预测高点前，若价格接近高点阈值则按基线卖一批
        if self.pred_high_time and now <= self.pred_high_time:
            threshold_price = 0.99 * price * (1 + preds[np.argmax(predicted_prices)])
            if price >= threshold_price and unsold_base > 0:
                to_sell = unsold_base
                logger.info("高位预卖: 卖出 %d 股", to_sell)
                # 重置，等待下次高点
                self.pred_high_time = None
        # 6. 段尾补基线
        if now > self.segment_end:
            to_sell = max(to_sell, unsold_base)
            self.segment_start = self.segment_end
            self.segment_end = self.segment_start + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None
        # 7. 缓冲结束清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining
        # 8. 执行卖出
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

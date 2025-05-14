# strategy_multi_return.py (多次高点卖出改进，修复 preds 定义)
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
        buffer_minutes=1,
        sell_multiplier=1.5     # 加速因子：基线卖出量乘以此值
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
        # 加速因子
        self.sell_multiplier = sell_multiplier
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

    def sell_strategy(self):
        # ... 前面代码不变直到 unsold_base 计算...
        # 3. 计算基线卖出量并应用加速因子
        t_sec = min(elapsed, self.effective_duration).total_seconds()
        base_cum = int(self.total_shares * t_sec / self.effective_duration.total_seconds())
        unsold_base = max(base_cum - sold, 0)
        # 加速基线卖出
        base_to_sell = int(unsold_base * self.sell_multiplier)
        to_sell = max(to_sell, base_to_sell)

        # 窗口高位加码卖出
        win_arr = np.array(self.price_window)
        if unsold_base > 0 and win_arr.size > 0:
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
            # 进入下一段
            self.segment_start = self.segment_end
            self.segment_end = self.segment_start + self.batch_duration
            self.segment_max = price
            self.segment_min = price
            self.pred_high_time = None

        # 缓冲结束清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining

        # 执行卖出
        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            sell_extra = max(to_sell - base_to_sell, 0)
            logger.info(
                "%s → SELL %d @%.2f (base*mult= %d, extra=%d)",
                now.strftime('%H:%M:%S'), vol, price, base_to_sell, sell_extra
            )
            return ['sell', vol]
        return ['hold', 0]

if __name__ == '__main__':
    strat = MultiHorizonReturnSell()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)
    strat = MultiHorizonReturnSell()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)

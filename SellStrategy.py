import datetime
import time
import requests
from collections import deque
import numpy as np

class SellStrategy:
    def __init__(self,
                 total_shares=100,
                 window_size=5,        # 滑动窗口大小
                 batch_count=5,        # 分几批
                 request_interval=2,   # 秒
                 morning_return=-0.017 # 上午涨跌幅（负数表示跌）
                 ):
        # —— 原有属性 —— 
        self.total_shares = total_shares
        self.shares_remaining = total_shares
        self.price_window = deque(maxlen=window_size)
        self.start_trading_dt = None
        self.trading_duration = datetime.timedelta(minutes=10)
        self.batch_count = batch_count
        self.batch_duration = self.trading_duration / batch_count
        self.last_datetime = None
        self.request_interval = request_interval

        # —— 新增：早盘动量 —— 
        self.morning_return = morning_return

        # —— 新增：权重分批 —— 
        # 默认权重：前两批 15%，后面均分
        w = [0.15, 0.15] + [ (1-0.30)/(batch_count-2) ]*(batch_count-2)
        self.w = np.array(w)
        # 上午大跌时加速出货
        if self.morning_return < -0.015:
            self.w[:3] *= 1.2
            self.w[3:] *= 0.8
            self.w /= self.w.sum()

    def get_index_status(self):
        # TODO: 替换成真实的 HTTP 请求
        return requests.get("https://api.example.com/index_status").json()

    def sell_strategy(self):
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(info["dateTime"], "%Y%m%d%H%M%S%f")
        # 防止重复数据
        if self.last_datetime and current_dt <= self.last_datetime:
            return ["hold", 0]
        self.last_datetime = current_dt

        # 只在 T 状态下可交易
        if info["status"] != "T":
            return ["hold", 0]

        # 记录交易开始时间
        if self.start_trading_dt is None:
            self.start_trading_dt = current_dt

        elapsed = current_dt - self.start_trading_dt
        phase = min(int(elapsed / self.batch_duration), self.batch_count - 1)

        # 根据权重算出该阶段累计应卖出的总量
        target_cumulative = int(self.total_shares * self.w[: phase+1].sum())
        already_sold = self.total_shares - self.shares_remaining
        unsold_target = target_cumulative - already_sold

        # 更新滑动窗口
        current_price = info["current"]
        self.price_window.append(current_price)

        # 如果窗口还没满，就先等
        if len(self.price_window) < self.price_window.maxlen:
            # 到结束前 1 分钟强制清仓
            if elapsed >= self.trading_duration and self.shares_remaining > 0:
                vol = self.shares_remaining
                self.shares_remaining = 0
                return ["sell", vol]
            return ["hold", 0]

        # 本地窗口最高价
        local_high = max(self.price_window)

        to_sell = 0
        # 1) 碰到窗口高点且未达目标 → 卖出未完成部分
        if current_price >= local_high and unsold_target > 0:
            to_sell = unsold_target
        # 2) 或者到了本阶段时间末尾但还没卖够 → 卖出未完成部分
        elif elapsed >= (phase + 1) * self.batch_duration and unsold_target > 0:
            to_sell = unsold_target
        # 3) 保护：超过总时长后全部清仓
        elif elapsed >= self.trading_duration and self.shares_remaining > 0:
            to_sell = self.shares_remaining

        if to_sell > 0:
            to_sell = min(to_sell, self.shares_remaining)
            self.shares_remaining -= to_sell
            return ["sell", to_sell]
        else:
            return ["hold", 0]


# 使用示例（每 2 秒调用一次）：
strategy = SellStrategy(
    total_shares=100,
    window_size=5,
    batch_count=5,
    request_interval=2,
    morning_return=-0.017
)

while strategy.shares_remaining > 0:
    action, vol = strategy.sell_strategy()
    print(f"{strategy.last_datetime} → {action} {vol}手，剩余 {strategy.shares_remaining}手")
    time.sleep(strategy.request_interval)

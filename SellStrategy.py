import datetime
import time
import requests
from collections import deque

class SellStrategy:
    def __init__(self, total_shares=100,
                 window_size=5,    # 滑动窗口大小
                 batch_count=5,    # 分几批
                 request_interval=2 # 秒
                 ):
        self.total_shares = total_shares
        self.shares_remaining = total_shares

        # 滑动高点追踪
        self.price_window = deque(maxlen=window_size)

        # 时间控制
        self.start_trading_dt = None
        self.trading_duration = datetime.timedelta(minutes=10)
        self.batch_count = batch_count
        self.batch_duration = self.trading_duration / batch_count

        # 记录上一条数据时间，防止重复
        self.last_datetime = None

    def get_index_status(self):
        # TODO: 替换成真实的 HTTP 请求
        resp = requests.get("https://api.example.com/index_status").json()
        return resp

    def sell_strategy(self):
        info = self.get_index_status()
        # 解析时间戳
        current_dt = datetime.datetime.strptime(info["dateTime"], "%Y%m%d%H%M%S%f")
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
        scheduled_sold = int((phase + 1) * self.total_shares / self.batch_count)
        already_sold = self.total_shares - self.shares_remaining
        unsold_target = scheduled_sold - already_sold

        # 更新滑动窗口
        current_price = info["current"]
        self.price_window.append(current_price)

        # —— 新增：滑动窗口未填满时，不触发任何卖出逻辑（只 hold），除非到清仓时刻 —— 
        if len(self.price_window) < self.price_window.maxlen:
            # 如果已经超过总时长，强制清仓
            if elapsed >= self.trading_duration and self.shares_remaining > 0:
                vol = self.shares_remaining
                self.shares_remaining = 0
                return ["sell", vol]
            return ["hold", 0]

        # 判断是否为窗口高点
        is_local_high = current_price >= max(self.price_window)

        to_sell = 0
        # 1) 出现局部高点且尚未完成本阶段目标，就卖出阶段剩余目标量
        if is_local_high and unsold_target > 0:
            to_sell = unsold_target
        # 2) 或者阶段已到且目标未完成，也要卖出
        elif elapsed >= (phase + 1) * self.batch_duration and unsold_target > 0:
            to_sell = unsold_target
        # 3) 冗余保护：超过总时长后全部清仓
        elif elapsed >= self.trading_duration and self.shares_remaining > 0:
            to_sell = self.shares_remaining

        # 执行卖出或持有
        if to_sell > 0:
            to_sell = min(to_sell, self.shares_remaining)
            self.shares_remaining -= to_sell
            return ["sell", to_sell]
        else:
            return ["hold", 0]


# 使用示例（每 2 秒调用一次）：
strategy = SellStrategy()
while strategy.shares_remaining > 0:
    action, vol = strategy.sell_strategy()
    print(f"{strategy.last_datetime} → {action} {vol}手，剩余 {strategy.shares_remaining}手")
    time.sleep(strategy.request_interval)

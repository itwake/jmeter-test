import datetime
import time
import requests
from collections import deque

class SellStrategy:
    def __init__(self, total_shares=100, 
                 window_size=5,    # 滑动窗口大小（价格点数）
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
        """
        TODO: 替换成真实的 HTTP 请求
        返回格式示例：
            {
                "current": float,
                "dateTime": "YYYYmmddHHMMSSfff",
                "status": "S"|"T"|"E"
            }
        """
        # 示例 stub
        resp = requests.get("https://api.example.com/index_status").json()
        return resp

    def sell_strategy(self):
        info = self.get_index_status()
        # 解析时间戳
        dt_str = info["dateTime"]
        current_dt = datetime.datetime.strptime(dt_str, "%Y%m%d%H%M%S%f")
        # 跳过重复或延迟数据
        if self.last_datetime and current_dt <= self.last_datetime:
            return ["hold", 0]
        self.last_datetime = current_dt

        status = info["status"]
        # 只有 T 时才交易
        if status != "T":
            return ["hold", 0]

        # 首次进入可交易状态，记录开始时间
        if self.start_trading_dt is None:
            self.start_trading_dt = current_dt

        # 计算已过去时长
        elapsed = current_dt - self.start_trading_dt
        # 计算当前处于第几个阶段（0 到 batch_count-1）
        phase = min(int(elapsed / self.batch_duration), self.batch_count - 1)

        # 计算到当前阶段“理想卖出量”
        scheduled_sold = int((phase + 1) * self.total_shares / self.batch_count)
        already_sold = self.total_shares - self.shares_remaining
        unsold_target = scheduled_sold - already_sold

        # 更新滑动窗口
        current_price = info["current"]
        self.price_window.append(current_price)

        # 判断是否为滑动窗口内的高点
        is_local_high = (
            len(self.price_window) == self.price_window.maxlen
            and current_price >= max(self.price_window)
        )

        to_sell = 0
        # 1) 若是滑点高点，就先卖掉该阶段剩余目标量（或最多阶段配额）
        if is_local_high and unsold_target > 0:
            to_sell = unsold_target

        # 2) 否则，如果本阶段时间已到、还有目标没完成，则强制卖出
        elif elapsed >= (phase + 1) * self.batch_duration and unsold_target > 0:
            to_sell = unsold_target

        # 3) 最后一个阶段结束前后，若仍有剩余，强清（冗余保护）
        if elapsed >= self.trading_duration and self.shares_remaining > 0:
            to_sell = self.shares_remaining

        # 执行卖出
        if to_sell > 0:
            to_sell = min(to_sell, self.shares_remaining)
            self.shares_remaining -= to_sell
            return ["sell", to_sell]
        else:
            return ["hold", 0]


# —— 使用示例 —— 
strategy = SellStrategy()
while strategy.shares_remaining > 0:
    action, vol = strategy.sell_strategy()
    print(f"{strategy.last_datetime} → {action} {vol}手，剩余 {strategy.shares_remaining}手")
    time.sleep(strategy.request_interval)

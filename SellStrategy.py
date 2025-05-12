import datetime
import time
import requests

class SellStrategy:
    def __init__(self, total_shares=100):
        self.shares_remaining = total_shares
        self.prices = []
        self.last_datetime = None             # 上次拉取的时间戳
        self.start_trading_dt = None          # 记录第一次 T 状态时的时间
        self.trading_duration = datetime.timedelta(minutes=10)

    def get_index_status(self):
        """
        TODO: 用真实接口替换这个 stub。
        返回格式示例：
            {
                "current": float,
                "dateTime": "YYYYmmddHHMMSSsss",
                "status": "S" | "T" | "E"
            }
        """
        resp = requests.get("https://api.example.com/index_status").json()
        return resp

    def decide_sell_volume(self, current_price: float) -> int:
        """
        简单“局部高点”示例：若达到历史最高价，则卖出最多 10 手。
        """
        if self.prices and current_price >= max(self.prices):
            return min(10, self.shares_remaining)
        return 0

    def sell_strategy(self):
        info = self.get_index_status()

        # 1. 解析并更新最新时间戳
        dt_str = info["dateTime"]  # e.g. "20250512221246000"
        current_dt = datetime.datetime.strptime(dt_str, "%Y%m%d%H%M%S%f")
        if self.last_datetime and current_dt <= self.last_datetime:
            return ["hold", 0]      # 重复或延迟数据，skip
        self.last_datetime = current_dt

        status = info["status"]
        if status != "T":
            # 未开盘或已结束，都不卖
            return ["hold", 0]

        # 2. 首次进入 T 状态，记录交易开始时间
        if self.start_trading_dt is None:
            self.start_trading_dt = current_dt

        # 3. 计算已交易时长，若 ≥ 10 分钟，强制清仓
        elapsed = current_dt - self.start_trading_dt
        if elapsed >= self.trading_duration and self.shares_remaining > 0:
            vol = self.shares_remaining
            self.shares_remaining = 0
            return ["sell", vol]

        # 4. 常规分批卖出逻辑
        current_price = info["current"]
        self.prices.append(current_price)
        to_sell = self.decide_sell_volume(current_price)
        if to_sell > 0:
            self.shares_remaining -= to_sell
            return ["sell", to_sell]
        else:
            return ["hold", 0]


# —— 使用示例 —— 
strategy = SellStrategy(100)
start = time.time()
while strategy.shares_remaining > 0:
    action, vol = strategy.sell_strategy()
    print(f"{strategy.last_datetime} → {action}, {vol}手，剩余 {strategy.shares_remaining}手")
    time.sleep(2)
    # 防止意外死循环：超过10分钟还未清仓，就跳出
    if time.time() - start > 10 * 60:
        print("超时，未能在10分钟内清仓！")
        break

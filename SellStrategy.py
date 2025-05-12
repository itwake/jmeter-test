import datetime
import time
import requests

class SellStrategy:
    def __init__(self, total_shares=100):
        self.shares_remaining = total_shares
        self.prices = []
        self.last_datetime = None  # 用于记录上次拉取的时间戳

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
        # 示例返回，实际请调用：requests.get(...).json()
        return {"current": 18500.0,
                "dateTime": datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3],
                "status": "T"}

    def decide_sell_volume(self, current_price: float) -> int:
        """
        简单局部高点逻辑示例：
        - 如果当前价 >= 历史最高价，则卖出最多 10 手
        - 否则不卖
        """
        if self.prices and current_price >= max(self.prices):
            return min(10, self.shares_remaining)
        return 0

    def sell_strategy(self):
        info = self.get_index_status()

        # 1. 解析并更新最新时间戳
        dt_str = info["dateTime"]  # 例如 "20250512221246000"
        current_dt = datetime.datetime.strptime(dt_str, "%Y%m%d%H%M%S%f")
        # 如果新拉取的时间不比上次晚，视为重复或延迟数据，直接 hold
        if self.last_datetime and current_dt <= self.last_datetime:
            return ["hold", 0]
        self.last_datetime = current_dt

        status = info["status"]
        # 2. 根据状态决定只能在 T 时卖出
        if status == "S":
            return ["hold", 0]
        if status == "E":
            # 交易已结束，无法再卖
            return ["hold", 0]

        # 3. status == "T" 时的卖出逻辑
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
    # 每2秒拉一次
    time.sleep(2)
    # 如果不想无限等，也可以加个超时判断：
    if time.time() - start > 10 * 60:
        print("超时，交易结束前清仓失败！")
        break

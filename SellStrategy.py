import time
import datetime
import random
import requests

class SellStrategy:
    def __init__(self,
                 m: int = 100,
                 n: int = 10,
                 t_minutes: int = 10,
                 p: float = 1.2,
                 request_interval: float = 2.0):
        # 参数初始化
        self.total_shares = m
        self.shares_remaining = m
        self.n = n
        self.p = p
        self.request_interval = request_interval

        # 时间控制
        self.start_trading_dt: datetime.datetime = None
        self.last_datetime: datetime.datetime = None
        self.total_duration = datetime.timedelta(minutes=t_minutes)
        self.segment_duration = self.total_duration / n
        # 在最后一分钟之前强制清仓
        self.final_cleanup_threshold = self.total_duration - datetime.timedelta(minutes=1)

        # 分段测量和状态
        self.measurement_data = {i: [] for i in range(n)}
        self.sold_segments = set()

    def get_index_status(self) -> dict:
        """
        TODO: 用真实接口替换此处 stub。
        期望返回格式：
        {
            "current": float,             # 当前价格
            "dateTime": "YYYYmmddHHMMSSfff",  # 服务器时间戳
            "status": "S"|"T"|"E"
        }
        """
        # —— 示例 stub：随机波动模拟 —— 
        now = datetime.datetime.now()
        price = 18500 + random.uniform(-20, 20)
        return {
            "current": price,
            "dateTime": now.strftime("%Y%m%d%H%M%S%f")[:-3],
            "status": "T"  # 始终视为可交易，用真实接口会返回 S/T/E
        }

    def sell(self, volume: int):
        """
        TODO: 用真实卖出接口替换此处。
        """
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} 卖出 {volume} 手")

    def sell_strategy(self):
        info = self.get_index_status()
        status = info["status"]
        dt_str = info["dateTime"]
        current_price = info["current"]

        # 解析服务器时间戳
        current_dt = datetime.datetime.strptime(dt_str, "%Y%m%d%H%M%S%f")

        # 跳过重复或延迟数据
        if self.last_datetime and current_dt <= self.last_datetime:
            return ["hold", 0]
        self.last_datetime = current_dt

        # 仅在交易状态可卖
        if status != "T":
            return ["hold", 0]

        # 首次遇到 T 时，记录交易开始时间
        if self.start_trading_dt is None:
            self.start_trading_dt = current_dt

        elapsed = current_dt - self.start_trading_dt

        # —— 最后 1 分钟前的终极清仓 —— 
        if elapsed >= self.final_cleanup_threshold and self.shares_remaining > 0:
            to_sell = self.shares_remaining
            self.shares_remaining = 0
            return ["sell", to_sell]

        # 计算当前属于第几段（0 到 n-1）
        segment_index = int(elapsed / self.segment_duration)
        if segment_index >= self.n:
            segment_index = self.n - 1

        seg_start = self.start_trading_dt + segment_index * self.segment_duration
        seg_mid   = seg_start + (self.segment_duration / 2)
        seg_end   = seg_start + self.segment_duration

        # —— 测量阶段（前半段）：收集价格 —— 
        if current_dt <= seg_mid:
            self.measurement_data[segment_index].append(current_price)
            return ["hold", 0]

        # —— 决策阶段（后半段）—— 
        # 还没对本段出手过
        if segment_index not in self.sold_segments:
            prices = self.measurement_data[segment_index]
            if prices:
                avg_price = sum(prices) / len(prices)
                # 1) 如果价格达到阈值，立即卖出本段量
                if current_price >= avg_price * self.p:
                    z = self.total_shares // self.n
                    to_sell = min(z, self.shares_remaining)
                    self.shares_remaining -= to_sell
                    self.sold_segments.add(segment_index)
                    return ["sell", to_sell]
            # 2) 否则如果已过本段末尾，还未卖，则强制卖出
            if current_dt >= seg_end:
                z = self.total_shares // self.n
                to_sell = min(z, self.shares_remaining)
                self.shares_remaining -= to_sell
                self.sold_segments.add(segment_index)
                return ["sell", to_sell]

        # 默认不卖
        return ["hold", 0]


if __name__ == "__main__":
    # 默认参数：m=100, n=10, t=10min, p=1.2 (即 20% 阈值), request_interval=2 秒
    strategy = SellStrategy()
    while strategy.shares_remaining > 0:
        action, vol = strategy.sell_strategy()
        print(f"{strategy.last_datetime} → {action} {vol}手，剩余 {strategy.shares_remaining}手")
        time.sleep(strategy.request_interval)

import time
import datetime
import requests

def get_index_status():
    """
    TODO: 用真实接口替换此处 stub。
    应返回字典，格式例如：
        {
            "current": float,            # 当前价格
            "dateTime": "YYYYmmddHHMMSSfff",  # 时间戳
            "status": "S"|"T"|"E"        # S=开盘前, T=交易中, E=交易后
        }
    """
    # 示例 stub（请替换为真实请求）
    # resp = requests.get("https://api.example.com/index_status").json()
    # return resp
    return {
        "current": 18500.0,
        "dateTime": datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3],
        "status": "T"
    }

def sell(volume):
    """
    TODO: 用真实卖出接口替换此处。
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} 卖出 {volume} 手")

def sell_strategy(n, m, t_minutes, p, request_interval=2):
    """
    分 n 批卖出 m 手，总时长 t_minutes 分钟。
    每批基础卖出量 z = m // n，剩余的放在最后一批一起卖。
    每段时长 d = t_minutes*60 / n 秒。
    在每段的前半段读取数据，算出平均价 a；
    在后半段检测实时价 ≥ a * p 时，卖出 z。
    在总时长结束前 1 分钟，强制清仓剩余所有手数。
    """
    shares_remaining = m
    z = m // n
    total_sec = t_minutes * 60
    segment_sec = total_sec / n
    start_ts = time.time()

    for i in range(n):
        seg_start = start_ts + i * segment_sec
        measure_end = seg_start + (segment_sec / 2)
        decision_end = seg_start + segment_sec

        # 前半段：采集价格，计算平均价
        prices = []
        while time.time() < measure_end:
            info = get_index_status()
            if info["status"] == "T":
                prices.append(info["current"])
            time.sleep(request_interval)
        avg_price = sum(prices) / len(prices) if prices else None

        # 后半段：达到阈值则卖出 z
        while time.time() < decision_end and shares_remaining > 0 and avg_price is not None:
            info = get_index_status()
            if info["status"] != "T":
                time.sleep(request_interval)
                continue
            current_price = info["current"]
            if current_price >= avg_price * p:
                sell(z)
                shares_remaining -= z
                break
            time.sleep(request_interval)

    # 在总时长结束前 1 分钟，强制清仓剩余手数
    final_deadline = start_ts + total_sec - 60
    while time.time() < final_deadline:
        time.sleep(request_interval)
    if shares_remaining > 0:
        sell(shares_remaining)
        shares_remaining = 0

if __name__ == "__main__":
    # 初始化参数
    m = 100           # 总份额
    n = 10            # 分批次数
    t_minutes = 10    # 总时长（分钟）
    p = 1.20          # 阈值系数（平均价的120%即20%涨幅）
    sell_strategy(n=n, m=m, t_minutes=t_minutes, p=p, request_interval=2)

import logging
import datetime
import time
import requests
import pickle
import pandas as pd
import numpy as np
from collections import deque

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class SellStrategy:
    def __init__(
        self,
        total_shares=100,
        window_size=2*60*60,    # 最大 2h 窗口
        batch_count=5,
        request_interval=2,
        model_path="hsi_price_model.pkl",
        buffer_minutes=1        # 提前 buffer 分钟清仓
    ):
        # 卖出总量与分批
        self.total_shares     = total_shares
        self.shares_remaining = total_shares
        self.batch_count      = batch_count

        # 交易时长与缓冲
        self.trading_duration  = datetime.timedelta(minutes=10)
        self.clear_buffer      = datetime.timedelta(minutes=buffer_minutes)
        # 实际用于分批的有效时长
        self.effective_duration = self.trading_duration - self.clear_buffer

        # 分批时长（基于有效时长）
        self.batch_duration   = self.effective_duration / batch_count

        # 时间管理
        self.start_dt         = None
        self.last_dt          = None
        self.request_interval = request_interval

        # 滑动窗口用于多周期特征
        self.price_window     = deque(maxlen=window_size)

        # 加载回归模型
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Loaded regression model from %s", model_path)

    def get_index_status(self):
        # TODO: 替换为真实行情接口
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        # 提取原始变量
        price   = float(info["current"])
        change  = float(info["change"])
        percent = float(info["percent"])

        # 更新窗口
        self.price_window.append(price)
        if len(self.price_window) < self.price_window.maxlen:
            return None

        ps = np.array(self.price_window)
        # 周期定义（秒）
        windows = {
            "1m":  60,
            "30m": 30*60,
            "1h":  60*60,
            "2h":  2*60*60
        }

        feats = {"price": price, "change": change, "percent": percent}
        for name, w in windows.items():
            vals = ps[-w:]
            feats[f"mean_{name}"]      = vals.mean()
            feats[f"std_{name}"]       = vals.std()
            feats[f"price_{name}_ago"] = ps[-w]

        return pd.DataFrame([feats])

    def sell_strategy(self):
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(
            info["dateTime"], "%Y%m%d%H%M%S%f")

        # 防止重复
        if self.last_dt and current_dt <= self.last_dt:
            return ["hold", 0]
        self.last_dt = current_dt

        # 非交易阶段
        if info["status"] != "T":
            return ["hold", 0]

        # 记录开始
        if self.start_dt is None:
            self.start_dt = current_dt

        elapsed = current_dt - self.start_dt

        # 计算已卖量与阶段目标
        sold       = self.total_shares - self.shares_remaining

        # 1. 基线：按时间线性分配至有效时长
        t_sec       = min(elapsed, self.effective_duration).total_seconds()
        eff_sec     = self.effective_duration.total_seconds()
        req_cum     = int(self.total_shares * t_sec / eff_sec)
        unsold_req  = max(req_cum - sold, 0)

        # 2. 构造特征并预测
        feat = self.make_features(info)
        if feat is None:
            # 在 buffer 时间段提前清仓：
            if elapsed >= self.effective_duration:
                vol = self.shares_remaining
                self.shares_remaining = 0
                logger.info("%s → FINAL CLEAR %d shares at %.2f", 
                            current_dt.strftime("%H:%M:%S"), vol, float(info["current"]))
                return ["sell", vol]
            # 基线卖出
            if unsold_req > 0:
                vol = min(unsold_req, self.shares_remaining)
                self.shares_remaining -= vol
                logger.info("%s → BASELINE SELL %d shares at %.2f", 
                            current_dt.strftime("%H:%M:%S"), vol, float(info["current"]))
                return ["sell", vol]
            return ["hold", 0]

        pred_price    = self.model.predict(feat)[0]
        current_price = float(info["current"])
        pred_ret      = (pred_price - current_price) / current_price

        to_sell = unsold_req  # 至少满足基线
        # 若预测未来回撤，卖出所有阶段剩余
        if pred_ret <= 0:
            # 按阶段逻辑：阶段大小 = total_shares / batch_count
            phase_idx = min(int(elapsed / self.batch_duration), self.batch_count - 1)
            phase_target = int(self.total_shares * (phase_idx + 1) / self.batch_count)
            unsold_phase = max(phase_target - sold, 0)
            to_sell = max(to_sell, unsold_phase)

        # 清仓：若已到 buffer 期限
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining

        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            logger.info(
                "%s → SELL %d shares at %.2f (pred_ret=%.4f, baseline=%d)",
                current_dt.strftime("%H:%M:%S"),
                vol, current_price, pred_ret, unsold_req
            )
            return ["sell", vol]
        return ["hold", 0]

if __name__ == "__main__":
    strategy = SellStrategy()
    while strategy.shares_remaining > 0:
        action, vol = strategy.sell_strategy()
        time.sleep(strategy.request_interval)

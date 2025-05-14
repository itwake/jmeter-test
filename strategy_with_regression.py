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
        window_size=30,         # 滑动窗口长度（秒），用于特征重建，最大30秒
        batch_count=5,
        request_interval=2,
        model_path="hsi_price_model.pkl",
        buffer_minutes=1        # 提前结束交易的缓冲时间（分钟）
    ):
        """
        total_shares:       总卖出份额
        window_size:        滑动窗口长度（秒），用于特征重建
        batch_count:        分几批卖出
        request_interval:   获取行情数据的间隔（秒）
        model_path:         回归模型文件路径
        buffer_minutes:     提前结束交易的缓冲时间（分钟）
        """
        # 基本参数
        self.total_shares     = total_shares
        self.shares_remaining = total_shares
        self.batch_count      = batch_count

        # 时间参数
        self.trading_duration   = datetime.timedelta(minutes=10)
        self.clear_buffer       = datetime.timedelta(minutes=buffer_minutes)
        self.effective_duration = self.trading_duration - self.clear_buffer
        self.batch_duration     = self.effective_duration / batch_count
        self.request_interval   = request_interval

        # 时间跟踪
        self.start_dt = None
        self.last_dt  = None

        # 滑动窗口，用于特征（单位：数据点数）
        self.price_window = deque(maxlen=window_size)

        # 加载模型
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Loaded model from %s", model_path)

    def get_index_status(self):
        """
        获取最新行情，返回包含 dateTime, status, current, change, percent
        """
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        """
        构造特征：当前值，以及最近10s/20s/30s的均值、标准差和相应的滞后价格
        """
        price   = float(info["current"])
        change  = float(info["change"])
        percent = float(info["percent"])

        # 更新窗口
        self.price_window.append(price)
        if len(self.price_window) < self.price_window.maxlen:
            return None

        arr = np.array(self.price_window)
        # 短周期窗口长度（数据点数，对应秒数）
        windows = {"10s": 10, "20s": 20, "30s": 30}

        feats = {"price": price, "change": change, "percent": percent}
        for name, w in windows.items():
            vals = arr[-w:]
            feats[f"mean_{name}"]      = float(vals.mean())
            feats[f"std_{name}"]       = float(vals.std())
            feats[f"price_{name}_ago"] = float(arr[-w])

        return pd.DataFrame([feats])

    def sell_strategy(self):
        """
        卖出策略：获取行情 -> 特征 -> 预测 -> 卖出决策
        返回 ["sell", vol] 或 ["hold", 0]
        """
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(
            info["dateTime"], "%Y%m%d%H%M%S%f")

        # 去重
        if self.last_dt and current_dt <= self.last_dt:
            return ["hold", 0]
        self.last_dt = current_dt

        # 仅在交易开放期
        if info["status"] != "T":
            return ["hold", 0]

        # 记录开始时间
        if self.start_dt is None:
            self.start_dt = current_dt

        elapsed = current_dt - self.start_dt
        current_price = float(info["current"])
        sold = self.total_shares - self.shares_remaining

        # 基线卖出量：按时间进度线性分配
        t_sec = min(elapsed, self.effective_duration).total_seconds()
        eff_sec = self.effective_duration.total_seconds()
        req_cum = int(self.total_shares * t_sec / eff_sec)
        unsold_req = max(req_cum - sold, 0)

        # 构造特征
        feat = self.make_features(info)
        if feat is None:
            # 到缓冲结束 -> 清仓
            if elapsed >= self.effective_duration:
                vol = self.shares_remaining
                self.shares_remaining = 0
                logger.info(
                    "%s → FINAL CLEAR %d at %.2f",
                    current_dt.strftime("%H:%M:%S"), vol, current_price
                )
                return ["sell", vol]
            # 基线卖出
            if unsold_req > 0:
                vol = min(unsold_req, self.shares_remaining)
                self.shares_remaining -= vol
                logger.info(
                    "%s → BASELINE SELL %d at %.2f",
                    current_dt.strftime("%H:%M:%S"), vol, current_price
                )
                return ["sell", vol]
            return ["hold", 0]

        # 预测回归
        pred_price = self.model.predict(feat)[0]
        pred_ret = (pred_price - current_price) / current_price

        # 决策：基线 + 预测触发
        to_sell = unsold_req
        if pred_ret <= 0:
            phase_idx = min(int(elapsed / self.batch_duration), self.batch_count - 1)
            phase_target = int(self.total_shares * (phase_idx + 1) / self.batch_count)
            unsold_ph = max(phase_target - sold, 0)
            to_sell = max(to_sell, unsold_ph)

        # 到缓冲结束 -> 清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining

        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            logger.info(
                "%s → SELL %d at %.2f (pred_ret=%.4f, base=%d)",
                current_dt.strftime("%H:%M:%S"), vol, current_price, pred_ret, unsold_req
            )
            return ["sell", vol]
        return ["hold", 0]

if __name__ == "__main__":
    strat = SellStrategy()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)

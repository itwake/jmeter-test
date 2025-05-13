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
        window_size=2*60,        # 滑动窗口长度（秒），此处为2分钟
        batch_count=5,
        request_interval=2,
        model_path="hsi_price_model.pkl",
        buffer_minutes=1        # 提前清仓缓冲时间（分钟）
    ):
        # 卖出参数
        self.total_shares     = total_shares
        self.shares_remaining = total_shares
        self.batch_count      = batch_count

        # 交易时长与清仓缓冲
        self.trading_duration   = datetime.timedelta(minutes=10)
        self.clear_buffer       = datetime.timedelta(minutes=buffer_minutes)
        self.effective_duration = self.trading_duration - self.clear_buffer
        self.batch_duration     = self.effective_duration / batch_count

        # 时间管理
        self.start_dt         = None
        self.last_dt          = None
        self.request_interval = request_interval

        # 滑动窗口，用于构造特征（单位：采样点）
        self.price_window = deque(maxlen=window_size)

        # 加载回归模型
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Loaded regression model from %s", model_path)

    def get_index_status(self):
        """
        获取最新行情数据，返回 dict 包含:
          - dateTime: "YYYYMMDDhhmmssfff"
          - status:   "T","S","E"
          - current, change, percent
        """
        # TODO: 替换为真实行情接口
        return requests.get("https://api.example.com/index_status").json()

    def make_features(self, info):
        """
        构造回归模型所需特征：
          price, change, percent,
          最近1m/2m的均值、标准差和价格滞后
        """
        # 原始值
        price   = float(info["current"])
        change  = float(info["change"])
        percent = float(info["percent"])

        # 更新窗口
        self.price_window.append(price)
        if len(self.price_window) < self.price_window.maxlen:
            return None

        ps = np.array(self.price_window)
        windows = {"1m": 60, "2m": 120}

        feats = {"price": price, "change": change, "percent": percent}
        for name, w in windows.items():
            # 计算最近w个采样点（等于秒数）
            vals = ps[-w:]
            feats[f"mean_{name}"]      = vals.mean()
            feats[f"std_{name}"]       = vals.std()
            feats[f"price_{name}_ago"] = ps[-w]

        return pd.DataFrame([feats])

    def sell_strategy(self):
        """
        主策略：获取行情 -> 构造特征 -> 预测 -> 卖出决策
        返回 ["sell", volume] 或 ["hold", 0]
        """
        info = self.get_index_status()
        current_dt = datetime.datetime.strptime(
            info["dateTime"], "%Y%m%d%H%M%S%f")

        # 去重
        if self.last_dt and current_dt <= self.last_dt:
            return ["hold", 0]
        self.last_dt = current_dt

        # 非交易时段
        if info["status"] != "T":
            return ["hold", 0]

        # 记录开始时间
        if self.start_dt is None:
            self.start_dt = current_dt

        elapsed = current_dt - self.start_dt
        # 当前价格
        current_price = float(info["current"])
        # 已卖出量
        sold = self.total_shares - self.shares_remaining

        # 基线卖出量：按时间进度线性分配
        t_sec      = min(elapsed, self.effective_duration).total_seconds()
        eff_sec    = self.effective_duration.total_seconds()
        req_cum    = int(self.total_shares * t_sec / eff_sec)
        unsold_req = max(req_cum - sold, 0)

        # 构造特征
        feat = self.make_features(info)
        # 若特征未准备好
        if feat is None:
            # 若到达清仓缓冲期，则全部清仓
            if elapsed >= self.effective_duration:
                vol = self.shares_remaining
                self.shares_remaining = 0
                logger.info(
                    "%s → FINAL CLEAR %d shares at %.2f",
                    current_dt.strftime("%H:%M:%S"), vol, current_price
                )
                return ["sell", vol]
            # 基线卖出
            if unsold_req > 0:
                vol = min(unsold_req, self.shares_remaining)
                self.shares_remaining -= vol
                logger.info(
                    "%s → BASELINE SELL %d shares at %.2f",
                    current_dt.strftime("%H:%M:%S"), vol, current_price
                )
                return ["sell", vol]
            return ["hold", 0]

        # 回归预测未来价格
        pred_price = self.model.predict(feat)[0]
        pred_ret   = (pred_price - current_price) / current_price

        # 决策逻辑
        to_sell = unsold_req
        # 若预测未来不会上涨，则保证卖出至少本阶段份额
        if pred_ret <= 0:
            phase_idx    = min(int(elapsed / self.batch_duration), self.batch_count - 1)
            phase_target = int(self.total_shares * (phase_idx + 1) / self.batch_count)
            unsold_ph    = max(phase_target - sold, 0)
            to_sell      = max(to_sell, unsold_ph)

        # 到达清仓缓冲期，也全部清仓
        if elapsed >= self.effective_duration:
            to_sell = self.shares_remaining

        if to_sell > 0:
            vol = min(to_sell, self.shares_remaining)
            self.shares_remaining -= vol
            logger.info(
                "%s → SELL %d shares at %.2f (pred_ret=%.4f, baseline=%d)",
                current_dt.strftime("%H:%M:%S"), vol, current_price, pred_ret, unsold_req
            )
            return ["sell", vol]

        return ["hold", 0]

if __name__ == "__main__":
    strat = SellStrategy()
    while strat.shares_remaining > 0:
        strat.sell_strategy()
        time.sleep(strat.request_interval)


# FX交易量预测模型 POC 技术总结

## 1. 建模与机器学习技术

### ✅ LightGBM 模型
- 用于预测每小时交易量。
- 优势：高效、支持特征重要性分析、适合结构化数据。

---

## 2. 特征工程（Feature Engineering）

### 🔹 滞后特征（Lag Features）
- 表示过去1~5小时的交易量，用于捕捉短期趋势。

### 🔹 滚动统计特征（Rolling Features）
- 近期交易量的均值与标准差，用于识别高峰或异常。

### 🔹 时间特征（Time Features）
- 如小时、星期几、月份等，捕捉周期性交易模式。

### 🔹 周期性编码特征（Fourier Features）
- 用正余弦函数编码时间变量，处理非线性时间关系。

### 🔹 节假日特征
- 标注香港和日本的节假日，识别假日前后交易波动。

---

## 3. 模型评估指标

- **MSE**：均方误差，衡量异常偏差。
- **MAE**：平均绝对误差，反映整体偏差水平。
- **WMAPE**：加权平均绝对误差，主评估指标。
  - 买入方向 WMAPE ≈ 9.83%
  - 卖出方向 WMAPE ≈ 17.48%

---

## 4. 时间序列稳定性检验

### ✅ ADF Test（单位根检验）
- ADF 值 = -13.13，p-value ≈ 1.47e-24
- 结论：交易量时间序列是平稳的，适合建模。

---

## 5. 高峰识别与加点策略模拟

### ✅ 高峰识别规则
- 若预测交易量 > 过去90天该小时中位数 × N，则认为是交易高峰。

### ✅ 加点模拟
- 假设高峰时段加点提高 0.05%，客户无反应变化。
- 模拟显示可带来潜在收益。

---

## 6. A/B 测试设计
- 实验组使用模型预测指导加点；
- 对照组按人工经验操作；
- 比较实际交易量和利润差异。

---

## 7. 特征重要性分析
- 使用 LightGBM 的特征重要性评分；
- Rolling 均值和滞后交易量是最关键特征。

---

## 8. 技术流程图

```
数据收集 → 特征工程 → 模型训练 → 模型评估
                ↓
     高峰识别 + 加点模拟 + 收益验证
```

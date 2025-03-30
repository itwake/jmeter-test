# 🌲 机器学习知识树（Machine Learning Knowledge Tree）

## 🧠 机器学习（Machine Learning）

机器学习是人工智能的一个分支，它使计算机通过数据学习模式。其主要分支如下：

---

## 1. 监督学习（Supervised Learning）
### 📌 分类（Classification）
- 逻辑回归（Logistic Regression）
- 决策树 / 随机森林（Decision Tree / Random Forest）
- 支持向量机（SVM）
- 神经网络（MLP）

### 📌 回归（Regression）
- 线性回归（Linear Regression）
- 岭回归 / 套索回归（Ridge / Lasso）
- 梯度提升树（XGBoost, LightGBM）

**📍 应用：** 房价预测、疾病诊断、信用评分

---

## 2. 无监督学习（Unsupervised Learning）
### 📌 聚类（Clustering）
- K-Means
- DBSCAN
- 层次聚类

### 📌 降维（Dimensionality Reduction）
- PCA（主成分分析）
- t-SNE / UMAP

**📍 应用：** 客户分群、异常检测、数据可视化

---

## 3. 强化学习（Reinforcement Learning）
- Q-Learning
- 深度强化学习（DQN, PPO, A3C）

**📍 应用：** 游戏 AI（AlphaGo）、自动驾驶、机器人控制

---

## 4. 深度学习（Deep Learning）
- 卷积神经网络（CNN） → 图像处理
- 循环神经网络（RNN, LSTM, GRU）→ 序列数据
- Transformer → 大语言模型的基础
- GAN（生成对抗网络）→ 图像/视频生成

**📍 应用：** 图像识别、语音识别、文本生成、语言理解

---

## 5. 生成式学习（Generative Learning）
- 自编码器（AutoEncoder）
- 变分自编码器（VAE）
- GAN（生成对抗网络）
- 大语言模型（LLM：GPT、BERT、T5等）

**📍 应用：** 图像生成、文本生成、音乐生成、视频生成

---

## 6. 半监督 / 自监督学习（Semi / Self-Supervised Learning）
- 伪标签
- 对比学习（SimCLR、BYOL）

**📍 应用：** 大模型预训练（特别是在 LLM 和视觉大模型中）

---

## 🌐 应用场景一览

| 领域         | 使用方法                 | 代表系统           |
|--------------|--------------------------|--------------------|
| 图像识别     | CNN, ResNet              | 图像分类、人脸识别 |
| 自然语言处理 | RNN, Transformer, LLM    | ChatGPT, 翻译      |
| 推荐系统     | 协同过滤、深度排序模型   | 抖音、淘宝推荐     |
| 金融风控     | 决策树、XGBoost          | 信用评分、欺诈识别 |
| 自动驾驶     | 深度强化学习、视觉模型   | Tesla Autopilot    |
| 生物医药     | 表达谱分析、蛋白质预测   | AlphaFold、药物筛选|
| 生成式 AI    | GAN、VAE、LLM            | Midjourney、Suno   |

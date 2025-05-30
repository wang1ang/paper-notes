# OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment

**链接**：[arXiv 2502.18965](https://arxiv.org/abs/2502.18965)

---

## 🔑 核心做法

### 1. 生成式推荐

将推荐建模为序列生成任务：给定用户历史行为序列，模型直接生成一个 item 序列作为推荐结果，统一召回和排序流程。  
item 被离散化为 token，token 空间通过 **Balanced K-means clustering** 从 embedding 中构建。

---

### 2. 模型训练流程

- **预训练**：在高质量 session 数据上，用 next-item prediction 目标训练生成模型（Transformer decoder）。
- **偏好对齐（Preference Alignment）**：
  - 单独训练一个 **reward model**，预测以下四个目标的期望值：
    - Session watch time
    - View probability
    - Follow probability
    - Like probability
  - 用该模型对候选推荐序列打分，作为后续强化优化信号。

---

### 3. DPO 优化方法

- 使用预训练模型 + beam search 生成多个推荐序列
- 用 reward model 打分，挑选得分高/低的正负样本 \( y^+, y^- \)
- 优化 decoder，使其更偏好高评分序列：

$$
\mathcal{L}_{\text{DPO}} = - \log \frac{ \exp( r(y^+) / \beta ) }{ \exp( r(y^+) / \beta ) + \exp( r(y^-) / \beta ) }
$$

其中 \( r(\cdot) \) 为 reward，\( \beta \) 为温度系数。

---

## 📈 效果

- OneRec 在无需两阶段召回排序的情况下达到更优推荐性能
- Preference Alignment 提高了推荐结果在满意度、多样性、相关性上的平衡性

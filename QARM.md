# QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou

**链接**：[arXiv 2411.11739](https://arxiv.org/abs/2411.11739)

---

## 🔑 核心方法

### 1. Item Alignment  
构造高质量 item-item 对，进行对比学习，微调预训练多模态模型，使其生成的表示更贴近用户行为数据，解决预训练目标与推荐任务不一致的问题。

对比损失函数如下：

$$
\mathcal{L}_{\text{align}} = -\log \frac{\exp\left( \text{sim}(z_i, z_j) / \tau \right)}{\sum_{k=1}^{N} \exp\left( \text{sim}(z_i, z_k) / \tau \right)}
$$

其中：
- $z_i, z_j$ 是正样本对的表示
- $\text{sim}$ 是余弦相似度
- $\tau$ 是温度参数

---

### 2. Quantitative Code（VQ & RQ）  
将多模态 embedding 离散化为推荐模型可用的 code ID，实现端到端训练。

#### VQ（Vector Quantization）  
使用 KNN 找到 top-K 最近邻：

$$
q_{\text{top-K}}(x) = \{ e_{k_1}, e_{k_2}, \dots, e_{k_K} \}
$$

多个 code 能保留更多语义信息，避免信息丢失。

#### RQ（Residual Quantization）  
残差递归量化，多层级编码：

$$
r_1 = x - e^{(1)}_{k_1}, \quad r_2 = r_1 - e^{(2)}_{k_2}, \quad \dots
$$

最终编码重构为：

$$
\hat{x} = e^{(1)}_{k_1} + e^{(2)}_{k_2} + \dots + e^{(L)}_{k_L}
$$

---

## 📈 实验收益（快手线上）

- 广告冷启动：**收入提升 +9.7%**
- 电商场景：**GMV 提升 +2.3%**

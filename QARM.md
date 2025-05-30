# QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou

**链接**：[arXiv 2411.11739](https://arxiv.org/abs/2411.11739)

---

## 🔑 核心方法

### 1. Item Alignment  
构造高质量 item-item 对，进行对比学习，微调预训练多模态模型，使其生成的表示更贴近用户行为数据，解决预训练目标与推荐任务不一致的问题。

对比损失函数如下：

$$
L_{align} = -\log \left[ \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_k \exp(sim(z_i, z_k)/\tau)} \right]
$$

其中：
- $z_i, z_j$ 是正样本对的表示  
- $sim$ 是余弦相似度  
- $\tau$ 是温度参数

---

### 2. Quantitative Code（VQ & RQ）  
将多模态 embedding 离散化为推荐模型可用的 code ID，实现端到端训练。

#### VQ（Vector Quantization）  
使用 KNN 找到 top-K 最近邻：

$$
q_{topK}(x) = \{ e_{k1}, e_{k2}, ..., e_{kK} \}
$$

多个 code 能保留更多语义信息，避免信息丢失。

#### RQ（Residual Quantization）  
残差递归量化，多层级编码：

$$
r_1 = x - e^{1}_{k1}
$$

$$
r_2 = r_1 - e^{2}_{k2}
$$

最终编码重构为：

$$
\hat{x} = \hat{x} + \dots + e^{L}_{kL}
$$

---

### 3. QARM 的使用：构造 item / user 特征

- **Item 特征**：从视频文本、封面图像、视频帧等多模态信息中提取表示，经 Item Alignment 微调后，通过 VQ/RQ 编码为若干个离散 ID，作为推荐系统输入特征。
- **User 特征**：聚合用户近期行为（如点击、观看过的 item 的离散 code ID），构造序列特征，喂给下游推荐模型（如 Transformer）进行建模。
- 这些离散特征与其它结构化特征（如 age、gender、category 等）拼接，形成完整的输入。

---

## 📈 实验收益（快手线上）

- 广告冷启动：**收入提升 +9.7%**
- 电商场景：**GMV 提升 +2.3%**

### 1. BERT 文本分类与实体识别的关系及 Loss 函数

**关系：**
BERT（Bidirectional Encoder Representations from Transformers）是一个预训练的语言模型底座，**文本分类**和**实体识别（NER）**都是基于 BERT 进行微调（Fine-tuning）的下游任务。它们共享相同的底层编码器（Encoder）结构来提取上下文语义表示，但在输出层和处理粒度上有所不同：

*   **文本分类 (Text Classification)：**
    *   **粒度：** 句子/文档级别（Sequence Level）。
    *   **机制：** 通常取 BERT 输出序列中第一个特殊 token `[CLS]` 的向量表示，将其输入到一个全连接层（Linear Layer）+ Softmax，预测整个句子的类别。
    *   **Loss 函数：** **交叉熵损失 (Cross-Entropy Loss)**。
        $$ L_{cls} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) $$
        其中 $C$ 是类别数，$y$ 是真实标签，$\hat{y}$ 是预测概率。

*   **实体识别 (Named Entity Recognition, NER)：**
    *   **粒度：** Token 级别（Token Level）。
    *   **机制：** 对输入序列中的**每一个** token（或子词）的输出向量都接一个全连接层 + Softmax，预测该 token 的实体标签（如 B-PER, I-PER, O 等）。
    *   **Loss 函数：** **Token 级别的交叉熵损失 (Token-level Cross-Entropy Loss)**。
        $$ L_{ner} = -\frac{1}{N} \sum_{j=1}^{N} \sum_{k=1}^{K} y_{j,k} \log(\hat{y}_{j,k}) $$
        其中 $N$ 是序列长度，$K$ 是标签类别数。通常会对填充位（Padding）的 loss 进行掩码（mask），不计入总 loss。

---

### 2. 多任务训练 `loss = seq_loss + token_loss` 的坏处

直接将两个任务的 Loss 简单相加（即权重系数 $\lambda_1 = \lambda_2 = 1$）存在以下主要问题：

1.  **量级不平衡 (Magnitude Imbalance)：**
    *   `seq_loss` 是针对整个样本计算一次，数值较小。
    *   `token_loss` 是针对序列中每个 token 计算并求平均（或求和），如果序列较长或负样本（O 标签）极多，其数值量级可能远大于 `seq_loss`。
    *   **后果：** 优化器会主要关注数值大的那个任务（通常是 NER），导致另一个任务（文本分类）几乎学不到东西，梯度被主导。

2.  **收敛速度不一致 (Convergence Rate Mismatch)：**
    *   不同任务的难易程度不同。简单的任务可能很快收敛（Loss 变得很小），而困难的任务 Loss 依然很大。
    *   **后果：** 如果简单任务的 Loss 已经很小但权重不变，它对梯度的贡献就会消失；反之，如果困难任务一直不收敛，巨大的梯度可能会破坏已经学好任务的参数表示（负迁移）。

3.  **梯度冲突 (Gradient Conflict)：**
    *   两个任务可能对共享参数（BERT Encoder）提出相反的更新方向。
    *   **后果：** 简单相加无法解决方向冲突，可能导致模型在两个任务上都表现不佳，震荡不收敛。

4.  **单位/物理意义不同：**
    *   一个是句子级概率分布的差异，一个是 Token 级标签分布的差异，直接相加缺乏理论上的尺度统一性。

---

### 3. 如何处理训练不平衡情况？

针对上述问题，学术界和工业界有多种成熟的解决方案，从简单的启发式方法到动态调整算法：

#### A. 静态加权 (Static Weighting) - 最简单
手动设置超参数 $\lambda_1, \lambda_2$：
$$ L_{total} = \lambda_1 L_{seq} + \lambda_2 L_{token} $$
*   **方法：** 通过网格搜索（Grid Search）或经验法则（如根据验证集表现调整）来确定比例。
*   **缺点：** 调参成本高，且无法适应训练过程中动态变化的需求。

#### B. 不确定性加权 (Uncertainty Weighting) - **推荐**
出自 Kendall & Gal (CVPR 2018) 的经典论文 *"Multi-Task Learning Using Uncertainty to Weigh Losses"*。
*   **原理：** 将每个任务的 Loss 视为服从高斯分布，引入可学习参数 $\sigma$ (同方差不确定性) 作为权重。
*   **公式：**
    $$ L_{total} = \frac{1}{2\sigma_1^2} L_{seq} + \frac{1}{2\sigma_2^2} L_{token} + \log(\sigma_1) + \log(\sigma_2) $$
*   **优点：** $\sigma$ 会随着训练自动更新。如果某个任务 Loss 难降，$\sigma$ 会变大，从而自动降低该任务 Loss 的权重，防止梯度爆炸；反之亦然。无需手动调参。

#### C. 动态梯度平衡 (Dynamic Gradient Balancing)
这类方法关注梯度的范数或下降速度，而非仅仅看 Loss 数值。

1.  **GradNorm (Gradient Normalization)：**
    *   **原理：** 强制让所有任务的梯度范数（Gradient Norm）保持在一个相似的范围内，或者让所有任务的学习速度（Loss 下降率）保持一致。
    *   **操作：** 动态调整权重 $\lambda_i(t)$，使得梯度范数较大的任务权重减小，梯度范数较小的任务权重增大。

2.  **PCGrad (Project Conflicting Gradients)：**
    *   **原理：** 专门解决梯度冲突。计算两个任务梯度的余弦相似度，如果夹角大于 90 度（冲突），则将一个任务的梯度投影到另一个任务梯度的法平面上，消除冲突分量。
    *   **适用：** 特别适合任务间存在强烈负迁移的场景。

#### D. 基于课程学习 (Curriculum Learning) 或 交替训练
*   **交替训练 (Alternating Training)：** 不是每步都加和，而是这一步只更新任务 A，下一步只更新任务 B。这可以避免单步内的梯度冲突，但需要调整学习率策略。
*   **分阶段训练：** 先训练难的任务，再联合训练；或者先预训练共享层，再冻结部分参数微调特定头。

### 总结建议
如果在工程实践中遇到 `seq_loss + token_loss` 导致的训练不平衡：
1.  **首选方案：** 实施 **不确定性加权 (Uncertainty Weighting)**，因为它实现简单（只需增加几个可学习参数）且效果通常优于手动调参。
2.  **次选方案：** 如果显存受限或不想修改 Loss 结构，可以先统计两个 Loss 在初始阶段的数量级比值，手动设定一个固定的 $\lambda$ 进行平衡（例如 $L = L_{seq} + 0.1 \times L_{token}$）。
3.  **进阶方案：** 如果发现两个任务互相干扰严重（一个涨一个跌），尝试 **PCGrad** 或 **GradNorm**。

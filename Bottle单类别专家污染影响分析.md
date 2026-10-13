# Bottle 单类别专家污染影响分析

## 1. 要回答的问题

此前 `latest_only` 三种子实验已经统计过专家生命周期：正常专家误删率为 22.22%，最终仍残留少量异常专家。但这些统计只能说明专家是否被准入、删除和保留，不能回答异常专家是否真正参与了评分，以及它们对最终检测性能造成了多大影响。

本次补充分析回答三个问题：

1. 异常专家实际参与评分多少步；
2. 异常专家参与评分时，正常图像和异常图像的分数如何变化；
3. 离线使用 GT 移除异常专家后，Image AUROC、Pixel AUROC 和 AUPRO 会变化多少。

## 2. 分析方法

分析对象是 `reentry-ablation-full/latest_only` 的 Bottle 完整测试流，包含 seed 0、1、2，每条流 83 张图像，其中首图因为没有严格历史参考而不可评分。

本次没有重新训练或改变专家生命周期，而是固定原始时间线进行离线重放：

1. 按原实验配置重新提取 DINO 特征：4 个层 `[5, 11, 17, 23]`、3 个尺度 `[1, 3, 5]`；
2. 使用原时间线中的 `scoring_reference_image_ids` 精确重放原始评分；
3. 只在正式专家委员会评分阶段，根据 GT 删除其中来自异常图像的专家引用；
4. 其他正常专家、专家准入/删除时间和严格历史 fallback 全部保持不变；
5. 重新计算图像级和像素级指标。

该 oracle 使用 GT，因此只用于事后判断污染影响，不能作为正式在线算法。

## 3. 重放有效性检查

三个种子的逐图重放分数与原时间线完全一致：最大绝对误差均为 0。重算得到的 Image AUROC、Pixel AUROC、AUPRO 等指标也与原 `metrics.json` 完全一致，所有指标误差均为 0。

因此，oracle 与原实验使用的是同一特征、同一评分公式和同一专家时间线，结果具有可比性。

## 4. 原有生命周期结果

`latest_only` 三种子均值为：

| 指标 | 均值 |
|---|---:|
| 正常专家准入数 | 2.67 |
| 异常专家准入数 | 1.33 |
| 正常专家删除数 | 0.67 |
| 异常专家删除数 | 1.00 |
| 最终正常专家数 | 2.00 |
| 最终异常专家数 | 0.33 |
| 正常专家误删率 | 22.22% |
| 异常专家删除率 | 50.00% |

这里的 22.22% 是“被删除的正常专家数 / 曾准入的正常专家数”，不是专家污染率。

## 5. 异常专家实际参与评分的程度

| seed | 正式专家评分步数 | 污染暴露步数 | 暴露比例 | 异常引用槽位占比 |
|---:|---:|---:|---:|---:|
| 0 | 64 | 50 | 78.13% | 41.46% |
| 1 | 72 | 0 | 0.00% | 0.00% |
| 2 | 61 | 49 | 80.33% | 29.70% |
| 汇总 | 197 | 99 | 50.25% | 26.35% |

“污染暴露步数”指当前图像评分所使用的正式专家中至少包含一个异常专家。“异常引用槽位占比”是全部正式专家引用中异常专家引用所占比例。

具体异常专家为：

| seed | 专家图像 ID | 异常类型 | 准入步 | 被引用步数 | 首次—最后引用步 |
|---:|---:|---|---:|---:|---|
| 0 | 30 | broken_large | 30 | 26 | 31—56 |
| 0 | 38 | broken_small | 38 | 42 | 39—80 |
| 2 | 22 | broken_small | 22 | 29 | 23—51 |
| 2 | 62 | broken_small | 62 | 20 | 63—82 |

这说明污染不是“异常专家虽然被准入但几乎没有被使用”。在 seed 0 和 seed 2 中，异常专家被长时间用于评分；只是 seed 1 没有产生异常专家。

## 6. 移除异常专家后的 oracle 结果

### 6.1 三种子平均指标

| 指标 | 原始 latest_only | oracle 移除异常专家 | oracle − 原始 |
|---|---:|---:|---:|
| Image AUROC | 0.996237 | 0.996774 | +0.000538 |
| Image F1 | 0.989376 | 0.991913 | +0.002538 |
| Image AP | 0.998773 | 0.998956 | +0.000183 |
| Pixel AUROC | 0.993080 | 0.993027 | −0.000053 |
| Pixel F1 | 0.837678 | 0.835592 | −0.002086 |
| Pixel AP | 0.898528 | 0.899740 | +0.001212 |
| AUPRO | 0.974722 | 0.975940 | +0.001218 |

### 6.2 各种子主要指标

| seed | Image AUROC 原始→oracle | Pixel AUROC 原始→oracle | AUPRO 原始→oracle |
|---:|---|---|---|
| 0 | 0.997581 → 0.998387 | 0.992667 → 0.992901 | 0.972873 → 0.976540 |
| 1 | 0.994355 → 0.994355 | 0.992554 → 0.992554 | 0.974773 → 0.974773 |
| 2 | 0.996774 → 0.997581 | 0.994018 → 0.993627 | 0.976520 → 0.976508 |

Image AUROC 在 seed 0、2 中各提高约 0.00081；seed 1 因为没有异常专家，结果完全不变。Pixel AUROC 和 AUPRO 的变化方向并不完全一致，三种子平均变化都很小。

## 7. 污染暴露时的分数变化

移除异常专家后，seed 0 污染暴露阶段：

- 正常图像平均分数从 0.4143 增至 0.4513，增加 0.0370；
- 异常图像平均分数从 0.7312 增至 0.7990，增加 0.0678。

seed 2 污染暴露阶段：

- 正常图像平均分数从 0.4108 增至 0.4275，增加 0.0167；
- 异常图像平均分数从 0.7554 增至 0.7990，增加 0.0436。

异常专家作为相似参考，会同时压低正常图像和异常图像的距离，但对异常图像的压低更明显，因此轻微压缩了图像级区分度。移除异常专家后，异常分数提高得更多，Image AUROC 因而小幅上升。

不能只用时间上的“污染前/污染后均值”判断因果：seed 0 在最后一次污染引用后仅剩 2 个可评分样本，seed 2 没有稳定的污染后阶段。本报告以同一步、同一正常专家集合下的原始与 oracle 配对重放作为主要证据。

## 8. 结论

### 8.1 对当前 Bottle 性能的影响

在当前 83 张 Bottle 完整测试流和三个随机种子上，异常专家虽然在两个种子中被频繁使用，但对最终指标的净影响较小：移除异常专家后，平均 Image AUROC 只提高 0.00054，Pixel AUROC 基本不变，AUPRO 提高 0.00122。

因此可以说：**当前 Bottle 检测性能对这一级别的专家污染具有较强容忍性，污染不是当前高 AUROC 的主要瓶颈。**

### 8.2 对专家机制可靠性的影响

不能说专家污染已经可以忽略。两个种子中约八成正式专家评分步骤暴露于异常专家，汇总后异常引用占全部专家引用槽位的 26.35%。这与“干净、可信的正常专家委员会”目标不一致，也可能在更复杂类别、稀有异常流或跨类别流中产生更大影响。

更准确的汇报方式是：

> Bottle 单类别中存在较明显的专家污染暴露，但离线 oracle 表明它对当前三种子检测指标的影响很小。现阶段可将专家污染列为机制可靠性限制，而不必把它视为 Bottle 性能下降的首要原因；其跨类别和长期流风险仍需保留观察。

## 9. 使用的文件

### 9.1 原实验结果

- `D:\multimusc2\reentry-ablation-full\reentry_ablation_summary.json`：实验配置、三种子汇总和生命周期统计。
- `D:\multimusc2\reentry-ablation-full\latest_only\seed_0\timeline.json`
- `D:\multimusc2\reentry-ablation-full\latest_only\seed_1\timeline.json`
- `D:\multimusc2\reentry-ablation-full\latest_only\seed_2\timeline.json`：逐步专家状态、评分引用、图像分数与离线 GT 标签。
- `D:\multimusc2\reentry-ablation-full\latest_only\seed_0\metrics.json`
- `D:\multimusc2\reentry-ablation-full\latest_only\seed_1\metrics.json`
- `D:\multimusc2\reentry-ablation-full\latest_only\seed_2\metrics.json`：原始性能指标。

### 9.2 数据、模型和源码

- `D:\multimusc2\MultiMuSc2\data\bottle`：MVTec AD Bottle 图像与像素掩码。
- `C:\Users\PC\.cache\torch\hub\checkpoints\dinov2_vitl14_pretrain.pth`：原实验 DINOv2 权重。
- `MultiMuSc2/examples/run_dynamic_dino.py`：DINO 特征提取器和原始实验流程。
- `MultiMuSc2/models/modules/_DYNAMIC_DINO_STREAM.py`：DINO patch 距离、top-min 聚合和在线评分公式。
- `MultiMuSc2/models/modules/_MSM.py`：区间平均聚合函数。
- `MultiMuSc2/datasets/mvtec.py`：Bottle 数据读取及图像、掩码预处理。
- `MultiMuSc2/utils/metrics.py`：Image AUROC、Pixel AUROC 和 AUPRO 计算。

### 9.3 本次新增分析及输出

- `MultiMuSc2/examples/analyze_bottle_expert_contamination.py`：固定原时间线、重放原分数并执行 GT-only oracle 移除分析。
- `outputs/bottle-contamination-impact/contamination_impact.json`：完整种子级和汇总结果。
- `outputs/bottle-contamination-impact/contamination_steps.csv`：每个可评分步骤的原始分数、oracle 分数、专家引用及污染暴露信息。

## 10. 限制

1. 只有 Bottle、三个随机种子，不能直接推广到其他类别或跨类别流。
2. 当前完整测试流的异常比例较高，不等同于真实工业场景中的稀有异常流。
3. oracle 使用 GT 删除异常专家，只能用于影响归因，不能进入在线方法。
4. 本次固定原专家时间线，只改变评分参考。当前架构中异常分数不反向参与专家生命周期，因此这种固定时间线适合隔离评分影响，但它不是一种可部署的专家净化算法。

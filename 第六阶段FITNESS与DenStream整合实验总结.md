# 第六阶段 FITNESS 与 DenStream 整合实验总结

## 目标与结论

本阶段将 FITNESS、DenStream 与统一 ΔJ 接回既有动态专家链路，不新建平行状态机。最终结论为：四类别、三个 seed 的 D 组均满足 Raw AUROC 相对 B/C 最佳值下降不超过 0.01，活动专家数始终不超过 2，因果引用违规和无效冗余均为 0。FITNESS 与 DenStream 在多个类别中产生了不同于单机制的 temporary、成熟及专家轨迹，因此不是仅增加日志。

冗余不再要求为零，而是要求具有独占覆盖、正移除损失或正评分收益。所有 D 组出现第二专家时均至少满足一项；有效冗余始终低于原始冗余。

## 核心实现

- `_record_photometric_state()` 与 `_update_matched_historical_state()`：FITNESS 匹配活动状态时建立不占槽位的 linked temporary，原活动专家中心不被拖动。
- `_admit_quarantined_member()`：成熟后统一经过 ΔJ；空闲槽位仅以 0.25 权重准入，满预算仍走 virtual swap。
- `_expert_dino_coverage_removal_losses()` 及覆盖审计：记录有效冗余、独占覆盖、Channel/DINO 移除损失和无效冗余。
- `_unified_delta_j_shadow()`：把被淘汰专家的可靠性纳入边际收益，避免保留低可靠污染专家、淘汰干净专家。
- `ablate_gradual_drift.py`：把生命周期、覆盖、冗余和污染审计写入实验摘要。

## 实验影响

| 既有结果 | 处理 |
| --- | --- |
| A/B/C 配对实验 | 代码路径未改变，复用原结果 |
| D 组自然流 | 可靠性淘汰项可能影响替换，重跑 Bottle/Cable/Capsule seed 0/1/2 与 Metal Nut seed 1 |
| 第五阶段污染 C | 作为污染贡献上界参考 |
| 第六阶段受控污染 D | 重新运行并确认污染专家真实进入专家库 |

## A/B/C/D 配对结果

表中差值为 `D - max(B, C)`。

| 类别 | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| Bottle | 1.0000 / +0.0000 | 1.0000 / +0.0000 | 1.0000 / +0.0000 |
| Cable | 0.9048 / +0.0286 | 0.7238 / -0.0095 | 0.9714 / +0.0190 |
| Capsule | 0.9048 / -0.0095 | 0.9238 / +0.0000 | 0.9524 / +0.0095 |
| Metal Nut | 0.9905 / +0.0000 | 0.9714 / -0.0095 | 1.0000 / +0.0000 |

所有配对的数据流指纹一致。Bottle seed 0 的 D 原始状态覆盖率为 0.6364，低于 DenStream-only 的 0.8182，但 B 建立的是 `a+a`，D 建立的是 `a+d`；D 的最差正常分数更低（0.8677 对 0.8784），DINO 移除损失更高（0.1088 对 0.0697）。这是可解释的覆盖—多样性取舍，不应通过复制同状态专家追平原始覆盖率。

## 污染与因果审计

Bottle patch 30% 受控污染中，D 组真实注入 1 个污染专家。异常有效贡献比例为 `4.46e-5`，反事实分数偏移的均值、95% 分位和最大值均为 0，Raw AUROC 为 1.0。第五阶段 C 的对应贡献为严格 0，因此 D 不能宣称数值更低；它满足预设 `1e-4` 可忽略容差，且未改变最终分数。全部实验 `causal_reference_violation_count=0`。

## 验证与限制

```powershell
& $py -m py_compile examples\ablate_gradual_drift.py examples\ablate_multistage_drift.py `
  models\modules\_DYNAMIC_EXPERT.py models\modules\_DYNAMIC_PIPELINE.py `
  models\modules\_MSM.py models\musc.py examples\run_priority5_contamination_matrix.py

& $py -m unittest MultiMuSc2.tests.test_channel `
  MultiMuSc2.tests.test_dynamic_dino_stream MultiMuSc2.tests.test_dynamic_expert `
  MultiMuSc2.tests.test_dynamic_pipeline MultiMuSc2.tests.test_gradual_drift `
  MultiMuSc2.tests.test_msm_online MultiMuSc2.tests.test_robust_shift
```

语法检查通过，247 项测试通过。已知限制：Capsule seed 0 的 pixel AUROC/AUPRO 仍低于部分单机制结果；Metal Nut seed 1 的最差正常分数有代价；当前完整实验每条流仅 22 个样本，结论仍需更长生产流验证。

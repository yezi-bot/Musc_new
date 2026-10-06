# Repository Guidelines

## 项目结构与模块组织

当前仓库是用于 MVTec AD `bottle` 类别 Channel 实验的轻量验证项目。核心脚本为 `validate_density_bottle.py`，负责读取图像、提取或加载 DINOv2 patch 特征、运行 span-only 基线和方案 B（位置约束、patch reliability、effective span），并生成 CSV、JSON 与 PNG 结果。数据集、模型权重、特征缓存和实验输出应放在仓库外的临时目录，不要加入 Git。

## 构建、测试与开发命令

使用项目已有的 Python 环境运行：

```powershell
$py='C:\Users\PC\miniconda3\envs\musc\python.exe'
& $py -m py_compile validate_density_bottle.py
```

上面的命令检查语法，不会运行模型。使用缓存特征进行完整 bottle 验证：

```powershell
& $py validate_density_bottle.py --data-root <MVTec根目录> `
  --output-dir <输出目录> --dinov2-repo <DINOv2仓库> `
  --features-cache <特征缓存> --scheme-b-only
```

快速回归可增加 `--max-samples 5`。结果应检查 `bottle_density_summary.json`、两条 CSV 曲线及 `bottle_contamination_curves.png`。

## 编码风格与命名约定

使用 Python 3 风格和 4 空格缩进；函数、变量使用 `snake_case`，类使用 `PascalCase`，命令行参数使用短而明确的 kebab-case。保持张量计算在 PyTorch 中批量执行，避免新增逐 patch 的 Python 双重循环。修改应优先保持现有函数接口和输出字段兼容。

## 测试指南

仓库目前没有独立测试框架。每次修改至少运行 `py_compile` 和 `--max-samples 5` smoke test；涉及 Channel 时同时检查 baseline 与方案 B 的 `sample_count`、污染率、保留数量及异常/正常 Channel 审计结果。GT 只能用于离线评估，不得参与在线匹配或 reliability 计算。

## 提交与 Pull Request

当前仓库没有可供归纳的历史提交，提交信息请使用简短、动词开头的描述，例如 `Add position-aware channel weighting`。PR 应说明实验设置、数据与特征缓存位置、baseline/B 结果差异，并附上关键 JSON/曲线截图或路径；不要提交数据集、模型权重、缓存、密钥或大体积生成物。

在 GitHub 仓库 `https://github.com/yezi-bot/Musc_new` 中，新建并长期使用一个专用测试分支 `codex/tests`，所有测试代码统一存放在该分支，不为每次测试重复创建新分支，也不直接写入主分支。每次提交必须清楚记录本次测试的内容、关键配置和验证范围；提交标题保持简短，必要时在正文中补充数据类别、参数与对比方案，例如 `Test global soft effective span on bottle`。

只要当前操作不修改 `main` 分支，在专用测试分支上的 `git commit` 与 `git push` 无需再次请求用户批准，可在检查 `git diff`、`git status` 和验证结果后直接执行。禁止强推、重写历史或把无关文件混入提交。

不要在已有测试脚本中无限追加彼此独立的实验。若测试目标明显不同，必须建立各自独立的 `.py` 文件保存，例如 density 验证保留在 `validate_density_bottle.py`，动态 EX-MSM 测试应新建类似 `validate_dynamic_ex_msm.py` 的脚本。只有同一实验的参数消融、缺陷修复或紧密相关阶段才可继续修改原脚本；可复用逻辑应提取为公共模块或通过导入复用，避免复制代码。提交前检查本次改动是否混入其他实验目标，并在提交信息中明确对应的测试脚本和目标。

## 配置与数据安全

通过命令行传入本地路径，不在脚本中硬编码私有凭证。提交前检查 `git status` 和 `git diff`，确认只包含源码或必要文档。

## 及时反馈与核心代码定位

完成用户指定功能后，应在交付说明中立即指出对应的核心代码块，包含文件名、函数或类名，以及它如何满足要求。例如实现“位置约束”后，应明确说明 `ChannelMemory.patch_reliability()` 的位置候选筛选和 `ChannelMemory.update()` 的 `position_radius` 匹配逻辑。若功能尚未实现、只完成了实验脚本或存在已知限制，也必须直接说明，不能只报告结果数字。

## 代码提交前的监督流程

每次生成或修改代码后，应先完成与改动范围相称的初步验证，再在向用户交付、进入下一阶段或执行 `git commit`/`git push` 前，自动读取并遵循 `C:\Users\PC\Documents\ChatGPT\监督\AGENTS.md`。按照监督项目规则核对需求、复查实际测试、寻找漏洞并给出修复清单，无需事先请求用户许可。

监督检查默认只读。监督未发现问题时才能继续交付或提交；监督提出需要修改的问题时，执行修复、重新测试或提交修复前必须获得用户明确批准。用户未批准时应停留在监督结果和修复建议，不得擅自修改。纯文档改动不触发代码监督，除非用户明确要求。

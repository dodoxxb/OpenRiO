# OpenRiO

**RiO-DETR 社区复现：可核验的模型权重、复现经验与实用优化。**

[![报告 DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.16995.11040-blue.svg)](https://doi.org/10.13140/RG.2.2.16995.11040) [![许可: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10-blue.svg)](requirements.txt)

[English](README.md) · 简体中文

[技术报告 PDF](report/main.pdf) · [ResearchGate 报告页](https://www.researchgate.net/publication/415911327_OpenRiO_A_Community_Reproduction_of_RiO-DETR_with_Practical_Accuracy_and_Efficiency_Improvements) · [模型说明](docs/MODEL_CARDS.rst) · [优化手册](docs/OPTIMIZATION_GUIDE.rst) · [评估协议](docs/PROTOCOL.rst) · [引用信息](CITATION.cff)

**Zhicheng Du · Lan Ma（通讯作者）**  
清华大学深圳国际研究生院

OpenRiO 围绕两个目标展开：先复现有竞争力的旋转目标检测模型，再基于固定的复现权重研究精度与执行效率的改进。我们同时提供权重、评估约定、训练状态经验和对照证据，帮助后续工作建立可信的基线。原始检测器与方法归功于 [RiO-DETR 作者](https://arxiv.org/abs/2603.09411)；本项目是在其[官方代码](https://github.com/RicePasteM/RiO-DETR)基础上开展的独立社区工作。

> **预发布 v0.1.0-preview.7 · 2026 年 10 月 16 日。** 当前准备的发布包包含 N/S/M/L 权重、预测、DIOR-R 完整评估及有收益的优化方案。X、FAIR1M、DOTA 仍在推进；可迁移的完整训练流程和独立多 seed 验证尚未完成。

本次精简版只保留一份报告 PDF、四份使用文档及合并后的必要实验记录；去掉论文中间文件和重复快照。模型与运行时代码和 preview.6 完全一致。

![OpenRiO 结果总览：社区复现与论文 AP50 的差距，以及两个历史 Nano 起点上的受控 AP75 改进。](docs/assets/evidence-overview.png)

*左：复现结果，X 尚在进行。右：历史固定 Nano 权重上的排序与几何改进，分别有对应对照；不能把这些增益叠加到最新复现分数。[图中数据](logs/mechanism_checks.json)。*

## 复现结果

**DIOR-R 完整测试集 11,738 张图像；单模型、单视图、未校准 EMA；旋转框 VOC2007 十一点 AP50（%）。**

| 规格 | 参数量 | 论文 AP50 | OpenRiO AP50 | 差值 | 权重 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| N | 4.01 M | 71.920 | **71.891** | −0.029 | `n.pth` |
| S | 8.27 M | 74.440 | **75.020** | +0.580 | `s.pth` |
| M | 19.21 M | 75.730 | **76.448** | +0.718 | `m.pth` |
| L | 28.10 M | 76.110 | **76.199** | +0.089 | `l.pth` |

四份权重均通过重新全量推理和独立多边形匹配/AP 核验。表中保留训练记录的选定节点；重新核验的差异均小于 0.001 AP 点。完整数值、源状态与哈希见 [models.json](models.json)。已测试的继续训练方案达到经验平台期，不代表所有可能配方的全局收敛。

**适用边界：** N 使用公开的五坐标几何路线，S/M/L 使用明确披露的紧凑四坐标重建。每条路线仍是一个训练谱系；历史选择参考过测试集反馈，标准划分中也发现了 23 张跨划分字节完全相同的图像。因此不能把分数接近等同于作者训练完全一致、无偏泛化或独立多 seed 验证。详见[评估协议](docs/PROTOCOL.rst)和[模型说明](docs/MODEL_CARDS.rst)。

## 固定模型上的优化

每项优化都绑定自己的源权重和必要对照，历史优化起点与上表最新复现权重分开标注。

| 方案 | 同起点比较 | 代价或适用范围 |
| :--- | :--- | :--- |
| S：多视图排序与加权几何融合 | AP50 **73.982 → 76.465** | 五次前向，历史 S 权重 |
| M：多视图排序与均匀几何融合 | AP50 **74.790 → 77.512** | 五次前向，历史 M 权重 |
| N：用训练图像校准归一化统计量 | AP50 **71.803 → 71.978** | 两层、256 张记录在案的训练图像，独立父权重 |
| N/S：带源模型约束的在线适配 | AP50 **+0.688 / +0.743** | 指定合成噪声流，涉及参数更新 |
| N：CUDA Graph 执行 | 中位耗时 **34.17 → 30.36 ms** | 共享设备上的配对测速，从图像字节到 CPU 输出 |

包内还提供两条 Nano 多视图路线及固定框重排序对照。研究中的 CUDA Graph 全量检查保持框与 logits 逐位一致；上述共享设备耗时不能作为所有硬件的加速承诺。适配在干净域下降，其 AP 属于完整在线预测轨迹，**不是把终点参数当作静态权重加载后的 AP**。可移植入口的完整流复评已通过：N/S 相对保留的匹配冻结对照分别提高 **0.727 / 0.775 AP50**。这是同一训练来源的重新执行，不是新增独立训练 seed；上表保留原始实验数值。[完整复验记录](logs/optimizations.json)。

[完整命令、对照与精确指标 →](docs/OPTIMIZATION_GUIDE.rst)

## 快速开始

使用 **Python 3.10**。已测试 PyTorch **2.9.1** / torchvision **0.24.1**；根据设备选择合适的 CUDA 或 CPU 构建。

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python -m pip install -r requirements.txt
```

### 放置权重

检测器权重与适配快照**不存放在 Git 中**。请从 [Releases 页面](https://github.com/dodoxxb/OpenRiO/releases/tag/v0.1.0-preview.7)下载，并按下表放入对应目录。

```text
OpenRiO/
├── README.md
├── openrio.py
├── models.json
├── configs/
├── weights/       # 检测器 .pth 文件  ->  下表的 10 份资产
└── adapters/      # 适配终点快照    ->  下表的 4 份资产
```

**检测器权重 — 放入 `weights/`**

| Release 资产 | 大小 | 用途 |
| :--- | ---: | :--- |
| `n.pth` | 15.6 MB | Nano 主复现参考（AP50 71.891） |
| `s.pth` | 32.0 MB | Small 主复现参考（75.020） |
| `m.pth` | 73.9 MB | Medium 主复现参考（76.448） |
| `l.pth` | 107.9 MB | Large 主复现参考（76.199） |
| `n_calibrated.pth` | 15.6 MB | Nano 归一化校准结果（71.978） |
| `n_compact_multiview.pth` | 15.7 MB | 紧凑四坐标 Nano 多视图源权重 |
| `n_rotation_multiview.pth` | 15.6 MB | 公开五坐标 Nano 多视图源权重 |
| `s_multiview.pth` | 32.0 MB | Small 多视图（73.982 → 76.465）的历史起点 |
| `m_multiview.pth` | 73.9 MB | Medium 多视图（74.790 → 77.512）的历史起点 |
| `n_calibration_parent.pth` | 15.6 MB | Nano 校准对照的父权重 |

**适配终点快照 — 放入 `adapters/`**

| Release 资产 | 大小 | 用途 |
| :--- | ---: | :--- |
| `n_noise_terminal_norms.pth` | 0.23 MB | Nano 噪声流的终点归一化快照 |
| `n_noise_terminal_norms.json` | 0.5 KB | 上者对应的元数据与源绑定 |
| `s_noise_terminal_norms.pth` | 0.32 MB | Small 噪声流的终点归一化快照 |
| `s_noise_terminal_norms.json` | 0.5 KB | 上者对应的元数据与源绑定 |

因此像 `n.pth` 这样的裸文件名指的是 `weights/n.pth`。所有资产文件名互不重复，从 Releases 页面平铺下载不会产生冲突。`models.json` 记录了每份权重的预期路径与 SHA-256，每个 `adapters/*.json` 则把快照绑定到其父模型。

`n`、`s`、`m`、`l` 是主要复现参考；`n_calibrated` 是校准结果。四份 `*_multiview` 与 `n_calibration_parent` 用于复现优化的精确历史起点。适配快照不是独立检测器。

资产齐全后核验：

```bash
python tools/verify_release.py
```

### 预测、全量评估与独立核验

```bash
python openrio.py predict --model m --image /path/to/square-tile.jpg \
  --device cuda:0 --output outputs/m-example

CUDA_VISIBLE_DEVICES=0 python openrio.py evaluate --model l \
  --data-root /path/to/DIOR-R --output outputs/l-full \
  --batch-size 1 --workers 2 --memory-fraction 0.18

python tools/independent_eval.py --data-root /path/to/DIOR-R \
  --predictions outputs/l-full/predictions.pt \
  --native-metrics outputs/l-full/metrics.json \
  --output outputs/l-independent
```

数据放置方式见[评估协议](docs/PROTOCOL.rst)。每次使用新的输出目录，每个进程加载一个模型。`--limit` 只做诊断，不报告 AP。当前验证范围是方形图块，任意图像 letterbox 的逆变换尚未验证。显存比例限制只约束当前分配器，不会为进程预留显存。

### 运行精度或速度优化

```bash
# 76.465 AP50 对应的历史 S 起点与五视图方案。
python optimize.py evaluate --model s_multiview --method weighted \
  --data-root /path/to/DIOR-R --device cuda:0 --output outputs/s-weighted

# 常规执行 / CUDA Graph 的配对测速与输出一致性检查。
python benchmark.py --model n --data-root /path/to/DIOR-R \
  --device cuda:0 --output outputs/n-graph
```

校准、在线适配、冻结源模型对照及独立评分命令见[优化手册](docs/OPTIMIZATION_GUIDE.rst)。

## 经验与洞察

- **模型状态是训练配方的一部分。** EMA、归一化统计、优化器历史和继续训练安排都会影响终点，配置一致并不足以保证状态一致。
- **候选覆盖、置信度排序、定位精度需要分别诊断。** 固定框重排序和仅修改坐标的对照，帮助区分“没有候选”“候选排得不好”和“位置不够准”。
- **收益有明确范围。** 相同方案在不同规格、不同域偏移下可能产生不同效果；精度、适配与执行优化分别报告源权重、代价和对照。

[技术报告](report/main.pdf)围绕这些问题组织机制分析。失败尝试、替代解释和局限保留在 discussion 与附录中，失败部署方案和实验权重不作为推荐资产发布。YOLO 迁移的正面证据也在报告中讨论，但尚未提供完整的跨框架模型库。

## 进行中 / 即将推出

以下是发布时的状态快照，不承诺最终分数或完成时间。

| 内容 | 状态 |
| :--- | :--- |
| DIOR-R N/S/M/L | **已提供：** 核验权重、预测与完整评估 |
| 有收益的优化方案 | **已提供：** 多视图、校准、在线适配与 CUDA Graph 测速 |
| DIOR-R X | **进行中：** 最佳 75.127 AP50；独立全量评测 75.126；继续训练与训练状态对照仍在推进；X 权重未纳入本次发布 |
| FAIR1M-2.0 M/X | **进行中：** 训练、场景/切片评估与论文协议核对 |
| DOTA-1.0 | **进行中：** 训练与导出，官方服务器分数待提交 |
| 可迁移的完整多阶段训练 | **准备中：** 安装至权重导出的全流程验证尚未完成 |
| 三个独立检测器 seed | **计划中：** 继续训练或校准 seed 不替代独立训练 |
| 更多模型范式 | **探索中：** 已有迁移证据，完整复现套件尚未发布 |

详见[日期实验快照](logs/research_status.json)和[训练阶段设置](docs/RECIPE_STATUS.rst)。

最近一个完成的 X 训练窗口收于 **75.067 AP50**，低于其保留的最佳值；下一轮平台确认窗口正在运行。FAIR-M epoch 32 达到 **40.480 场景 / 34.877 切片 AP50**，但其评估协议尚未与论文对齐。新的归一化统计与物理 batch 对照仍在等待首次完整评测。[日期快照](logs/research_status.json)将这些进行中的试验与已发布的分开记录。

## 文档与证据

- [模型说明](docs/MODEL_CARDS.rst)：训练谱系、配置、EMA 与选择依据。
- [评估协议](docs/PROTOCOL.rst)：数据布局、角度、difficulty 与独立核验。
- [优化手册](docs/OPTIMIZATION_GUIDE.rst)：命令、对照、源权重与适用边界。
- [技术报告 PDF](report/main.pdf)：复现、机制、优化与失败分析。
- [证据目录](logs/)：完整评估、身份核验与保留下来的继续训练记录。
- [上传说明](UPLOAD_GUIDE.rst)：GitHub 仓库根目录和 Releases 资产的放置方式。

保留的继续训练曲线不冒充完整从头训练历史；技术报告目前是工作草稿，尚未提交 arXiv 或经过同行评审。

## 致谢、引用与许可

请为原始方法引用 [RiO-DETR](https://arxiv.org/abs/2603.09411)，并依据 [CITATION.cff](CITATION.cff) 为社区复现、分析与工程工作引用当前版本。技术报告 PDF 同时发布于 [ResearchGate](https://www.researchgate.net/publication/415911327_OpenRiO_A_Community_Reproduction_of_RiO-DETR_with_Practical_Accuracy_and_Efficiency_Improvements)。

引用代码与证据记录：

```bibtex
@software{du2026openrio,
  author       = {Du, Zhicheng and Ma, Lan},
  title        = {OpenRiO: A Community Reproduction of RiO-DETR with
                  Practical Accuracy and Efficiency Improvements},
  version      = {0.1.0-preview.7},
  month        = {10},
  year         = {2026},
  institution  = {Tsinghua Shenzhen International Graduate School,
                  Tsinghua University},
  url          = {https://github.com/dodoxxb/OpenRiO}
}
```

引用技术报告：

```bibtex
@techreport{du2026openrioreport,
  author      = {Du, Zhicheng and Ma, Lan},
  title       = {OpenRiO: A Community Reproduction of RiO-DETR with
                 Practical Accuracy and Efficiency Improvements},
  month       = {10},
  year        = {2026},
  doi         = {10.13140/RG.2.2.16995.11040},
  institution = {Tsinghua Shenzhen International Graduate School,
                 Tsinghua University},
  url         = {https://www.researchgate.net/publication/415911327_OpenRiO_A_Community_Reproduction_of_RiO-DETR_with_Practical_Accuracy_and_Efficiency_Improvements}
}
```

源代码采用 [Apache-2.0](LICENSE)，上游归属见 [NOTICE.txt](NOTICE.txt)。权重和报告图像的条款单独列于 [WEIGHTS_NOTICE.txt](WEIGHTS_NOTICE.txt) 及相关 notices，数据集图像和标注不随包分发。

联系：[Zhicheng Du](mailto:duzc21@mails.tsinghua.edu.cn) · [Lan Ma](mailto:malan@sz.tsinghua.edu.cn)。

技术报告新增官方 baseline／社区模型同图对比、固定来源几何优化与反例、全量诊断分布及机制示意图。[图中数值与样例选择记录](logs/mechanism_checks.json)。

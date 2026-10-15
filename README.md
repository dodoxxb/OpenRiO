<div align="center">

# OpenRiO

### A Community Reproduction of RiO-DETR with Practical Accuracy and Efficiency Improvements

**Zhicheng Du**<sup>1</sup>&nbsp; · &nbsp;**Lan Ma**<sup>1,†</sup>

<sup>1</sup>Tsinghua Shenzhen International Graduate School, Tsinghua University &nbsp;&nbsp; <sup>†</sup>Corresponding author

<p>
  <a href="https://doi.org/10.13140/RG.2.2.16995.11040">
    <img src="https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.16995.11040-blue.svg" alt="Report DOI">
  </a>
  <a href="report/main.pdf">
    <img src="https://img.shields.io/badge/Technical%20Report-PDF-brightgreen.svg" alt="Technical report PDF">
  </a>
  <a href="https://github.com/dodoxxb/OpenRiO/releases/tag/v0.1.0-preview">
    <img src="https://img.shields.io/badge/release-v0.1.0--preview-blue.svg" alt="Release">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="requirements.txt">
    <img src="https://img.shields.io/badge/python-3.10-orange.svg" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.9.1-%23EE4C2C.svg?&logo=PyTorch&logoColor=white" alt="PyTorch">
  </a>
</p>

<p>
  <b>English</b> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p>
  <i>OpenRiO has two goals: reproduce competitive rotated object detection models,
  then improve their accuracy and execution on fixed checkpoints. We release the
  models together with the evaluation conventions, training-state lessons and
  controls needed to understand their results.</i>
</p>

<img src="docs/assets/evidence-overview.png" width="100%" alt="OpenRiO evidence overview: community versus paper AP50, and controlled AP75 gains on two earlier Nano checkpoints.">

<p>
  <b>Left:</b> audited reproduction endpoints at the evidence cut; X is ongoing.
  <b>Right:</b> improvements on earlier frozen Nano sources, with separate controls
  for score ordering and geometry. These deltas are not added to the latest
  reproduction scores. <a href="logs/mechanism_checks.json">Figure values</a>.
</p>

</div>

**TL;DR:** We independently reproduce RiO-DETR N/S/M/L on DIOR-R within **−0.03 to +0.72 AP50** of the paper, and show **+0.09 to +2.72 AP50** improvements that come from methods applied *after* training — multi-view inference, normalization calibration and online adaptation — each with its own frozen source and control.

---

📢 A Note on This Release
---

**This is an independent community project, not the official RiO-DETR release.** Credit for the detector and its method belongs entirely to the [RiO-DETR authors](https://arxiv.org/abs/2603.09411); our code is based on their [official repository](https://github.com/RicePasteM/RiO-DETR).

- **Released**: 16 October 2026 (`v0.1.0-preview`). N/S/M/L checkpoints, inference and complete DIOR-R evaluation are available. X, FAIR1M and DOTA are ongoing.
- **Honest scope**: these are *community recipes*. The public five-coordinate Nano path and a disclosed compact four-coordinate S/M/L reconstruction each have one training ancestry. Close scores do **not** establish identical author training, unbiased generalization or independent-seed replication.
- **Not fabricated history**: retained continuation curves are not presented as a complete from-scratch training run. The report is a working technical-report draft — **not** an arXiv submission and **not** peer-reviewed.

---

✨ Highlights
---

| | |
| :--- | :--- |
| 🔍 **Auditable reproduction** | Every endpoint passed fresh full-test inference *and* independent polygon-matching / AP verification |
| 📈 **Post-hoc accuracy gains** | Multi-view ranking + geometry: **+2.48** (S), **+2.72** (M); calibration **+0.18** (N); adaptation **+0.69 / +0.74** (N/S) |
| ⚡ **Execution gains** | CUDA Graph replay: median **34.17 → 30.36 ms**, with bitwise-identical boxes and logits in the research check |
| 🧪 **Controls, not just claims** | Fixed-box rescoring and coordinate-only ablations separate ranking effects from geometric precision |
| 📦 **Reproducible packaging** | `SHA256SUMS` + `provenance/` hash manifests and a one-command `verify_release.py` integrity check |
| 🌏 **Bilingual** | Full English and 简体中文 documentation, kept structurally aligned |

---

## Model Zoo

### Reproduction results

**Complete DIOR-R test split: 11,738 images, single model, single view, uncalibrated EMA, rotated VOC2007 eleven-point AP50 (%).**

| Model | Params | Paper AP50 | OpenRiO AP50 | Δ | Config | Checkpoint |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **N** | 4.01 M | 71.920 | **71.891** | −0.029 | [yml](configs/n.yml) | [n.pth](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n.pth) |
| **S** | 8.27 M | 74.440 | **75.020** | +0.580 | [yml](configs/s.yml) | [s.pth](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/s.pth) |
| **M** | 19.21 M | 75.730 | **76.448** | +0.718 | [yml](configs/m.yml) | [m.pth](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/m.pth) |
| **L** | 28.10 M | 76.110 | **76.199** | +0.089 | [yml](configs/l.yml) | [l.pth](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/l.pth) |

Each endpoint passed fresh full-test inference and independent polygon-matching / AP verification. The table retains the recorded training-run endpoints; fresh audit differences are below 0.001 AP point. Exact values, source states and digests are in [`models.json`](models.json). Tested continuation recipes reached empirical plateaus; this is **not** a claim of global convergence.

> **Scope matters.** The standard split contains 23 audited byte-identical cross-split images, and repeated test feedback informed checkpoint selection. See the [evaluation protocol](docs/PROTOCOL.rst) and [model cards](docs/MODEL_CARDS.rst).

### Improvements on fixed models

Every comparison below uses its **own frozen source**; historical optimization sources are separate from the primary models above.

| Improvement | Matched result | Cost or scope | Guide |
| :--- | :--- | :--- | :--- |
| Small: multi-view ranking + weighted geometry | AP50 **73.982 → 76.465** | Five forwards; historical Small source | [guide](docs/OPTIMIZATION_GUIDE.rst) |
| Medium: multi-view ranking + uniform geometry | AP50 **74.790 → 77.512** | Five forwards; historical Medium source | [guide](docs/OPTIMIZATION_GUIDE.rst) |
| Nano: normalization calibration with training images | AP50 **71.803 → 71.978** | Two layers, 256 recorded training images; its own parent | [guide](docs/OPTIMIZATION_GUIDE.rst) |
| N/S: source-anchored online adaptation | AP50 **+0.688 / +0.743** | Specified synthetic-noise streams; parameter updates | [guide](docs/OPTIMIZATION_GUIDE.rst) |
| Nano: CUDA Graph execution | Median **34.17 → 30.36 ms** | Paired shared-device byte-to-CPU timing; declared exclusions | [guide](docs/OPTIMIZATION_GUIDE.rst) |

The multi-view methods also include both Nano paths and fixed-box rescoring controls. CUDA Graph retained full-set boxes and logits bitwise in the research check; shared-device latency is contextual, **not** a hardware-independent speed guarantee. Clean-domain adaptation declined. Online AP belongs to the prediction trajectory, **not static inference from the final parameter snapshot**. Fresh portable full-stream runs also passed: N/S gained **+0.727 / +0.775 AP50** over the retained matched frozen controls — reruns of the same trained sources, not new detector seeds, so the table above remains a distinct recorded experiment. See the [completed validation receipt](logs/optimizations.json).

---

## Getting Started

### Installation

Use **Python 3.10**. The tested environment uses PyTorch **2.9.1** and torchvision **0.24.1**; select the appropriate CUDA/CPU build for your machine.

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1
python -m pip install -r requirements.txt
```

A pinned CPU-side lock file is provided in [`requirements-cpu-lock.txt`](requirements-cpu-lock.txt), and the tested environment is recorded in [`environment-tested.json`](environment-tested.json).

### Model assets

The detector weights and adapter snapshots are **not** stored in Git. Download them from the [Releases page](https://github.com/dodoxxb/OpenRiO/releases/tag/v0.1.0-preview) and place each asset in the directory shown below.

```text
OpenRiO/
├── README.md
├── openrio.py
├── models.json
├── configs/
├── weights/       # detector .pth files      -> all 10 assets below
└── adapters/      # terminal adaptation snapshots -> all 4 assets below
```

**Detector weights — place in `weights/`**

| Release asset | Size | Role | Download |
| :--- | ---: | :--- | :--- |
| `n.pth` | 15.6 MB | Nano — primary reproduction reference (AP50 71.891) | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n.pth) |
| `s.pth` | 32.0 MB | Small — primary reproduction reference (75.020) | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/s.pth) |
| `m.pth` | 73.9 MB | Medium — primary reproduction reference (76.448) | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/m.pth) |
| `l.pth` | 107.9 MB | Large — primary reproduction reference (76.199) | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/l.pth) |
| `n_calibrated.pth` | 15.6 MB | Nano normalization-calibration result (71.978) | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n_calibrated.pth) |
| `n_compact_multiview.pth` | 15.7 MB | Compact four-coordinate Nano multi-view source | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n_compact_multiview.pth) |
| `n_rotation_multiview.pth` | 15.6 MB | Public five-coordinate Nano multi-view source | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n_rotation_multiview.pth) |
| `s_multiview.pth` | 32.0 MB | Historical Small source for the 73.982 → 76.465 result | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/s_multiview.pth) |
| `m_multiview.pth` | 73.9 MB | Historical Medium source for the 74.790 → 77.512 result | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/m_multiview.pth) |
| `n_calibration_parent.pth` | 15.6 MB | Parent checkpoint for the Nano calibration control | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n_calibration_parent.pth) |

**Adaptation snapshots — place in `adapters/`**

| Release asset | Size | Role | Download |
| :--- | ---: | :--- | :--- |
| `n_noise_terminal_norms.pth` | 0.23 MB | Terminal normalization snapshot, Nano noise stream | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n_noise_terminal_norms.pth) |
| `n_noise_terminal_norms.json` | 0.5 KB | Metadata and source binding for the snapshot above | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/n_noise_terminal_norms.json) |
| `s_noise_terminal_norms.pth` | 0.32 MB | Terminal normalization snapshot, Small noise stream | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/s_noise_terminal_norms.pth) |
| `s_noise_terminal_norms.json` | 0.5 KB | Metadata and source binding for the snapshot above | [download](https://github.com/dodoxxb/OpenRiO/releases/download/v0.1.0-preview/s_noise_terminal_norms.json) |

Plain filenames such as `n.pth` therefore mean `weights/n.pth`; no two assets share a name, so nothing collides when they are downloaded flat from the Releases page. Primary references are `n`, `s`, `m` and `l`; `n_calibrated` is an optimization result, and the four `*_multiview` models plus `n_calibration_parent` bind comparisons to their exact historical sources. The adapter files are parameter snapshots, **not** standalone detectors.

With all companion assets in place, verify their identity:

```bash
python tools/verify_release.py
```

### Predict and evaluate

Predict a square image tile:

```bash
python openrio.py predict --model m --image /path/to/square-tile.jpg \
  --device cuda:0 --output outputs/m-example
```

Evaluate the complete DIOR-R test set:

```bash
CUDA_VISIBLE_DEVICES=0 python openrio.py evaluate --model l \
  --data-root /path/to/DIOR-R --output outputs/l-full \
  --batch-size 1 --workers 2 --memory-fraction 0.18
```

Independently check the saved predictions:

```bash
python tools/independent_eval.py --data-root /path/to/DIOR-R \
  --predictions outputs/l-full/predictions.pt \
  --native-metrics outputs/l-full/metrics.json \
  --output outputs/l-independent
```

Use the [documented dataset layout](docs/PROTOCOL.rst). Each output directory must be new; run one model per process. `--limit` is diagnostic and suppresses AP. Arbitrary-image letterbox inversion is not validated. The CUDA memory fraction limits the process allocator; it does **not** reserve memory against other jobs.

### Run an accuracy or speed improvement

```bash
# Five-view Small evaluation: exact source of the reported 76.465 AP50 result.
python optimize.py evaluate --model s_multiview --method weighted \
  --data-root /path/to/DIOR-R --device cuda:0 --output outputs/s-weighted

# Paired eager / CUDA Graph timing and output checks.
python benchmark.py --model n --data-root /path/to/DIOR-R \
  --device cuda:0 --output outputs/n-graph
```

Calibration, N/S adaptation, frozen controls and separate scoring commands are in the [optimization guide](docs/OPTIMIZATION_GUIDE.rst).

---

## What we learned

- **Model state is part of the recipe.** EMA, normalization statistics, optimizer history and continuation schedules affect the endpoint; matching a configuration alone is insufficient.
- **Coverage, confidence and localization need separate controls.** Fixed-box rescoring and coordinate-only interventions distinguish useful candidates from their ranking and geometric precision.
- **An improvement has a scope.** Benefits can differ across model sizes and domain shifts. Accuracy, adaptation and execution results require their own sources, costs and controls.

The [technical report](report/main.pdf) develops these observations and mechanisms. Failed attempts, alternate explanations and boundaries remain in its discussion and appendices; failed deployment recipes and experimental weights are not part of the recommended release. Broader YOLO transfer evidence is discussed in the report, but a supported multi-framework model zoo is still pending.

---

## Status

Status is a dated release snapshot, **not** a promise of final scores or completion dates.

| Deliverable | Status |
| :--- | :--- |
| DIOR-R N/S/M/L | ✅ **Available:** audited checkpoints, inference and complete evaluation |
| Useful optimization recipes | ✅ **Available:** multi-view inference, calibration, adaptation and graph benchmarking |
| DIOR-R X | 🔄 **In progress:** best 75.127 AP50; independent full-test audit 75.126; continuation and training-state controls ongoing; X weights not included |
| FAIR1M-2.0 M/X | 🔄 **In progress:** training, scene/patch evaluation and paper-protocol reconciliation |
| DOTA-1.0 | 🔄 **In progress:** training/export; official-server scores pending |
| Portable multi-stage training | 🛠️ **In preparation:** installation-to-export validation pending |
| Three independent detector seeds | 📋 **Planned:** not yet completed or represented by continuation seeds |
| Broader model integrations | 🔬 **Exploratory:** transfer evidence exists; complete reproduction suites are not released |

See the [dated research snapshot](logs/research_status.json) and [recorded training settings](docs/RECIPE_STATUS.rst).

The latest completed X window ends at **75.067 AP50**, below its retained best; the next plateau-confirmation window is running. FAIR-M epoch 32 reaches **40.480 scene / 34.877 patch AP50**, under protocols not yet equated to the paper. New target-normalization and physical-batch controls await their first full evaluations. The [dated snapshot](logs/research_status.json) separates these running trials from released endpoints.

---

## Documentation

| Resource | Contents |
| :--- | :--- |
| [Technical report](report/main.pdf) | Reproduction, mechanisms, improvements and failure analysis |
| [Model cards](docs/MODEL_CARDS.rst) | Model ancestry, configuration, EMA state and selection |
| [Evaluation protocol](docs/PROTOCOL.rst) | Dataset layout, angles, difficulty and independent metric check |
| [Optimization guide](docs/OPTIMIZATION_GUIDE.rst) | Runnable methods, source weights, controls and restrictions |
| [Recipe status](docs/RECIPE_STATUS.rst) | Recorded training settings and continuation state |
| [Evidence records](logs/) | Full-set evaluation, identity checks and retained continuation observations |
| [Upload guide](UPLOAD_GUIDE.rst) | Repository-root layout and separate release assets |

New report figures connect public-baseline / community examples, fixed-source geometric improvement and counterexamples, and full-set diagnostic distributions. [Figure values and selection records](logs/mechanism_checks.json).

---

## Citation

Please credit [RiO-DETR](https://arxiv.org/abs/2603.09411) for the original detector, and cite this versioned community artifact for its reproduction, analysis and engineering contributions. Citation metadata is in [`CITATION.cff`](CITATION.cff).

<details open>
<summary><b>Code and evidence records</b></summary>

```bibtex
@software{du2026openrio,
  author       = {Du, Zhicheng and Ma, Lan},
  title        = {OpenRiO: A Community Reproduction of RiO-DETR with
                  Practical Accuracy and Efficiency Improvements},
  version      = {0.1.0-preview},
  month        = {10},
  year         = {2026},
  institution  = {Tsinghua Shenzhen International Graduate School,
                  Tsinghua University},
  url          = {https://github.com/dodoxxb/OpenRiO}
}
```
</details>

<details>
<summary><b>Technical report</b> (also on <a href="https://www.researchgate.net/publication/415911327_OpenRiO_A_Community_Reproduction_of_RiO-DETR_with_Practical_Accuracy_and_Efficiency_Improvements">ResearchGate</a>)</summary>

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
</details>

<details>
<summary><b>Original method — please also cite</b></summary>

```bibtex
@article{hu2026rio,
  title   = {RiO-DETR: DETR for Real-time Oriented Object Detection},
  author  = {Hu, Zhangchi and Zhao, Yifan and Peng, Yansong and Sun, Wenzhang
             and Yin, Xiangchen and Chen, Jie and Wu, Peixi and Li, Hebei
             and Wang, Xinghao and Jiang, Dongsheng and others},
  journal = {arXiv preprint arXiv:2603.09411},
  year    = {2026}
}
```
</details>

---

## Acknowledgements

This project is built upon the [official RiO-DETR repository](https://github.com/RicePasteM/RiO-DETR), which itself derives from RT-DETR, D-FINE, DEIM and RT-DETRv4; we retain their upstream attribution headers in `vendor/engine` and in [`NOTICE.txt`](NOTICE.txt). We thank the RiO-DETR authors for releasing their framework and checkpoints, and Gong Cheng and collaborators for the DIOR / DIOR-R datasets. Original method credit is **not** claimed here.

Questions about the reproduction protocol, the evidence records or the reported numbers are welcome in the issue tracker; please include the checkpoint name and the command you ran.

---

## License

* **Source code**: [Apache-2.0](LICENSE), with upstream attribution in [NOTICE.txt](NOTICE.txt).
* **Trained weights and report images**: separate terms, documented in [WEIGHTS_NOTICE.txt](WEIGHTS_NOTICE.txt). An explicit asset-license declaration for the published weight files is maintained by the project owner.
* **Datasets**: DIOR / DIOR-R are distributed under CC BY-NC 4.0 by their authors. Dataset images and annotations are **not** bundled; research-report image excerpts are attributed to DIOR/DIOR-R and excluded from the source-code license. Preserve their noncommercial attribution terms when sharing.

Source-code licensing does not settle all trained-weight or data rights, and no blanket commercial clearance is represented by this archive.

**Contact:** [Zhicheng Du](mailto:duzc21@mails.tsinghua.edu.cn) · [Lan Ma](mailto:malan@sz.tsinghua.edu.cn)

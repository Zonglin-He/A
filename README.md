# STVG OPD

TubeDETR 独立迁移分支的接口、资产版本核对、工程资格检查和新 query 平均评分入口见 [docs/tubedetr.md](docs/tubedetr.md)。本机只负责 TubeDETR 两个跨域方向；该入口不修改下面保留的 TA-STVG 方法配置或旧结果。

用户授权的VidSTG离线诊断、原完整实验结束后的固定128有限调参及所选参数全量流程见 [docs/tubedetr_vid_development.md](docs/tubedetr_vid_development.md)。开发与全量结果、曝光和选择轨迹独立保存。

本地独立项目位于 `stvg-opd/`，先进入该目录运行以下命令。当前机器通过忽略的本地链接使用外层数据、权重、运行库、输出和环境，不重复复制；独立克隆时也可以将这些资产放进同名目录。

唯一主方法：**单 ReferDINO 初始化 + 空间 OPD + Strong 时间分支**。

冻结 TA-STVG 和一个 ReferDINO。ReferDINO 在八个既定位置产生同一条目标轨迹，既用于空间初始化，也用于监督当前视频的临时 content 状态。时间使用 TA-STVG 自身的高分辨率与多层预测融合，再按原物理时间轴进行 MBR 解码。不加载独立 GroundingDINO，不训练模型权重；每条查询独立重置状态及优化器。

## 固定配置

唯一配置文件：[configs/default.json](configs/default.json)。

| 目标数据集 | Source 权重 | 学习率 | 温度 | 步数 | 初始化 γ | 输出 α |
|---|---|---:|---:|---:|---:|---:|
| HC-STVG2（`hc2`） | TA-STVG VidSTG | 0.014 | 0.14 | 40 | 0.5 | 1 |
| VidSTG（`vidstg`） | TA-STVG HC-STVG2 | 0.01 | 0.5 | 30 | 1 | 1 |

两域均为 seed 41、八个 256 维临时状态、32 个双尺度候选、尺度 `(0.05, 0.25)`。本仓库不保留其他活动方法、配置搜索入口或自动实验队列。

## 环境与资产

已核验环境为 Linux、Python 3.11、PyTorch 2.7.0 / CUDA 12.8、FFmpeg 8.0.1。当前机器使用 `.conda/tubedetr`；该环境和本地数据不进入 Git。

```bash
conda create -p .conda/tubedetr python=3.11 ffmpeg=8.0.1
.conda/tubedetr/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.conda/tubedetr/bin/python -m pip install -r requirements.txt
.conda/tubedetr/bin/python -m pip install --target runtime/referdino/python --no-deps -r requirements-teacher.txt
```

按 [资产说明](docs/assets.md) 放置三个模型检查点、Source 的初始化资产、分词器和解析器。ReferDINO 所需 BERT 权重从同一个 ReferDINO 检查点提取，不是额外专家。不会自动下载模型。

```bash
# 仅在首次安装时准备已有资产；--tokenizer 指向 BERT tokenizer 文件目录。
.conda/tubedetr/bin/python -B tools/setup.py --prepare --tokenizer /path/to/bert-tokenizer
# 需要 CUDA toolkit；当前机器已有此算子，不必重复编译。
TORCH_CUDA_ARCH_LIST=12.0 .conda/tubedetr/bin/python -B tools/setup.py --build
.conda/tubedetr/bin/python -B tools/setup.py --check
```

## 统一入口

所有推理与正式评估使用 `python -m stvg`，不依赖旧实验脚本或历史缓存。

```bash
# 单条视频与文本。先填写 examples/request.json 中的路径、哈希和真实帧网格。
.conda/tubedetr/bin/python -B -m stvg predict --request request.json --output outputs/query.npz

# 固定清单逐条推理；完成全部预测后写 sealed.json。
.conda/tubedetr/bin/python -B -m stvg evaluate --manifest data/evaluation/development.jsonl --output outputs/evaluation

# 独立评分进程；只读取封存预测和标注，不运行模型。
.conda/tubedetr/bin/python -B -m stvg score --manifest data/evaluation/development.jsonl --labels data/evaluation/development_labels.jsonl --predictions outputs/evaluation --output outputs/evaluation/metrics.json
```

数据泄漏检查见 [docs/data_integrity.md](docs/data_integrity.md)：未发现当前推理链读取GT，128条离线选参合法，开发曝光与完整划分重叠已单独记录。

数据格式、帧号及评估语义见 [docs/evaluation.md](docs/evaluation.md)。当前本地完整清单为 HC 3,482 条查询 / 237 个父来源、Vid 10,303 条查询 / 732 个父视频；它们只是保留的数据清单，**本次整理没有运行这些完整评估**。清单、标注、视频与权重均被 `.gitignore` 排除。

## 已保存结果

[正式结果表](results/report.md)和[全量逐查询结果](results/evaluation.json)保留原 HC128 / Vid128 的五行评价及正负样本。这两批数据已参与开发和选参，不能当作独立完整测试集。

| 目标数据集 | Source vIoU | 完整方法 vIoU | OPD 相对自身初始化＋Strong |
|---|---:|---:|---:|
| HC | 25.4635% | 29.0669% | +0.4793 pp |
| Vid | 14.5827% | 19.7649% | +0.1009 pp，区间跨零 |

Vid 的严重负例保留，不能把完整方法总增益都归于 OPD。方法定义、证据范围及成本见 [docs/method.md](docs/method.md) 和结果表。

## 工程检查

```bash
.conda/tubedetr/bin/python -B tools/check.py
# 只对已有样例检查迁移一致性；本地需保留原始参考预测。
.conda/tubedetr/bin/python -B tools/check.py --smoke hc2
.conda/tubedetr/bin/python -B tools/check.py --smoke vidstg
```

[迁移核验](results/migration_checks.json)记录核心算法函数一致性，[评分回归](results/score_regression.json)记录原256条预测的五行复核。工程整理不改变方法或参数。

## 项目结构

```text
stvg/          主方法、模型接口、解码和独立评分
configs/       唯一固定配置
tools/         安装与工程检查
scripts/       项目档案维护
examples/      请求格式示例
results/       最终结果与必要核验
external/      两个模型的官方源码及 CUDA ABI 兼容源码
docs/          方法、资产、评估和历史说明
```

本地另有 `data/`、`checkpoints/`、`runtime/`、`outputs/` 和 `.conda/`，均不上传。历史方案从当前工作树移除，仍可通过正常 Git 历史查阅；没有重写或压缩提交历史。见 [docs/history.md](docs/history.md)。

TubeDETR两方向五行消融复用同一完整预测，见 [docs/tubedetr_ablation.md](docs/tubedetr_ablation.md)。

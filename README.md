# STVG 完整框推理

当前主线：**单 ReferDINO ERef 初始化＋完整框传播＋原 BoxFixed 固定评分与精确 DP**。不执行 OPD、参数 TTA 或优化器更新。Strong 保留为已知时间参考，`native224` 与 `tube_replay` 是明确选择的时间模式，后者为实验功能。

主配置为 [configs/default.json](configs/default.json)。HC 使用 VidSTG Source，Vid 使用 HC2 Source；初始化强度分别0.5和1.0，沿用原源域校准。一个 ReferDINO 在相同八个位置的一次调用同时导出所选query的完整框和软mask；mask加权对应，不通过粗组件重建框。空间代码独立于旧实验，缓存只是可选加速，不依赖历史输出目录。

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

先进入本地 `stvg-opd/`。原始输入只含视频、文本及物理采样网格，不包含GT。按 [输入示例](examples/request.json) 填写路径、哈希和帧号；本地数据、权重、运行库和环境不上传。

```bash
# 当前已知参考：BoxFixed + Strong
.conda/tubedetr/bin/python -B -m stvg predict --request request.json --output outputs/query.npz
# 不运行448：BoxFixed + 原生224时间
.conda/tubedetr/bin/python -B -m stvg predict --request request.json --temporal native224 --output outputs/native.npz
# 实验性：全轴辅助轨迹→一次缓存时间重放→最终区间BoxFixed
.conda/tubedetr/bin/python -B -m stvg predict --request request.json --temporal tube_replay --output outputs/tube.npz
# 标签隔离的manifest批量运行；temporal可显式选择同上模式
.conda/tubedetr/bin/python -B -m stvg evaluate --manifest inputs.jsonl --output outputs/evaluation --teacher-cache outputs/expert_cache
# 全部预测封存后，另行离线评分
.conda/tubedetr/bin/python -B -m stvg score --manifest inputs.jsonl --labels labels.jsonl --predictions outputs/evaluation --output outputs/evaluation/metrics.json
```

当前完整框/时间重放接入只验证TA-STVG。历史CG适配源码保留，但新入口不自动运行旧OPD，CG与正式Vid队列保持暂停。其他骨干的时间memory/共享decoder需要各自适配，不能据TA结果推广。

## 当前算法与验证

[方法定义](docs/method.md)说明完整框范围、原评分及两种空间decode视图。`interval`严格恢复原BoxFixed；`full_timeline`辅助轨迹不使用Strong event/active，是明确的方法扩展。时间重放使用实际缓存的motion/text/query/position/padding，逐head/query只重分配同帧合法视觉attention且保持其总量；不是ROI事件打分。固定alpha0.5、框读取上下文1.5，不训练新头。

[HC64空间迁移证明](results/boxfixed_migration.json)记录2240个张量逐位检查及原DP动作、完整框、score和Strong区间。它是工程回归，原BoxFixed的31.3985%完整vIoU是已曝光HC64开发成绩，不是全量独立测试。新时间功能的固定A/B/C/D比较、先封存后GT和成本口径见 [本轮协议](docs/boxfixed_temporal.md)。

本轮[HC64时间验证](results/tube_temporal_hc64/ANALYSIS.md)已完成：Strong/native224/tube_replay/置换的完整vIoU分别31.3985/29.0568/28.8196/29.1586%。轨迹反馈相对native224下降0.2372pp，未达到替代Strong的目标，默认Strong保持。C的缓存重放与两次DP平均0.0810秒；这是额外分支成本，完整分项、缓存命中边界和全部负结果均公开。

```bash
.conda/tubedetr/bin/python -B tools/check.py
# 只在已有私有资产齐备时，重建旧64条空间迁移回归，不做GT评分
.conda/tubedetr/bin/python -B tests/boxfixed_migration.py --run
.conda/tubedetr/bin/python -B scripts/research_archive.py check
```

## 缓存全局／轨迹时间融合验证

已按固定等权完成[HC64融合验证](results/tube_fusion_hc64/ANALYSIS.md)，空间保持原BoxFixed。未适应Source、全局六层、全局＋真实轨迹、全局＋置换的完整vIoU分别26.8938/29.6952/29.7235/29.6977%。完整融合相对Source +2.8297pp [.9075,4.8356]，但相对关闭反馈仅+.0283pp且区间跨零，tIoU略降；固定空间分析亦未建立时间补充优势。Strong仅作为参考，不作为必须超越的门槛。本轮停止该反馈方向，不扫描融合权重。

本次只通过原冻结时间头读取已封存六层hidden，无新RGB/编码器/decoder/专家；缓存执行阶段6.515秒不代表部署耗时。实验脚本为`scripts/tube_fusion/run.py --run`及封存后独立`score.py --run-after-seal`，默认生产配置未自动切换；全部[正负变化](results/tube_fusion_hc64/signed_changes.tsv)、[固定空间分析](results/tube_fusion_hc64/report.md)和[协议](docs/tube_fusion.md)保留。

## 最后两轮时间研究与轻量目标

已完成[归因＋单方案干预＋双向冻结复核](results/interval_pool/ANALYSIS.md)。唯一新方案是12个完整物理区间分布等权混合，保留原精确MBR和BoxFixed；HC64相对旧Global6完整v/t下降3.2209/6.6697pp，两域各32父复核亦分别下降0.5715/2.3750pp、1.0251/1.7805pp。按预注册判据淘汰混合，不扫描第三组权重。

研究保留**原BoxFixed＋Global6旧hull＋原MBR**。其相对未适应Source的完整vIoU点增量在HC32为+4.4253pp（CI跨零）、Vid32为+8.2804pp [3.5570,13.5748]；全划分历史已曝光，只能称冻结父隔离开发复核。生产默认仍Strong，研究读出没有自动注册。完整框传播与六层全局读出的收益，不能写成失败空间时间反馈或OPD的优势。

新混合的Vid后验构造约179.75→5.00ms，但准确率下降，不能以更快替代涨点。两域实际共享capture均3.5968/3.8904秒，包含同一八帧专家，RGB另列；缓存毫秒不等于部署总耗时，首次模型加载和稳态分开。协议见[interval_pool](docs/interval_pool.md)，[时间瓶颈排序](docs/temporal_audit.md)、[原论文核查](docs/temporal_sources.md)、[工程核验](results/interval_pool/engineering.json)及全部匿名signed结果一并保留。

研究入口按锁定顺序为`scripts/interval_pool/development.py`、`diagnose.py`、`intervention.py`、`confirmation.py`和`score.py`；这些是独立研究，禁止覆盖现有sealed目录。`scripts/select_interval_recheck.py`重现无GT名单规则，`audit_interval_pool.py`核对保存态，`summarize_interval_pool.py`只汇总公开封存指标。当前不继续本轮实验，不恢复正式Vid或CG。

## 双向调参与当前全量

新授权的[有限调参](results/boxfixed_tuning_v1/ANALYSIS.md)已完成：HC64实际14配置、Vid64实际17配置，开发最佳有收益，但两方向冻结32父复核均退化，因此全量按预定规则回退原(temperature1,observed.5,guard.1,link.25)，没有将开发最好点冒充稳定改进。全部单因素、邻域、交互和撤回结果及严重损伤保留。

新全量队列使用[显式profile](configs/tuned_boxfixed_v1.json)的原参数完整框＋Global6旧hull＋原MBR，HC1902→Vid10303；生产Strong参考不变，旧formal8405和CG仍暂停。Final/Untuned是alias只算一次，主对照Source。已逐位验收的512MiB RGB/4GiB实际tensor视觉缓存启用；根负责修复，一个Luna max半小时只读巡检。见[运行/恢复/评价协议](docs/tuning_v1.md)，全量运行中不宣称已有完整划分成绩。

## 历史与数据边界

旧OPD源码、参数及完整正负结果均保留，数值配置在 [configs/legacy_opd.json](configs/legacy_opd.json)；旧主线可按Git提交579757a或私有原源码快照恢复。默认流程不再调用其采样/临时状态/训练。历史[HC128/Vid128结果](results/report.md)、[完整框研究](results/box_propagation_hc64/ANALYSIS.md)、[A/B策略](results/evidence_policy_hc64/ANALYSIS.md)、[单点续走](results/correction_value_hc64/ANALYSIS.md)、[承诺策略负结果](results/committed_policy_hc64/ANALYSIS.md)保留原协议，不能改名为新方法成绩。

正式评价口径为HC2 1902查询/227来源、VidSTG10303/732，旧raw3482清单撤销原因见[划分审计](docs/split_audit.md)。HC1902旧方法已评分；正式Vid按用户要求停在8405/10303，CG暂停。不得自动恢复。开发128/64选参与曝光均有记录，不将其视为独立测试；[数据检查](docs/data_integrity.md)与[评价语义](docs/evaluation.md)说明标签隔离及官方指标。

工程目录：`stvg/`正式推理/evidence/spatial/temporal/评价与backbone接口；`configs/`当前及明确标记的历史配置；`scripts/`隔离研究与档案；`tests/`工程验证；`results/`公开匿名指标及证明；`external/`保留上游版权源码。`data/`、`checkpoints/`、`runtime/`、`outputs/`、`.conda/`均为私有本地资产。依赖核查后退出默认入口的历史组件仍可恢复，不删除数据/权重/缓存，不重写Git历史。完整决策见[历史档案](docs/history.md)。

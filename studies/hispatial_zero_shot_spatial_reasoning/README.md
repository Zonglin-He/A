# Training-Free Zero-Shot Spatial Reasoning

本分支研究冻结空间多模态模型在现有 Spatial QA Benchmark 上的表现。使用每道题的原始图像、问题与选项。冻结 MoGe-2 从 RGB 生成 XYZ 和有效性掩码，冻结 HiSpatial-3B 给出答案。目标基准的标签仅用于离线评分，基线使用公开权重直接推理。模型已有空间监督，预训练来源与基准的重叠情况待核查。[HiSpatial 官方实现](https://github.com/microsoft/HiSpatial#inference) · [HiSpatial-3B 权重](https://huggingface.co/lhzzzzzy/HiSpatial-3B)

## 第一阶段

先复现官方冻结基线，按 CV-Bench、3DSRBench、EmbSpatial-Bench 的顺序运行。直接采用官方数据版本、题目及评分约定。为每一条预测保留基准版本、原题 ID、题型、原始回答、解析结果和正确性。[官方评测入口](https://github.com/microsoft/HiSpatial#evaluation)

| 基准 | 官方入口 | 第一阶段的报告重点 |
|---|---|---|
| CV-Bench | `eval/eval_cv_bench.py` | 分开报告 2D 关系和 3D 深度、距离项目 |
| 3DSRBench | `eval/eval_3dsrbench.py` | 记录视角、题型及原题分组，按官方规则聚合同一题的增广行 |
| EmbSpatial-Bench | `eval/eval_emb_spatial.py` | 按空间关系报告成绩，另计无法解析的回答 |

代码提交、模型权重版本、数据版本、解码参数和官方评分入口见 [运行配置](OFFICIAL_BASELINE.json)。CV-Bench 使用完整 3D 与 2D Relation 子集。3DSRBench 按原题分组，组内所有变体均正确才计为答对，分别保留官方 `truncate` 与 `regex` 解析。EmbSpatial-Bench 使用全部原始记录。 [HiSpatial 官方评测](https://github.com/microsoft/HiSpatial/tree/9b0a5718ed0fb3b8bd9d9e0b36b6192bd3e99be1/eval)

## 运行

从仓库根目录运行。`assets` 指向模型缓存，其 `sources/HiSpatial`、`sources/MoGe` 为配置中锁定的源码检出，`weights/hispatial`、`weights/moge` 为对应的权重目录。可用 [现有权重获取工具](../3eed_hispatial_observation_audit_20261121/fetch_models.py) 下载并校验两个模型。评分环境需要 Pillow、PyArrow、Datasets、NumPy、tqdm 和 Requests。本机 GPU 环境使用 PyTorch 2.7.0+cu128、Transformers 4.50.0、OpenCV 4.11.0 及两套官方模型所需依赖。

```bash
study=studies/hispatial_zero_shot_spatial_reasoning
assets=/path/to/model-cache
cpu_python=python  # 或指向已安装评分依赖的解释器
gpu_python=python  # 或指向已安装推理依赖的解释器

"$cpu_python" "$study/download.py" cv_bench 3dsrbench embspatial
for benchmark in cv_bench 3dsrbench embspatial; do
  "$cpu_python" "$study/prepare.py" "$benchmark" || exit
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$gpu_python" "$study/predict.py" "$benchmark" --model-cache "$assets" || exit
  "$cpu_python" "$study/score.py" "$benchmark" --official-source "$assets/sources/HiSpatial" || exit
done
```

输出写入 `outputs/official_baseline_20261121/<benchmark>/`。`predictions.jsonl` 保留原始输出，`scored_rows.jsonl` 记录逐题判分，`errors.jsonl` 保留错误项，`breakdown.csv` 汇总各题型的正确和错误计数。3DSRBench 的 `original_groups.jsonl` 另存原题组结果。预测封存记录含输入、预测和运行配置哈希。评分程序在核验封存后读取标签，向官方评分函数重放原始回答并与逐题计数交叉验证。

GPU 端读取分离的图像、问题、选项和行 ID。每套先用 24 道题核对缓存与批量推理的输出及官方单题调用的一致性，匹配后使用批量推理，出现不匹配时退回单题。两套冻结权重和解码参数在三个评测中共用。

## 官方基线复跑结果

2026 年 11 月 21 日完成 17,176 条预测。三套评测的缓存、批量输出各通过 24/24 条单题对照。各阶段预测封存后，官方评测重放与逐题计数一致。 [运行记录及产物哈希](results/official_baseline_20261121/SUMMARY.json)

| 官方评测项 | 正确 / 评分单位 | 错误 | 本次准确率 | 论文表 2 |
|---|---:|---:|---:|---:|
| CV-Bench 3D | 1,173 / 1,200 | 27 | 97.75% | 97.58% |
| CV-Bench 2D Relation | 620 / 650 | 30 | 95.38% | 95.69% |
| 3DSRBench 原题组，`truncate` 与 `regex` | 1,670 / 2,625 | 955 | 63.62% | 63.81% |
| EmbSpatial-Bench | 2,932 / 3,640 | 708 | 80.55% | 80.71% |

3DSRBench 的 2,625 个原题组共包含 11,686 条变体预测。CV-Bench 的深度题错 11 题，距离题错 16 题。3DSRBench 多物体题错 428/875 组，其中 `multi_object_viewpoint_towards_object` 错 119/175 组。EmbSpatial-Bench 的 `far` 和 `close` 合计错 423 题，占 708 个错误的 59.75%。 [完整分项结果](results/official_baseline_20261121/breakdown.csv) · [HiSpatial 论文表 2](https://arxiv.org/html/2603.25411v1)

## 失败归因与独立验证

- 先按对象定位、空间几何读取和关系推理分析错误。几何输入诊断可在开发样本上比较原始 MoGe、错配 XYZ 和仅 RGB 条件。有真值深度时，将它作为单独标明的诊断条件。
- 用开发数据形成无训练的推理方法。确定方法后在未参与选型的评测范围复核，与官方基线及同预算对照比较，并记录模型调用次数、运行时间和显存。

[单区域语义核验与 Oracle 报告](REGION_VERIFICATION_ZH.md)记录 EmbSpatial 近／远题上物理区域匹配、同区矩形读出和联合绑定的配对结果。

[SpatialRGPT 首交等价高度报告](HEIGHT_TRANSFER_ZH.md)记录官方 Region-Given 的 Height／Width 对照、射线保持与同模型问答。

## 分支状态

本次基线复跑在 `research/hispatial-zero-shot-spatial-reasoning` 登记。分支从 `Zonglin-He/A` 的 `main` 提交 `5a39e13ff2ee073c137e378a5b5780ac3fd7cce2` 建立。 [分支起点](https://github.com/Zonglin-He/A/commit/5a39e13ff2ee073c137e378a5b5780ac3fd7cce2)

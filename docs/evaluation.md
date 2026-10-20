# 推理与正式评估

`python -m stvg predict` 接收一个 JSON 对象：`dataset`（`hc2` 或 `vidstg`）、确定性采样 `key`、不含 GT 的 `metadata`。格式见 `examples/request.json`。`metadata` 至少提供 caption、index、width、height、video_path、video_sha256 和严格递增的真实 frame_ids；保留数据集物理帧号，不按预测结果改变支持。

`frame_ids` 必须来自预先固定的数据清单。HC 使用原官方 FFmpeg 默认输出时序；Vid 使用原始零基帧索引的 select/vsync=0 解码。两种路径不能互换，也不自动截断、补帧或改变端点。FFmpeg 版本及指纹在资产登记中固定。

评估清单为 JSONL，每条含 `id`、`dataset`、`key`、`metadata`；id 仅用字母、数字、下划线和短横线且全局唯一。key 决定已冻结的随机流，复现旧结果时必须保持不变。多个查询可共享父视频，但不能当作多个独立父视频统计。

标注单独保存为 JSONL，每条含相同 `id`、`dataset`、`parent`、`interval` 和 `boxes`。interval 使用对应数据集既有的物理时间约定；boxes 为按物理帧号索引的原图像素 xyxy。不得为了使指标更高而统一改写 HC 与 Vid 的端点语义。旧数据转换已在本地保留，原始官方标注另存 `data/annotations/`，不进入版本控制。

`evaluate` 不接受标注路径，不调用评分器。所有预测完成后才生成 sealed.json，绑定清单与每份输出的 SHA256；已封存目录不能覆盖。`score` 验证这些绑定后在独立 CPU 进程中读取 GT；不运行 Source、ReferDINO 或适应。

五行固定为 Source、Spatial、Temporal、Full、WithoutOPD。空间评估保留固定实际特征节点，以及八个观察位置之外的实际节点；它们与密集插值评测帧分开。官方 vIoU 使用官方插值和数据集端点约定，并与独立 DenseTube 交叉核对。主表按查询等权平均（`query_main_table`），另保留每父视频内先平均查询、再对父视频等权平均的历史聚合（`means`、`parent_main_table`）。所有正负、无更新、零时间交集样本都保留。`query_paired` 是主表查询均值差及父视频 cluster bootstrap 区间；`paired` 保留父视频等权的差值与区间，两者不混用。bootstrap seed 固定为20261020，重采样10,000次。

本地 `development.jsonl` 为已有256条调参样本；`hc2.jsonl` 和 `vidstg.jsonl` 是保留的完整可用清单，尚未因此获得新成绩。对于新完整评估，应直接冻结清单后运行，不从评价结果选择视频或切换参数。

数据隔离、开发重叠与推理/评分边界见 [数据审计](data_integrity.md)。128条离线开发选参符合设定，不定义为数据泄漏；完整划分中已参与开发的父视频需如实披露。

## 正式评分字段与数据登记

五行主表均报告 `m_vIoU`、`vIoU@0.3`、`vIoU@0.5`、`m_tIoU`、`m_sIoU` 和 `m_sIoU_unobserved_actual`。官方两数据集 evaluator 使用 **严格大于**0.3／0.5（TA-STVG的 `datasets/evaluation/hcstvg_eval.py:110`、`vidstg_eval.py:112`）；逐查询对应保存为 `vIoU_at_03/05`。原 `vIoU_ge_03/05` 保留为附加的大于等于命中字段，不能替代官方主表。所有IoU、均值和命中比例保存为0–1，展示百分数时统一乘100。`m_sIoU` 使用原官方密集固定GT帧集合的空间指标 `s`，经既有插值评估且与预测时间区间无关；固定实际特征节点指标另存为 `m_sIoU_fixed_actual`。未观察实际节点指标排除既定八个观察位置，并不冒称密集完整空间指标。额外记录观察节点 `m_sIoU_observed_actual`、`tIoU_ge_05`、`tIoU_eq_0` 与 `boundary_MAE_seconds`。保留逐查询原始 `v/t/s`、各实际节点空间字段和物理帧单位 `boundary_MAE` 以兼容原结果。

边界误差秒数严格等于逐查询物理帧边界平均绝对误差除以该输入清单的显式 `metadata.fps`，然后才做均值。不得用 `used_segment` 长度、GT事件时长或假设帧率替代。现有完整HC 3,482条和Vid 10,303条输入的fps均为有限正值；若后续输入缺失或无效，该查询秒数记为null，并通过 `query_valid_counts` 和 `provenance.datasets.*.missing_or_invalid_fps_queries` 披露有效分母。没有对应标注的实际节点组同样记为null，不伪造零分。

每次评分保存完整输入清单SHA256、当前开发清单SHA256及查询／父视频数。开发曝光仅按无标签清单中的 `metadata.source` 父视频标识匹配，生成 `current_development_parent` 与 `not_current_development_parent` 两个事后报告层；不据此改变预测、参数或主表。HC当前开发128父均包含在完整237父中，剩109父；Vid当前开发128父均包含在完整732父中，剩604父。后者只表示未在当前256条开发清单中，不能自动解释为历史未曝光。离线开发选参符合本项目设定，不定义为在线GT泄漏。

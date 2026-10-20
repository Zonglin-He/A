# 推理与正式评估

`python -m stvg predict` 接收一个 JSON 对象：`dataset`（`hc2` 或 `vidstg`）、确定性采样 `key`、不含 GT 的 `metadata`。格式见 `examples/request.json`。`metadata` 至少提供 caption、index、width、height、video_path、video_sha256 和严格递增的真实 frame_ids；保留数据集物理帧号，不按预测结果改变支持。

`frame_ids` 必须来自预先固定的数据清单。HC 使用原官方 FFmpeg 默认输出时序；Vid 使用原始零基帧索引的 select/vsync=0 解码。两种路径不能互换，也不自动截断、补帧或改变端点。FFmpeg 版本及指纹在资产登记中固定。

评估清单为 JSONL，每条含 `id`、`dataset`、`key`、`metadata`；id 仅用字母、数字、下划线和短横线且全局唯一。key 决定已冻结的随机流，复现旧结果时必须保持不变。多个查询可共享父视频，但不能当作多个独立父视频统计。

标注单独保存为 JSONL，每条含相同 `id`、`dataset`、`parent`、`interval` 和 `boxes`。interval 使用对应数据集既有的物理时间约定；boxes 为按物理帧号索引的原图像素 xyxy。不得为了使指标更高而统一改写 HC 与 Vid 的端点语义。旧数据转换已在本地保留，原始官方标注另存 `data/annotations/`，不进入版本控制。

`evaluate` 不接受标注路径，不调用评分器。所有预测完成后才生成 sealed.json，绑定清单与每份输出的 SHA256；已封存目录不能覆盖。`score` 验证这些绑定后在独立 CPU 进程中读取 GT；不运行 Source、ReferDINO 或适应。

五行固定为 Source、Spatial、Temporal、Full、WithoutOPD。空间评估保留固定实际特征节点，以及八个观察位置之外的实际节点；它们与密集插值评测帧分开。官方 vIoU 使用官方插值和数据集端点约定，并与独立 DenseTube 交叉核对。每父视频内先平均查询，再对父视频平均；所有正负、无更新、零时间交集样本都保留。配对区间以父视频为单位、固定 bootstrap seed 20261020。

本地 `development.jsonl` 为已有256条调参样本；`hc2.jsonl` 和 `vidstg.jsonl` 是保留的完整可用清单，尚未因此获得新成绩。对于新完整评估，应直接冻结清单后运行，不从评价结果选择视频或切换参数。

数据隔离、开发重叠与推理/评分边界见 [数据审计](data_integrity.md)。128条离线开发选参符合设定，不定义为数据泄漏；完整划分中已参与开发的父视频需如实披露。

# 官方评测标注核对与纠正

2026-10-20，用户指出HC2只有1900多条。根进程与两个只读审计子代理独立核对原始成员、作者公开加工标注、查询展开与评分标签，未运行新任务评分或选参。结果见 [机器可读审计](../results/formal/split_audit.json)。

## HC2：原始3482误当成作者评测1902

本地旧输入取自 `data/hcstvg2_official_metadata/val_v2.json`。整理时原样复制为 `data/annotations/hc2_val.json`，没有与TA-STVG作者实际发布的评测成员核对。3482条均为不同片段的一条查询，不是当前生成器将一个视频重复展开。

TA-STVG官方README链接的数据包，固定revision `38e89a5895346d77cba87b0273f9ede95874befc`，其中 `data/hc-stvg2/annos/test.json` 实为1902条、227电影父来源，训练文件为9586条。论文§4描述验证集2000条，应区分论文概述规模和实际公开包；本次正式评价使用作者代码包的实际1902条。

| 集合核对 | 数量 |
|---|---:|
| 原始raw名单 | 3482 |
| 当前官方包评测成员，全部在raw名单中 | 1902 |
| raw额外、但不在官方包train/test的成员 | 1580 |
| raw与官方包train的相同片段 | 0 |
| 两版本共有、但bbox数值不同的查询 | 534 |
| 对应改变的帧框 | 78454 |

额外1580条的直接来源已经确定是原始val_v2名单；其精确原始下载回执及作者筛除理由未保留，不能继续猜测。它们不是训练集混入，也不能整体说成坏样本。空文本、空框、基本时间合法性和统一严格几何过滤均不能复现该筛选，官方保留成员本身也存在越界框。

当前正式HC清单及标签同步采用1902条和对应官方bbox，不能只改计数后保留旧框。原295条partial与旧计划完整保留；其中162条通过完全相同的输入、采样key、模型代码指纹、预测字节与逐query回执核验，可复用预测。其余133条不进入此次正式结果，原始产物不删除。

## VidSTG：10303名单正确，评分端点转换有误

本地原始文本标注与 [VidSTG官方仓库](https://github.com/Guaranteer/VidSTG-Dataset/blob/master/annotations/test_annotations.json) JSON内容完全相同：4610个关系记录，含4610条描述和5693条问句，共10303个query、732个父视频。与TA-STVG公开加工 `data/vidstg/annos/test.json` 逐query比较，成员、文本、对象、输入窗口、尺寸和fps全部一致。

从原始VidOR标注独立重建的2,727,412个官方框全部一致；包括1549次官方预处理所使用的终点框沿用上一帧规则。旧独立标签转换有两项错误，已用官方加工标注修正：

- 8558条少了轨迹闭端点最后一帧的框，补回该帧。
- 另外1745条的GT区间终点没有按官方loader限制到used_segment.end；全部仅相差1帧。

区间指标仍沿用官方半开range，GT框字典则包含闭端点，二者不统一改成end+1。查询数量无需减少，也不因无更新或零交集剔除样本。

## 执行与证据边界

旧raw3482计划已停止，原记录保留于本地 `outputs/superseded/formal_raw3482_20261020/`。新清单和离线标签哈希一起锁定，不放宽原resume哈希、不混用代码指纹、不重新选参。复用预测的成本保留原实测值并单列复用数，不冒充本次重算耗时。

输入采样继续采用本轮已经冻结的物理帧网格，Source与Ours配对相同。这些网格与TA官方测试loader的128点等距采样不同；这属于明确披露的推理协议差异，不是查询名单扩张，不能将跨域结果冒充论文域内推理复现。本次不通过改网格改变已冻结方法。

HC128/Vid128离线选参仍是授权开发设定，不算数据泄漏。历史256条开发成绩按旧raw标注口径保留，不改写数值或将其改称当前官方划分成绩。当前HC开发128中67条直接属于官方1902，61条不在该正式名单；按父来源计，122个开发父与正式227父重叠，对应正式1011条，剩余891条/105父也不因此保证历史未曝光。

来源：[TA-STVG官方数据准备](https://github.com/HengLan/TA-STVG#dataset-preparation)、[固定版本数据包](https://huggingface.co/Gstar666/TASTVG/resolve/38e89a5895346d77cba87b0273f9ede95874befc/data.tar)、[TA-STVG论文§4](https://arxiv.org/html/2502.11168v1#S4)、[HC-STVG数据提供方](https://github.com/tzhhhh123/HC-STVG)。不上传原始标注、媒体、查询正文或权重。

## 离线标注转换入口

`tools/prepare_labels.py` 只接收本地官方JSON与既有无标签输入清单，输出完整官方成员和独立评分标签。它不导入模型、不评分，不将GT写入metadata。新输出路径必须不存在，避免覆盖已经锁定的清单。示例：

```bash
python -B tools/prepare_labels.py --dataset vidstg \
  --annotations data/annotations/vidstg_tastvg_eval.json \
  --base-manifest data/evaluation/vidstg.jsonl \
  --manifest outputs/reproduction/vidstg.jsonl \
  --labels outputs/reproduction/vidstg_labels.jsonl
```

HC使用同一入口、`--dataset hc2`及官方HC评测JSON。原raw全量输入可以作为base，但只按官方成员过滤，不按标签质量或成绩过滤。两个方向转换结果已逐query与独立审计标签核对一致；JSON序列化顺序可不同，正式任务仍绑定实际使用文件的字节哈希。新增合成测试验证成员筛选、修订框、闭端点与区间末端裁剪，不使用真实GT计算新成绩。

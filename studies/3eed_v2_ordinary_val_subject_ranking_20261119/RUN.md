# R183 复现

本轮使用 [R182 冻结头与运行环境](../3eed_v2_fixed_slate_subject_ranking_20261119/RUN.md) 和 [R180 的主体名单](../3eed_v2_subject_retrieval_20261119/RUN.md)。运行从已有的三份 R182 `generic` 参数加载，等权训练的选定轮次和哈希记录在本研究的 `PROTOCOL.json`。本轮只运行推理和评分。

从项目根目录执行。令 `study` 为本研究目录。依照 R96 保留下来的本地 `crossplatform/round96/data_view` 取得普通 Validation 清单。权限受保护的清单和样本不参与清单构建。源域所需资产由既有运行环境提供。

```bash
python "$study/scope183.py"
flock research/GPU_RESOURCE.lock python "$study/predict183.py" parity
flock research/GPU_RESOURCE.lock python "$study/predict183.py" run
python "$study/score183.py"
flock research/GPU_RESOURCE.lock python "$study/verify183.py"
```

清单封存后，正式预测核对注册哈希。加速路径略去检索头用不到的特征，只缓存确定性的语言分析。等价性检查逐值比较了候选、分数、特征和预测。预测每 128 条封存一个块。中断后使用同一命令校验并接续已封存的块，目标评分仅在六方向完整封存后执行。

真值、点云、图像、原始文本、私有清单、框和模型权重由本地环境提供，公开目录只包含实现、协议、核验与匿名标量。

> 历史2026-09-27快照。运行状态和optimizer解释已被2026-09-28更正；不要把下文旧running当当前状态。

# A — STVG adaptation research code

用于时空视频定位（STVG）的研究代码，包含当前的 **DESTA-3D v2**、已有 **DeCoTA C+D 精简实现**，以及这些入口实际依赖的历史工具。

本仓库包含代码快照和已完成实验的汇总结果。数据集、标注、视频、backbone/adapter 权重、缓存、逐样本预测和个人研究记录均不包含在内。

**2026-09-27 更新：** [实现修复与已完成结果](../results/desta3d_v2/2026-09-27/README.md)，包括共享 reference 的 198-query 源验证对照、源训练 E0 回读、真实校准梯度和 8-parent 梯度尺度诊断。完整源训练仍在进行，目标 v2 TTA 尚未开始。

## 主要入口

| 内容 | 代码 |
|---|---|
| DESTA-3D v2：共享 THW stem、query-conditioned 双 reader、独立残差 | [`vg_tta/desta3d_v2.py`](../vg_tta/desta3d_v2.py) |
| 冻结 PTD 的 event → spatial 两次解码、caption token 提取 | [`vg_tta/desta3d_v2_ptd.py`](../vg_tta/desta3d_v2_ptd.py) |
| 共享 event reference/time、spatial 新 KV、官方分块缓存路径 | [`vg_tta/desta3d_v2_shared_reference_cached.py`](../vg_tta/desta3d_v2_shared_reference_cached.py) |
| 源监督的 temporal/spatial token mask | [`vg_tta/desta3d_v2_source.py`](../vg_tta/desta3d_v2_source.py) |
| Evidence loss、分组优化器、warmup/cosine | [`vg_tta/desta3d_v2_training.py`](../vg_tta/desta3d_v2_training.py) |
| 真实模型工程验收入口 | [`scripts/desta3d_v2_p0.py`](../scripts/desta3d_v2_p0.py) |
| 四臂 source fit、断点续跑、封存与选态 | [`scripts/desta3d_v2_source_fit.py`](../scripts/desta3d_v2_source_fit.py) |
| 保持训练配置的续跑与每轮不可变权重快照 | [`scripts/desta3d_v2_source_continue.py`](../scripts/desta3d_v2_source_continue.py) |
| Pre-gate TTA 目标与 FiLM/LN 更新组 | [`vg_tta/desta3d_v2_tta_objective.py`](../vg_tta/desta3d_v2_tta_objective.py) |
| 共享 reference 的真实 pilot / 完整源对照入口 | [`scripts/desta3d_v2_reference_audit_cached_v3.py`](../scripts/desta3d_v2_reference_audit_cached_v3.py) |
| 封存后独立几何评分、父源 CI 与负尾 | [`scripts/score_desta3d_v2_reference_audit.py`](../scripts/score_desta3d_v2_reference_audit.py) |
| 已有 DeCoTA 精简运行模块 | [`methods/decota_final_simplified_v1/`](../methods/decota_final_simplified_v1/) |
| 单次 DeCoTA 预测入口 | [`scripts/predict_decota.py`](../scripts/predict_decota.py) |

`scripts/` 中保留递归导入依赖，因而包含若干旧实验、oracle 诊断和 teacher 接口。这些文件的存在不表示它们属于 v2 当前方法，也不表示它们已建立有效性；当前 v2 没有启用新教师或 OPD 训练。

## DESTA-3D v2 计算图

```text
Frozen PTD visual merger [T,H,W,2560] + frozen caption token sequence
                          |
                learned projection / shared THW stem
                   /                         \
        spatial text pooling           event text pooling
           reader-internal FiLM       reader-internal FiLM
                   |                         |
             spatial reader              event reader
                   |                         |
       referent occupancy M(t,x,y)      frame event presence a(t)
                   |                         |
          independent gated V_S       independent gated V_E
                   |                         |
                   |                    PTD temporal pass
                   |                         | reference tokens + I*
                   +-------- PTD spatial pass (fresh KV) ------>
                                     full tube (I*, B*)
```

两个 branch 不做平均后统一注入。归一化是每个 THW cell 的 channel-only LayerNorm；event head 使用平滑空间池化后的独立帧级输出。初始 gate 为 `sigmoid(-6)`，out projection 为非零小方差初始化。文本 pooling 是否学出对象/动作语义分工仍需实验验证。

`dF/dt` 和 dilation pyramid 是可选代码，本快照的 source 配置中 `p1_enabled=false`。约 2.49M 个 dual adapter 参数。最初 FiLM/LN/gates 集合为 66,818 参数；本次真实 pre-gate TTA 目标没有到 gate 的梯度路径，因此冻结两个 gate，实际校准范围为 **66,816 个 FiLM/LN 参数**。

共享 reference 路径是新增的独立审计入口；原四臂源训练仍保留已锁定的独立-reference 验证路径，没有热改历史训练。`desta3d_v2_shared_reference.py` 保留 prefix 校验工具和首次 whole-prefix 尝试；该尝试未通过真实 zero-gate 框等价检查。实际通过验收的解码入口是 `_cached.py`，不要把旧 whole-prefix decode 当作通过验收的实现。

## 无数据 CPU 使用

以下示例和测试只验证模块与训练接口，不会下载模型、访问标注或启动 GPU 实验。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
python examples/desta3d_v2_cpu.py
python -m pytest -q tests/test_desta3d_v2.py tests/test_desta3d_v2_training.py tests/test_desta3d_v2_tta_objective.py
```

CPU 模块示例使用合成特征，不是视频定位 demo，也不代表任务精度。

## 完整实验依赖与状态

源训练配方见 [`configs/desta3d_v2_source.json`](../configs/desta3d_v2_source.json)，设计记录见 [`protocols/desta3d_v2.md`](../protocols/desta3d_v2.md)。入口保留原研究中的严格配置与完整性检查：需要自行准备官方依赖、权重、合法数据及原格式 manifest。仅克隆本仓库不能直接复现历史全量实验。

截至 2026-09-27 此次快照：v2 的初始 P0、源 E0、共享 reference 的 4-query 真实接口与 198-query 源验证对照均已完成；四臂源训练仍在进行，尚没有最终选态的完整 v2 source-fit 效用结论，目标 TTA 尚未开始。612 条 CPU mask 检查不是 612 次模型反向传播。真实 pre-gate 一步校准改变了坐标 logits，但该单例 argmax 框未变；工程检查通过不等同于精度提升。

已有 v1 与 DeCoTA 的结论不自动转移到 v2。训练修复的各臂共用数据曝光，但优化步数、PTD CE 次数和阶段不同，不能把组合配方的差异归因于单一 trick。数据边界、评价和当前局限见 [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)。

第三方项目和固定版本见 [`docs/DEPENDENCIES.md`](../docs/DEPENDENCIES.md)。出版副本的路径脱敏及源码哈希见 [`docs/PUBLICATION_MANIFEST.json`](../docs/PUBLICATION_MANIFEST.json)。

本次增量源码映射见 [`docs/PUBLICATION_UPDATE_20260927.json`](../docs/PUBLICATION_UPDATE_20260927.json)，选态与校准接口澄清见 [`configs/desta3d_v2_review_amendment.json`](../configs/desta3d_v2_review_amendment.json)。

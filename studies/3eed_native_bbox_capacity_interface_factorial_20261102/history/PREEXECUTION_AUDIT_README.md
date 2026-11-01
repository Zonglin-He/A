# R45独立CPU审计/评分草稿，当前未执行

紧凑复用已验证的独立R39 bbox parser/center读出、R40向量化rectIoU/readout与R42 CPU processor、seal/hash/npz工具。复用的是独立实现，不调用R45生产parser/readout来认证自己。代码只做AST检查；实际执行前必须等全208回答/21臂预测seal、root绑定seal授权并协调CPU所有权。

`audit45_draft.py` 无任务GT读者：重构208个raw/token/input bindings、原R39逐条prompt/RGB/grid/count及CPU input-ID/pixel tensor hash；严格分开parsed、null、invalid_box、parse_failure，不交换/裁剪/修raw坐标。复算center/rectIoU×full/old共832条读出、4368个固定框，要求17缓存臂query/box exact。标准rectIoU使用全部原detector库存，最大空组不转次优；tie/全0/坏库存UNKNOWN。复核主机/GPU资源回执与R45固定208调用预算，0 toy。

同一预封24记录在audit后join四新臂，保留18明确/6歧义及prior_seen、非专家声明。不新增看图、改annotation或用IoU筛feedback。bbox中心或矩形匹配只能说明与现有mask记录的对应，不能单独认证真实主体语义。

`score45_draft.py` 在audit PASS后才加载授权A208 metadata GT，独立计算21×208=4368个IoU，复现17旧控制3536个任务值。输出整体/逐平台mIoU、Acc25/50、unknown/全部有序负比较、场景效果；主2×2容量×point/bbox-center交互和secondary rectIoU交互分别报告，不混选。身份表只附封存后任务值，不能救总体。

primary center未超过旧G时可按资源门停止固定主配置；这不否定全部7B/native接口。预登记secondary若有效保留发展候选价值，但不反选primary或自动接TTA；确认要另锁。小样本既有场景不输出广义显著性CI。

root需要创建 `research/round45/experiment/POSTSEAL_EVALUATION_AUTHORIZATION.json`：

```json
{
  "independent_CPU_audit_authorized": true,
  "A208_task_GT_scoring_authorized": true,
  "prediction_seal_sha256": "完成后的R45 SEALED.json SHA256"
}
```

root可先仅给audit授权，随后再将GT评分授权置true。CPU必须外部协调，因为night_guard的cpu receipt本身不是互斥锁。未来授权后的顺序为：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
crossplatform/env/bin/python tools/night_guard.py --resource cpu -- \
crossplatform/env/bin/python research/round45/independent_audit/audit45_draft.py

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
crossplatform/env/bin/python tools/night_guard.py --resource cpu -- \
crossplatform/env/bin/python research/round45/independent_audit/score45_draft.py
```

以上当前均未运行。无生成模型、GPU、优化器、新VLM调用或FINAL/U/T入口；独立权重证明复用绑定的下载和loader SHA回执，不二次读取16.6GB权重。

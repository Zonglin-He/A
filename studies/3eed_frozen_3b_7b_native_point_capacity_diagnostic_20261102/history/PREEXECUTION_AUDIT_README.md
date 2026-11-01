# R42 独立CPU审计与评分草稿：当前未运行

此目录不改root已审核的R42 runner/PLAN/IMPLEMENTATION。当前仅做AST静态检查，没有加载processor/7B模型、读取真实媒体或任务GT，也没有推理/评分。后续须预测全体seal、root授权、协调CPU所有权后才可执行。

`audit42_draft.py` 不读取任务GT。它独立解析全部216份原回答，核token/raw/binding/hash及持久化mtime顺序，CPU重构216份processor输入，核原RGB/prompt/grid/token count及新增token IDs/pixels hash；不用生成模型或GPU。重建208×full/old的mask→组→原source代表读出，核1664个固定框和所有缓存对照，复核8个方块命中/尺度响应及资源记录。几何点预处理不重跑，仍清楚限定为对封存support矩阵及unique阈值的独立重构；16.6GB权重校验引用绑定的download/loader证据，不重复读取所有权重。

同一24份预封身份记录仅在全体预测封存后join：18明确、6歧义，保留prior_seen与非专家声明，不重新看图、不改annotation、不用IoU纠正身份。先输出GT-free的identity/mapping表。这里的mask命中不能独立证明语义正确，也不能按小子集反选7B。

`score42_draft.py` 只有在CPU审计PASS且root另行授权后才读取A208各自的metadata GT。独立重算208×8=1664个任务值，对六个旧缓存臂要求复现，报告每平台/平衡mIoU、Acc25/50、相对source/同接口3B的改善恶化及query变化、全部有序负比较、场景与unknown覆盖。无跨场景CI/显著性声称；18/6身份记录只附加封存后IoU诊断，不变训练或反馈。完全不读取U/T/FINAL专属媒体/标签。

root完成预测seal并准备执行时需创建 `research/round42/experiment/POSTSEAL_EVALUATION_AUTHORIZATION.json`：

```json
{
  "independent_CPU_audit_authorized": true,
  "A208_task_GT_scoring_authorized": true,
  "prediction_seal_sha256": "R42预测SEALED.json的SHA256"
}
```

root也可先只给audit授权、将GT字段置false，审计PASS后再正式授权评分。两个脚本都先检查seal绑定，评分还检查审计receipt的同一seal。CPU所有权仍须先与其他代理确认，`night_guard --resource cpu`不提供互斥锁。

未来明确授权后的顺序为：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
crossplatform/env/bin/python tools/night_guard.py --resource cpu -- \
crossplatform/env/bin/python research/round42/independent_audit/audit42_draft.py

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
crossplatform/env/bin/python tools/night_guard.py --resource cpu -- \
crossplatform/env/bin/python research/round42/independent_audit/score42_draft.py
```

这些命令当前没有运行。执行前由reportagent/root独立审查草稿；如runner因技术原因改了字段、目录或seal协议，应保留草稿历史并重新绑定，不为通过审计放松数值或科学门。

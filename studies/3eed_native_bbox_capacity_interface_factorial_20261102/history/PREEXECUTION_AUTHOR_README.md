# R45 实现交接：待root审核与GPU授权

本轮补齐固定capacity(3B/7B)×native接口(point/bbox)的7B bbox单元；没有训练或TTA。正式 `experiment/PLAN_LOCK.json` 和 `IMPLEMENTATION_LOCK.json` 已绑定代码、输入seal、模型文件hash，当前没有 `EXECUTION_AUTHORIZATION.json`，runner在模型加载前拒绝无授权执行。

`crossplatform/round45/run45.py` 紧凑复用R42严格本地BF16/SDPA加载、资源计时、token先保存与decoded raw立即atomic保存、原3B CPU processor输入重构。只替换为R39原 `messages_bbox`/`parse_bbox`/center读出和R40原标准rectIoU关联/读出；没有新提示、解析放宽或单位校准。

新增208次bbox生成、0 toy，两条原预检复用208；全部R40十五臂与R42两条7B point缓存不重跑，追加center/rectIoU×full/old四臂，共21臂。完整/旧映射、source最高分组代表和三维框不变。原始回答存盘后才解析或附加hash，后处理失败不得自动重新生成。

纯CPU合成合同检查已经PASS：raw在注入hash失败后保留；singleton/label安全解析；非数/多位置/坏JSON拒绝；反边界与越界不修；逐轴center映射；全部detector库存先唯一正max IoU，最大者组为空时不能改选有正重叠的次优；精确tie/全0 UNKNOWN。没有加载processor、模型、真实媒体、GT或GPU。

center/full为预指定primary，rectIoU/full为secondary，oldsubset为关联消融。主臂不胜旧G可按本夜资源门停这个固定主配置、不接TTA；这不科学否定全部7B/native接口族。若预登记secondary有效，必须保留发展候选价值，不在R45反选primary，确认需另行锁定。旧草案较宽的停止表述已保留在 `review/MECHANISM_CARD_ORIGINAL_DRAFT.md`，澄清及新hash记录在 `experiment/RESOURCE_SCOPE_CLARIFICATION.json`。

root与独立代理审核后，执行授权文件需绑定当前PLAN、IMPLEMENTATION及完整下载manifest：

```json
{
  "model_GPU_execution_authorized": true,
  "plan_sha256": "当前PLAN_LOCK SHA256",
  "implementation_sha256": "当前IMPLEMENTATION_LOCK SHA256",
  "complete_download_manifest_sha256": "21c56f351bf1d6bf002f94990a6fe504e63308ecd668d8b291744659c0a2a9b8"
}
```

只有root明确释放GPU并另行授权后才可运行：

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
crossplatform/env/bin/python tools/night_guard.py --resource gpu -- \
crossplatform/env/bin/python crossplatform/round45/run45.py
```

本次没有执行该命令。模型阶段（含加载）最大2700秒并受全局09:30截止约束；输入/权重核验耗时另计。所有208回答、21臂原框、unknown/完整IoU关联中间量与资源回执seal后，再另行授权独立CPU审计与4368个任务值评分。A208全为暴露开发数据；不读FINAL/U/T，不按身份子集救总体。

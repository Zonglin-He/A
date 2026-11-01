# R42 实现交接：当前禁止执行模型

本轮只实现同族3B→7B的冻结native-point容量对照。`experiment/PLAN_LOCK.json`、`IMPLEMENTATION_LOCK.json` 为待root审核的正式锁草稿。没有 `EXECUTION_AUTHORIZATION.json`，runner会在模型加载前拒绝执行；当前没有任何7B加载、生成或任务评分。

实现文件：

- `crossplatform/round42/run42.py`：同R37完整文字/未标记单原图point提示，原8个固定正方形+208个固定A观测，共216调用；两条按原PREFLIGHT12规范顺序各平台第一条作为主集前两条复用。
- `common42.py`：授权/完整权重校验、原子raw与token持久化、资源计时、原R38兼容parser及原R37映射。
- `checks42.py`：纯CPU合成测试，不加载模型/processor/真实媒体/GT，不评分。

原生成核心拆分的唯一必要原因是持久化顺序：生成后先atomic保存token IDs；decode成功后立即atomic保存原始字符串；之后才解析、派生hash、资源注释和映射。因此，即使后处理失败，原回答仍保留。若decode本身失败，已有token IDs可审计。没有自动重答，已有输出目录拒绝覆盖；技术失败须单独root处理。

为确保“同输入”不是只比token数量，每条还用原3B目录的processor/tokenizer做CPU重构，与7B所有输入tensor逐项exact比较（包括token IDs、pixels、grid与mask），绝不加载第二个模型。旧缓存只有历史RGB/prompt/grid/count等证据，没有保存历史完整token向量，计划对此明确披露。功能文件hash相同和CPU重构共同保证接口可比，不能假装存在未保存的历史token缓存。

资源计量：权重与旧封存输入的SHA核验耗时在 `INPUT_WEIGHT_VERIFY_COST.json` 单列，不计入2700秒模型加载/生成墙时。2700秒从进入模型加载前开始，并受夜间全局截止约束。主机MemAvailable≥20GiB只是资源门；低于时保存资源停止回执，不能记为7B科研负结果。原正方形指标直接查像素前，必须断言原尺寸=实际grid尺寸=封存fixture尺寸；命中率不改变坐标制或运行门。

root审核后、完整下载且5分片和必要metadata核验后，需锁定最终PLAN/IMPLEMENTATION，重新运行纯CPU checks使其绑定最终implementation hash，再创建执行授权文件，字段如下：

```json
{
  "model_GPU_execution_authorized": true,
  "plan_sha256": "最终PLAN_LOCK文件SHA256",
  "implementation_sha256": "最终IMPLEMENTATION_LOCK文件SHA256",
  "complete_download_manifest_sha256": "已完成下载清单SHA256"
}
```

只有在单卡资源已由root明确释放时，才可由root执行：

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
crossplatform/env/bin/python tools/night_guard.py --resource gpu -- \
crossplatform/env/bin/python crossplatform/round42/run42.py
```

该命令在本次交接中没有运行。runner自身不读取任务标签、不评分、不调参、不下载、不安装、不生成新mask/框，不执行TTA。全部216回答、208八臂预测和资源记录封存后，才由另行授权的独立CPU审计和评分处理。预测目录将为 `crossplatform/round42/pointA208_attempt001`，当前不存在。FINAL始终没有入口。

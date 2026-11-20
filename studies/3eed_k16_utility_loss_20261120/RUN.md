# 在固定 K16 上复核五种排序损失

工作目录为研究工作区根目录。以下命令复用 R182 的源域特征、R183 的普通验证清单、三个源模型及其冻结权重。原始观测、候选坐标、权重和逐例特征保存在私有工作区。当前精简 v1 的入口由 `crossplatform/CURRENT_METHOD.json` 指定。

公开导出将自有脚本放在各研究目录的 `own_code/` 下。复现时将这些脚本放回各自研究目录的根目录，并按 R182、R183 的说明准备冻结模型和许可数据。可下载的 `results/anonymous_rows/` 保存随机重排后的逐例标量，所有方法和种子在每行保持配对。

```bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
d=studies/3eed_k16_utility_loss_20261120
py=crossplatform/env/bin/python
"$py" "$d/common.py"
test -s "$d/results/mechanism.json" || "$py" "$d/mechanism.py"
test -s "$d/run_PRIVATE/heads/LOCK.json" || "$py" "$d/fit.py"
"$py" "$d/source_profile.py"
for source_platform in waymo drone quad; do
    flock research/GPU_RESOURCE.lock "$py" "$d/capture.py" "$source_platform"
done
"$py" "$d/predict.py"
test -s "$d/results/score_receipt.json" || "$py" "$d/score.py"
"$py" "$d/verify.py"
```

第一次提取 18,620 条记录的 299 维特征需要 GPU。三套源模型也可在一次 GPU 锁内并行提取。每条记录只提取一份特征，所有损失和种子共享。已有完整特征收据时，`capture.py` 会验证并复用各分块。源训练、反例、梯度检验、排序头推理、最终 IoU 评分和表格交叉核验可用 CPU 运行。在已有完整结果的工作区，直接执行最后的 `verify.py` 命令即可复核。

`PROTOCOL.json` 给出样本、分数定义、训练预算和评价口径。`run_PRIVATE/heads/LOCK.json` 登记 45 个只在本源留出录制上选出的模型。`predict.py --watch` 可在特征提取期间接收已核验的分块，只有所有方向完成后才生成完整预测封存文件。`score.py` 核对该文件后才读取普通验证标签。`verify.py` 从匿名逐例结果重建各方向、各种子和所有对照的结果，并独立重放一组排序决策。

`results/target_metrics.csv` 中 `seed_id=mean,direction=six_equal` 是主结果，数值单位为百分数。`results/paired.csv` 的差值单位是百分点；`rescue`、`harm` 对应严格跨过或跌出 IoU 门槛的记录数。三个种子的平均计数可以是小数，单个种子的计数均为整数。`results/all_rows.csv.gz` 每行给出一条方向记录及全部对照 IoU，包括所有未通过和退化记录。

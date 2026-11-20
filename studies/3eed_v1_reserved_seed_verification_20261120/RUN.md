# 复现同一帧的备用 seed 核验

使用精简 v1 登记的三份源权重、已编译的 3EED 环境和项目普通清单。源域留出和开发面板读取[上轮已封存的四个输出框及原始分数](../3eed_v1_camera_word_replacement_20261120/RUN.md)，每块第一条再用部署入口核对。每条独立重新执行冻结 3EED，核对四个框的原生来源，并对所有备用 seed 执行两种六层读出。标注在整批预测封存之后由独立评分入口读取。收集结果按块原子写入，可使用相同代码和配置接续。

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
study=studies/3eed_v1_reserved_seed_verification_20261120
py=crossplatform/env/bin/python
"$py" "$study/collect_probes.py" source --smoke
"$py" "$study/collect_probes.py" source
"$py" "$study/score_probes.py" source
"$py" "$study/recount.py" source
"$py" "$study/collect_probes.py" pilot
"$py" "$study/score_probes.py" pilot
"$py" "$study/recount.py" pilot
```

公开的匿名逐例表可以独立重算各平台、六方向宏平均及每项阈值救回和误改。`recount.py` 只需要 Python 标准库。私有框、原始点、自然描述、权重和数据清单沿用本地登记。评测口径为原句的七维真值和有向 3D IoU，Acc 使用严格大于阈值。

# 重放

在能运行精简 v1 的完整研究工作区中执行。输入包括三个登记的单源 checkpoint、源域自然描述清单、原相机标定和点云、原始精简 v1 的封存框以及项目的普通目标清单。公开研究包提供代码和匿名结果。私有输入沿用各自的登记路径。

```bash
study=studies/3eed_v1_camera_word_replacement_20261120
py=crossplatform/env/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$py" "$study/verify.py" scope
"$py" "$study/calibrate.py"
"$py" "$study/predict.py" source
"$py" "$study/assess.py" source
"$py" "$study/verify.py" source
"$py" "$study/check_label_access.py"
"$py" "$study/predict.py" pilot
"$py" "$study/no_fit.py" seal pilot
"$py" "$study/assess.py" pilot
"$py" "$study/no_fit.py" score pilot
"$py" "$study/verify.py" pilot
```

源域标定和文字改写分别见 [calibrate.py](calibrate.py) 与 [factor.py](factor.py)。官方加载器实际返回的真值字段已经逐行核对，项目的 `historical` 列使用同一七维真值。[官方路径与附加指标说明](OFFICIAL_PATH.md)

目标预测按方向、每 64 条一块保存，重启时按照代码与输入清单摘要检查块。原句的精简 v1 赢家须与该条已有封存框逐值一致。只有原句中存在可解析的位置词并且有足够物理支持时才需要第二次语言条件前向。两个前向读取相同点云采样，复用冻结的点云骨干输出。

原阶段和零新增训练对照各自在标签评分前封存选框。评分复用同一原句对应的原始目标框。普通 18,620 条的入口为以上四个目标阶段命令中的 `ordinary`，其预测入口读取预定的开发实验决定。查看全部相对失误时使用每个阶段的 `rows.csv.gz`，可按同一匿名记录与 `no_fit_rows.csv.gz` 连接。研发决策与数据范围见[预设协议](PROTOCOL.json)和[新增训练对照](NO_FIT_PROTOCOL.json)。

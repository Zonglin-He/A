# 运行要求

使用精简 v1 的三份已登记权重、原始单帧输入、已编译的 3EED CUDA 环境和支持 C++17 的编译器。首次调用按源码 SHA-256 将 CPU 几何库编译到本地私有工作目录。除这一评分库外，此实验没有训练或安装步骤。

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
study=studies/3eed_v1_shared_heading_overlap_20261120
py=crossplatform/env/bin/python
"$py" "$study/check_geometry.py"
"$py" "$study/run_heading.py" source --smoke
for part in source pilot
do
  "$py" "$study/run_heading.py" "$part" &&
  "$py" "$study/assess_heading.py" "$part" &&
  "$py" "$study/recount_heading.py" "$part" || exit 1
done
```

`run_heading.py` 对三份源模型分别复用既有的普通清单和采样，实际运行冻结模型，对每条记录的 v1 四个框和最终赢家进行逐值核对。它保存每种读出的四个原生候选编号，以及整个候选池在传感器坐标下的 PCA 框。`assess_heading.py` 验证整批预测封存后，以原句和原始七维真值计算所有任务结果。每个被任一方法提名的框均使用两种几何实现再次评分。`recount_heading.py` 仅凭公开匿名标量和 Python 标准库复算。

普通 Validation 的 `ordinary` 运行使用相同三个入口，其条件在实验开始前登记。开发面板中“在原四槽以外选择”的净救回，按该子集的 Acc@50 救回数减去误改数计算。同时区分原四个槽全都未达标时的实际救回。相同几何采用中心和尺寸的 $10^{-6}$ 米容差，以及轴向取模后的 $10^{-6}$ 弧度容差。

有研究工作区的复现实验使用精简 v1、候选 PCA 实现、源域与目标清单及同一传感器坐标下的有向框评分器。PCA 模块在原研究工作区的相邻研究目录中名为 `mechanisms.py`，其公开版放在 `own_code` 下。原文、传感器数据、封存清单和权重由运行者提供。[候选 PCA 实现](https://github.com/Zonglin-He/A/blob/9ef49fce7f3a2b15fc7c0028ca7f4ba538ec5ccc/studies/3eed_v1_retrieval_binding_refinement_20261119/own_code/mechanisms.py) · [清单桥接](../3eed_v1_camera_word_replacement_20261120/predict.py) · [有向框评分复核](../3eed_v1_camera_word_replacement_20261120/assess.py)

只复核公开成绩时，从本实验目录运行 `python3 recount_heading.py source` 和 `python3 recount_heading.py pilot`。这两个命令直接读取随附的匿名记录，只需 Python 标准库。

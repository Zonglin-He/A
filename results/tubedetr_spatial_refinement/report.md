# TubeDETR空间小改动：两域固定128完整结果

本轮为用户授权的两种未观察候选公式小改动及有限标量验证；主干、教师、八观察、时间、原评分与精确DP保持不变。先完整预测封存，再独立CPU评分。数据已多次曝光并用于选择，不能称独立确认。没有新神经网络调用或全量。

| 面板 | 配置数 | Temporal dense | 所选dense | dense增量pp | 未观察增量pp | Full v | v增量pp | 事前护栏通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| vidstg128 | 14 | 48.616419 | 48.582456 | -0.033963 | +0.049438 | 19.699881 | +0.026127 | False |
| hc2128 | 15 | 57.596687 | 57.630989 | +0.034302 | +0.008856 | 27.718721 | +0.069176 | True |

所选参数、正负结果、逐阶段理由见各域selected_profile/all_metrics/decisions。原仅参数最佳是本轮incumbent；没有用旧未执行机制草稿的结果选参。两个候选族分别针对WLS收缩和绝对框迁移风险，不同时改变置信或评分。候选小改动属于输出侧后处理，不能称原学生decoder新预测。

## vidstg

参数 `{"gamma": 0.0, "admission": 0.4, "guard": 2.0, "link": 1.0, "mode": "relative_residual", "alpha": 1.0, "beta": 0.5}`；事前护栏通过=False。

相对Temporal：v +0.026127pp，描述性父bootstrap95%区间[-0.231911,0.309739]，胜/负/平=54/35/39；s -0.033963pp，描述性父bootstrap95%区间[-0.585495,0.548272]，胜/负/平=54/36/38；unobserved_sIoU +0.049438pp，描述性父bootstrap95%区间[-0.421432,0.570248]，胜/负/平=50/34/44。

相对incumbent：v +0.046709pp，描述性父bootstrap95%区间[-0.063092,0.172305]，胜/负/平=21/12/95；s +0.052094pp，描述性父bootstrap95%区间[-0.061762,0.181409]，胜/负/平=21/13/94；unobserved_sIoU +0.051839pp，描述性父bootstrap95%区间[-0.064896,0.184929]，胜/负/平=21/13/94。

各族完整结果中dense最高配置：

- original: dense=48.530362，v=19.653172，未观察=48.589243，平均改动帧=5.3047，参数{"gamma": 0.0, "admission": 0.4, "guard": 2.0, "link": 1.0, "mode": "original"}。
- center_preserve: dense=48.528953，v=19.657963，未观察=48.587726，平均改动帧=6.4453，参数{"gamma": 0.0, "admission": 0.4, "guard": 2.0, "link": 1.0, "mode": "center_preserve", "alpha": 1.0, "beta": 0.5}。
- relative_residual: dense=48.582456，v=19.699881，未观察=48.641082，平均改动帧=16.4453，参数{"gamma": 0.0, "admission": 0.4, "guard": 2.0, "link": 1.0, "mode": "relative_residual", "alpha": 1.0, "beta": 0.5}。

## hc2

参数 `{"gamma": 0.025, "admission": 0.7, "guard": 0.4, "link": 1.0, "mode": "original"}`；事前护栏通过=True。

相对Temporal：v +0.069176pp，描述性父bootstrap95%区间[-0.138254,0.278461]，胜/负/平=77/44/7；s +0.034302pp，描述性父bootstrap95%区间[-0.325269,0.382996]，胜/负/平=80/48/0；unobserved_sIoU +0.008856pp，描述性父bootstrap95%区间[-0.314095,0.318148]，胜/负/平=81/47/0。

相对incumbent：v +0.000000pp，描述性父bootstrap95%区间[0.000000,0.000000]，胜/负/平=0/0/128；s +0.000000pp，描述性父bootstrap95%区间[0.000000,0.000000]，胜/负/平=0/0/128；unobserved_sIoU +0.000000pp，描述性父bootstrap95%区间[0.000000,0.000000]，胜/负/平=0/0/128。

各族完整结果中dense最高配置：

- original: dense=57.630989，v=27.718721，未观察=57.571559，平均改动帧=4.8516，参数{"gamma": 0.025, "admission": 0.7, "guard": 0.4, "link": 1.0, "mode": "original"}。
- center_preserve: dense=57.609057，v=27.691555，未观察=57.549854，平均改动帧=5.1172，参数{"gamma": 0.0, "admission": 0.7, "guard": 0.4, "link": 1.0, "mode": "center_preserve", "alpha": 1.0, "beta": 0.5}。
- relative_residual: dense=57.510381，v=27.630537，未观察=57.449528，平均改动帧=7.4766，参数{"gamma": 0.0, "admission": 0.7, "guard": 0.4, "link": 1.0, "mode": "relative_residual", "alpha": 1.0, "beta": 0.25}。

护栏只保证本开发面板点估计，没有解决多次比较和重复曝光。所有bootstrap均在选择之后描述，不作独立显著性确认。若无可行配置则不晋升；未启动新全量。零初始化、尺寸保留或更保守修改不能自动证明传播可靠性，不能把减少改动当成未观察帧获得真实信息。

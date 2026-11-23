# 视觉指代研究

当前主方法是 **Qwen 单帧三维指代 · 只缩顶部**。冻结 Qwen 提供对象位置和框先验，LiDAR 约束三维框，最后只收缩顶面并保留已有几何证据。 [方法、运行入口和主表](docs/qwen-grounding.md) · [主方法登记](QWEN_CURRENT_METHOD.json)

| 要找什么 | 入口 |
| --- | --- |
| 在新设备上打开或恢复项目 | [换机说明](docs/device-setup.md) |
| 论文主结果和消融 | [论文实验](studies/3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) |
| WildDet3D 和 Talk2Car | [外部复测](studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) · [Talk2Car 调参](studies/3eed_talk2car_parameter_tuning_20261123/REPORT_ZH.md) |
| 旧版 3EED、STVG、AVIS、GREAT | [项目和历史分支索引](docs/research-index.md) |
| 按名称找 `studies/` 下的实验 | [全部实验目录](docs/studies.md) |

普通 3EED Validation 的 **9,310 条**描述上，主方法的三平台等权 **Acc@50 为 37.23%**，相对同批冻结 `C_ray_simple` 提高 **2.82 个百分点**。 [逐平台结果](docs/qwen-grounding.md#普通-validation)

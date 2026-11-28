# 视觉指代研究

最终论文方法为 **Minimal-D（Bind–Project–Select）**。四尺度绑定后，对两个姿态分别进行统一轴恢复，再以首次 Qwen 二维框排序，平局时使用射线代价。 [推理入口](docs/qwen-grounding.md) · [主方法锁](QWEN_CURRENT_METHOD.json) · [完整论文稿](studies/3eed_minimal_pipeline_20261128/MANUSCRIPT_ZH.md)

| 要找什么 | 入口 |
| --- | --- |
| 在新设备上打开或恢复项目 | [换机说明](docs/device-setup.md) |
| 论文主结果和消融 | [Minimal-D 等价消融](studies/3eed_minimal_pipeline_20261128/REPORT_ZH.md) · [原三轴与二维排序](studies/3eed_extent_2d_factorial_20261128/REPORT_ZH.md) · [冻结后评估](studies/3eed_minimal_pipeline_20261128/FINAL_EVALUATION_ZH.md) |
| WildDet3D 和 Talk2Car | [外部复测](studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) · [Talk2Car 调参](studies/3eed_talk2car_parameter_tuning_20261123/REPORT_ZH.md) |
| 旧版 3EED、STVG、AVIS、GREAT | [项目和历史分支索引](docs/research-index.md) |
| 按名称找 `studies/` 下的实验 | [全部实验目录](docs/studies.md) |

普通 3EED Validation 的 **9,310 条**描述上，旧射线加顶缩的三平台等权 **Acc@50 为 37.23%**，相对同批冻结 `C_ray_simple` 提高 **2.82 个百分点**。 [逐平台结果](docs/qwen-grounding.md#普通-validation)

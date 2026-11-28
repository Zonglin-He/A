# 视觉指代研究

当前论文主方法为 `ray_equivalent_extent_2d_rank`。双姿态候选分别执行三轴观测等价投影，再复用首次 Qwen 二维框排序。 [推理入口](docs/qwen-grounding.md) · [主方法锁](QWEN_CURRENT_METHOD.json) · [论文稿与配对结果](studies/3eed_extent_2d_factorial_20261128/MANUSCRIPT_ZH.md)

| 要找什么 | 入口 |
| --- | --- |
| 在新设备上打开或恢复项目 | [换机说明](docs/device-setup.md) |
| 论文主结果和消融 | [三轴投影与二维排序](studies/3eed_extent_2d_factorial_20261128/REPORT_ZH.md) · [历史高度投影实验](studies/3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) |
| WildDet3D 和 Talk2Car | [外部复测](studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) · [Talk2Car 调参](studies/3eed_talk2car_parameter_tuning_20261123/REPORT_ZH.md) |
| 旧版 3EED、STVG、AVIS、GREAT | [项目和历史分支索引](docs/research-index.md) |
| 按名称找 `studies/` 下的实验 | [全部实验目录](docs/studies.md) |

普通 3EED Validation 的 **9,310 条**描述上，旧射线加顶缩的三平台等权 **Acc@50 为 37.23%**，相对同批冻结 `C_ray_simple` 提高 **2.82 个百分点**。 [逐平台结果](docs/qwen-grounding.md#普通-validation)

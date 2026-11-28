# 视觉指代研究

论文主方法为 **S5-pass**。冻结 Qwen3-VL 两次读出，以 Minimal-D 为锚构造最多 13 个候选，由只用 Waymo 训练的 Verifier 计算单一相对效用。开发 665 条为 **38.10% mIoU / 40.15% Acc@50**。 [推理入口](docs/qwen-grounding.md) · [论文登记](QWEN_PAPER_METHOD.json) · [方法稿](studies/3eed_minimal_verifier_20261128/METHOD_ZH.md)

| 要找什么 | 入口 |
| --- | --- |
| 在新设备上打开或恢复项目 | [换机说明](docs/device-setup.md) |
| 论文主结果和消融 | [S5-pass 配对结果](studies/3eed_minimal_verifier_20261128/REPORT_ZH.md) · [Minimal-D 等价消融](studies/3eed_minimal_pipeline_20261128/REPORT_ZH.md) · [原三轴与二维排序](studies/3eed_extent_2d_factorial_20261128/REPORT_ZH.md) · [冻结后评估](studies/3eed_minimal_pipeline_20261128/FINAL_EVALUATION_ZH.md) |
| WildDet3D 和 Talk2Car | [外部复测](studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) · [Talk2Car 调参](studies/3eed_talk2car_parameter_tuning_20261123/REPORT_ZH.md) |
| 旧版 3EED、STVG、AVIS、GREAT | [项目和历史分支索引](docs/research-index.md) |
| 按名称找 `studies/` 下的实验 | [全部实验目录](docs/studies.md) |

普通 3EED Validation 的 **9,310 条**描述上，旧射线加顶缩的三平台等权 **Acc@50 为 37.23%**，相对同批冻结 `C_ray_simple` 提高 **2.82 个百分点**。 [逐平台结果](docs/qwen-grounding.md#普通-validation)

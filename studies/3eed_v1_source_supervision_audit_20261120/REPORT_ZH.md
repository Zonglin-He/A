# 精简 v1 的源域监督和左右改写核验

精简 v1 在各自源平台的 TRAIN 上训练过新增解码器。自然描述提供主体框，合成关系描述额外提供主体和参照框，目标平台上的模型冻结。本次只读核验复原了这两类监督，并检验在固定原参照的条件下，把一句合成描述的左右方向反转后，能否依原规则找到新的主体。[原解码器训练](https://github.com/Zonglin-He/A/blob/261676dd7ed9cc9b6d87aa9514d5a083b4e5c860/studies/3eed_r174_strong_anchor_joint_support_20261119/REPORT_ZH.md) · [核验协议](PROTOCOL.json)

## 实际训练用了什么

逐条重建三个源平台的历史训练和监测清单。全部 12,728 条合格自然描述对应的原始记录中，`others` 共包含 62,134 条上下文对象记录，同一帧的对象可以出现在多个描述下。这些条目只有 `class_other` 和 `bbox_3d_other` 两个字段。直接将其中某个框指定给自然句中的某个参照短语，仍需要额外的实例关联依据。

代码核对了每条记录属于相应平台官方 TRAIN 录制，重建的两个框和参照有效位与训练特征缓存逐值相同。每条自然描述的参照均标为未知。已知双端来自源帧内的规则构造，选取可见参照与同类候选，要求投影中在指定一侧横向最近的主体具有唯一性。后续解码器训练使用三种句式表达已构造的关系。三份改写特征缓存覆盖的行恰好是这些合成行，所有改写缓存都只含特征。登记的三份推理权重通过 SHA-256 校验。[样本构造](https://github.com/Zonglin-He/A/blob/261676dd7ed9cc9b6d87aa9514d5a083b4e5c860/studies/3eed_recurrent_role_support_decoder_20261119/prepare.py) · [改写和训练输入](https://github.com/Zonglin-He/A/blob/261676dd7ed9cc9b6d87aa9514d5a083b4e5c860/studies/3eed_r174_strong_anchor_joint_support_20261119/own_code/capture_source174.py) · [全部核验数值](RESULTS.json)

## 当改写改变了指向

固定参照实例、参照短语和主体类别，保留原相机投影、可见对象及原始的 0.015 图宽间隔要求。只反转主体相对参照的左右谓词。表中的投影重叠或距离近似并列均按原规定保留为未知。最邻近排序和逐物体枚举约束两种实现，对全部 977 条原合成关系及其反转给出相同判定。[核验实现](audit.py)

| 源平台和划分 | 自然描述 | 合成关系 | 原方向复现旧主体 | 反转后可唯一指定新主体 | 另一侧无同类候选 | 投影间隔不足 | 最近距离近似并列 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Waymo 训练 | 3,223 | 607 | 607 | 46 | 472 | 86 | 3 |
| Drone 训练 | 3,865 | 74 | 74 | 0 | 74 | 0 | 0 |
| Quad 训练 | 4,764 | 181 | 181 | 0 | 180 | 1 | 0 |
| **训练合计** | **11,852** | **862** | **862** | **46** | **726** | **87** | **3** |
| Waymo 源域监测 | 459 | 52 | 52 | 7 | 33 | 12 | 0 |
| Drone 源域监测 | 249 | 63 | 63 | 0 | 18 | 45 | 0 |
| Quad 源域监测 | 168 | 0 | 0 | 0 | 0 | 0 | 0 |

训练清单中反转后仍可沿用原主体的是 **0/862**。只有 **46/862（5.34%）** 能在同一源帧的已标注可见物体中按原规则指定另一个主体，全部来自 Waymo。其余 816 条缺少能在该库存和间隔规则下确定的新标签。这个计数限定于固定原参照、只反转明确左右关系及既有投影定义；图像中真实存在而未列入库存的物体需要另外核验。此次反转仅检验标签有效性，模型训练、参数选择和目标评测均未使用这些反转记录。

## 训练、评测与报告口径

每个源的自然主体标注与源内派生关系对都属于有监督训练信息。项目原来的四个解码器消融臂共用这些样本、初始化和每源 12 个训练轮次。与官方 Frozen 的差值衡量整套源域准备和推理的效果。用相同源样本、监督、训练步数和输出协议对比两个解码器，才能进一步归因给解码机制。若单独测新增关系监督的贡献，需要同一架构的自然描述训练对照并匹配训练预算。[实际优化目标](https://github.com/Zonglin-He/A/blob/261676dd7ed9cc9b6d87aa9514d5a083b4e5c860/studies/3eed_r174_strong_anchor_joint_support_20261119/own_code/model174.py#L113-L147)

目标域单目标比较沿用原句及官方加载器返回的七维真值。Drone、Quad 原始记录含九项，官方单目标实现取 `bbox[:7]`。精简 v1 的框允许输出 PCA 朝向，项目使用支持有向预测的 IoU 与同样的严格阈值做配对比较。18,620 条普通验证方向记录来自 9,310 条描述，属于已曝光的普通评价范围。本次核验仅打开源域训练数据和源训练缓存；目标数据及 FINAL 未访问，模型更新为零。[官方加载器](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/joint_det_dataset.py#L485-L549) · [官方评分器](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/grounding_evaluator.py#L244-L294) · [普通评价清单](https://github.com/Zonglin-He/A/blob/b3c9f9b6cc92ab79706ff3c0fa6241f6d9e05198/studies/3eed_v2_ordinary_val_subject_ranking_20261119/REPORT_ZH.md)

公开发布派生标注还涉及数据授权。3EED 论文附录 A.6 写 CC BY-SA 4.0，作者链接的 Hugging Face 数据卡页首标 Artistic-2.0，正文第 7 节写 Apache-2.0，并要求核对上游 Waymo、M3ED 条款。本核验公开程序与分组计数。若需分发逐例派生标注，应按实际资产向发布方核实适用许可。[论文附录 A.6](https://arxiv.org/html/2511.01755v1#A1.SS6) · [官方数据卡](https://huggingface.co/datasets/RRRong/3EED#7-license)

## 复核

在保存了三份源 TRAIN 清单、原始源帧、当前 v1 权重及两组源训练特征缓存的工作区运行以下命令。历史代码来自协议固定的 Git 提交。程序不写训练数据或权重，只产生汇总文件；已有汇总必须与重算结果一致。

```bash
crossplatform/env/bin/python studies/3eed_v1_source_supervision_audit_20261120/audit.py \
  --archive-repo "$public_repo" \
  --historical-source-cache "$role_support_cache" \
  --reworded-feature-cache "$reworded_cache"
```

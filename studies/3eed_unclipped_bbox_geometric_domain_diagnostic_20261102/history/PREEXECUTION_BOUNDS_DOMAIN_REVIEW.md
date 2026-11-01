# Bbox 四角全在图内：独立几何域审查（只读，未执行诊断）

**判断：有理由做一次 cached-only 域对照。四角全在图内是当前提示词/解析合同的额外条件，不是标准矩形 IoU 定义所需条件。它可能提前丢弃具有可见交集的矩形，但尚未证明被丢弃的输出定位正确或会改善任务。** 当前 R45 主次结果与 STOP 保持不变；本审查没有生成新预测、评分或访问 FINAL。

## 现有实现和历史覆盖

- `crossplatform/round39/common39.py:23–56`：JSON 安全/唯一位置、有限四数、正 xyxy 面积检查后，第55行以四角不全落在 `[0,W]×[0,H]` 判 invalid_box。也就是说该原因下的 box 已经通过有限、正面积，而非倒置或坏 JSON。
- `common39.py:58–67` 的中心接口只接受 parsed，然后把原始四角平均得到的二维点按原 grid 逆变换，检查唯一 mask。
- `crossplatform/round40/common40.py:17–33` 的 rectIoU 仍先检查 parsed 并 assert 四角图内，**没有测试图外角但合法正面积的标准 IoU**。R40 PLAN 的 original_bbox_contract 也明确 reuse R39 boundsvalid。R45复用同一门；目前目标代码/原计划/报告内未发现这个域消融已测。只对相关路径作此历史结论，不声称扫描了所有私有历史内容。
- `crossplatform/round13/association.py:32–34` 已有标准 rectangle IoU 的交并面积式，不输入 imageW/H，也不裁剪至图像。因此无需引入新几何拟合、分数或阈值。

## 几何依据

设原生输出 b=(x1,y1,x2,y2)，有限且 x1<x2,y1<y2。可见 detector 矩形 d 位于图内。标准 IoU 为交面积除以 `area(b)+area(d)-intersection`，四角越图不使定义失效。示意（非任务数据）：图宽100，b=(-10,10,40,40)，d=(0,10,40,40)，原矩形 IoU=1200/1500=0.8。裁剪 b 后变1.0，**会改变证据**，所以这个假设必须使用原始面积，不能 clip。

[torchvision 官方 box_iou](https://docs.pytorch.org/vision/main/generated/torchvision.ops.box_iou.html) 与其[官方实现](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/boxes.html)分别给出框间 Jaccard 与独立于 image-size 的交并面积式；clip_boxes_to_image 是另一独立操作。引用的是标准几何定义，**不是 Qwen 图外输出语义正确的保证**。

二维中心查 mask 的合同不同：由原坐标算中心后，中心必须在实际图内，且恰好命中一个 mask；若中心在图外，仍 UNKNOWN。不把它当真实三维物体中心，不生成/修正任何三维框。

## 一个未执行的最小诊断计划

1. 输入只用 R39 3B 与 R45 7B 各208条封存 bbox 原始回答、相同 model grid、原全 mask 库/完整3D组和旧 subset。原 strict raw/schema/readout/task 表不可覆盖。不要读 FINAL，也不增加模型调用。
2. 新安全解析规则保持原 schema、singleton、惰性 label、唯一 key、坏JSON/非有限/倒置拒绝；唯一改变为对有限正面积 xyxy 不再用四角全图内作几何域否决。保留 `declared_range_compliant` 独立旗标，全部越域数量仍报告，不把它们改称提示词遵从或模型坐标已认证。
3. 不 clip、swap、reorder、重新缩放、0–1000切换、最近框/阈值选择、手工总分或新提示词。严格按原 image_grid_thw×14 作轴逆变换。若有限坐标的中间面积运算溢出/非有限，则 UNKNOWN并完整记录，不修值。
4. 两种读出都固定：原坐标均值得二维中心→唯一 mask；标准 **unclipped rectIoU**→全部 detector inventory 中唯一正最大值。最大 mask 无3D组时不跳次优；同组用既有 source 最高分代表，完全相同分用既有固定索引规则。fullmap/oldsubset 都保留。
5. 预设主诊断为 **7B finite-positive-domain rectIoU_fullmap vs 7B strict rectIoU_fullmap**，同时保留3B同接口、center与oldsubset匹配控制。全 A208 暴露开发集，只能得到域敏感性诊断；不能反选原 R45 primary，也不构成新确认或 TTA。
6. 在任何 GT 评分前锁定实现/输入/输出并封存，再由独立 CPU 审计资格、原值、映射、query/boxes及任务指标。总体报 balanced mIoU、两平台 Acc25/50、全部接受/UNKNOWN、收益/损害/不变/全部负例，以及原 strict 有效样本逐值不变。不可按封后 IoU决定是否接纳某越域样本。
7. 新模型调用=0、优化=0、单一域改变=1，CPU预算≤5分钟。结果不改善→停止这个域解释；改善只说明原严格门限制了当前缓存接口的证据利用，仍需单独新观测确认，不自动启动TTA。不据一两个负例追提示词，不新增坐标/置信阈值。

## 竞争解释和限制

越域可能是完整物体/遮挡估计，也可能是错误单位、幻觉或错误实例。保留同一绝对坐标假设的 cached 控制不能把这些原因分辨完全。7B大量越域是值得审的测量竞争解释，不是解除所有安全条件或断言原模型更好的理由。矩形IoU可合法计算，也不意味着2D→3D身份对应可靠。原 strict 结论是其合同内的真实负结果，新的域读出如果获益也不应抹去该失败记录。

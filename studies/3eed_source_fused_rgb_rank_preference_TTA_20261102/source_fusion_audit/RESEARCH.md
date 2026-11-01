# 复用R25源域融合模型的只读可行性审查

**结论：可以复用现有源训练权重，值得作为一项有限因子实验候选；目前没有证据认定“缺RGB”就是R30–R35迁移不稳的原因。** A208已具备全部RGB缓存，但DEV和后续集合没有R25 RGB缓存，不能写成零提取成本。本文件不是R36运行登记；没有实现新适配接口、训练、GPU前向、媒体访问或FINAL读取。

## 1. 默认模型实际是哪一个

实际默认入口是 `research/round25/experiment/SOURCE_MODEL_MANIFEST.json` → `FINAL_SOURCE_SELECTED_LOCK.json` → `model25.make_model('residual_rank')` → **rank_fusion25.RankFusion**。仅调用最早的 `fusion25.TrimodalFusion` 会装错家族。

可直接复用的主状态：

- RGB，seed0，epoch1：training_rank_attempt001/states_PRIVATE/rgb_seed0_e01.npz，SHA256 `0ba0e54400af46b612c227b79e6f69546212d8152b92a171bb30ef4d732aec2b`。
- 同结构／源监督无RGB，seed0，epoch1：no_rgb_seed0_e01.npz，SHA256 `46a5b3912a043ea9a34846753614aa6c19bf437867f6cf4cfa962d0a7d3dd481`。
- seed1 RGB epoch5与无RGB epoch30可作预先固定的独立种子复核，不能按当前目标效果换成其一。

本次已核验四份实际权重hash、source family规则hash、目标预测前的selection hash链，见 PROVENANCE_CHECKS.json。源训练使用Waymo TRAIN3682和VAL512；两读出家族、两输入设置、两种子，家族与epoch由源VAL规则选择，当前存档未显示目标标签参与该权重选择。R25目标数据／此前研究存在历史暴露，不能把整段研究历史称为从未看过目标；也不能因为R25备选家族的目标数字更好就改选它。

R25的001源缓存因遗漏Waymo视角旋转已作废。有效训练脚本明确读取source_features_attempt002、rgb_features_attempt002、source_labels_attempt002；不得复用001。无需新源训练，也无需重新下载ResNet34资产。

## 2. 学生实际获得了什么RGB／文字信息

原3EED点云文字主干仍给出固定256个候选及h288、完整text memory和token。ImageNet预训练ResNet34完全冻结，图像最长边缩为768；layer2/layer3合成384维特征。每个预测3D框的8角投影包围成2D矩形，用2×2 ROIAlign得到4个局部token，加1个全图平均token。RGB缓存以float16存储。

融合注意力query来自h288与句子memory均值；RGB作为K/V。它读取整句，而非只取粗类别，但文字侧是 `text_h[1:-1].mean(0)`，包含追加的“not mentioned”部分；不是逐短语与主体、参照物的显式关系执行。全图信息压成一个token，也没有对不同候选RGB进行cross-candidate attention。主体颜色有可能进入表示，复杂参照关系不因此自动可识别。

默认最终分数为：

`log(token_mass(project(h + alpha*delta))) + alpha*rank([q, visual, q*visual, geometry])`。

alpha同时受学习门和valid投影标记控制。框、原decoder、原点云/文字编码器、原匹配投影及RGB编码器均冻结；源训练新增136,930参数。它确实是已经训练过的多模态预测路径，不是将随机cross-attention接到checkpoint。

但它仍不等于Qwen可见信息：Qwen的候选图/场景和角色判断，与按预测3D框投影得到的ROI不一致；压缩后的ResNet特征不保证保留细衣着、文字或参照物信息。后续只能检验“现有RGB表征是否有助于反馈迁移”，不能事先称学生获得了反馈的全部证据。

## 3. 2D–3D对应仍有哪些硬限制

RGB区域由原预测3D框投影生成，没有独立实例mask、遮挡解析或深度排序。正深度、矩形面积、裁剪比例等检查只能证明投影可计算，不能证明ROI属于正确实例。遮挡邻物、背景以及重复query共用像素仍存在；所有256框在同一单帧相机下处理，候选缺失和原6D/yaw0表示不变。

本次仅检查已有A缓存：22条偏好的P/N两侧都存在valid ROI，源分数代表query也22/22有效。因此“所有反馈端点没有RGB输入”不是当前缓存层面的事实；但valid不认证身份。所有R25缓存都声明缺少distortion时按0处理，这个既有标定假设必须保留披露，不能新增“像素严格对应”的结论。

## 4. 缓存覆盖和新增成本

只盘点现成文件，没有打开新媒体或执行特征提取。U仅用已存在source缓存文件名匹配，没有读取U清单／媒体／GT。

| 集合 | 点云文字源缓存可复用 | R25 RGB缓存现有 | 需新增RGB提取 |
|---|---|---:|---:|
| A208 | 全部 | 208 | 0 |
| DEV64 | 全部 | 0 | 64 |
| R64 | 全部 | 0 | 64 |
| S64 | 全部 | 0 | 64 |
| T128 | 全部 | 0 | 128 |
| U128 | 全部 | 0 | 128 |

最小开发只需补DEV64的RGB，不必先把所有历史诊断重提一遍。原R25的4594条RGB缓存耗时323.05秒、峰值约143.3MB，仅作数量级参考，不是本次新图像的已测延迟。A可直接利用缓存；DEV/新确认的逐观测RGB成本必须计入部署，不能仅报告缓存后TTA耗时。

旧 `fusion25.bank` 的feature/rgb_feature路径硬编码在R25目录；直接给它DEV/R/S/T/U会找不到缓存。未来若登记实现，应使用显式bank入口读取当前source缓存和新增RGB缓存，不要改旧模块全局feature函数或覆盖旧结果。

## 5. 原R25负结果必须保留

源VAL的RGB优势很小，默认RGB与无RGB两种子平均约44.8904% vs44.3526%。全400目标，seed0 RGB 23.0251%，原source22.5666%，同源预算无RGB22.5128%。新64却是RGB25.7905%，原source26.4317%，下降0.6412pp；Quad从35.4778%降至31.9064%。seed1新64也只是26.4265% vs source26.4317%。

同一权重RGB清零至完整RGB，全400两种子只有约+0.0870/+0.2099pp；已有Qwen旧336平均26.2617%，内部融合默认旧336仅22.4984%。这证明旧融合尚不能替代Qwen或建立稳健跨平台优势。新增非视觉排序偏置也可能伤害源选择，不能把RGB/noRGB源模型差距都解释为RGB信息本身。

## 6. 最小可更新接口与一项因子比较

优先只适应已训练过的 **rank MLP（200→64→1，12,929参数）**，其余fusion／门／QKV／delta／原projection全部冻结。末层只有65参数，gate有4,417参数，但本轮不应将三个接口都扫一遍。rank MLP直接决定最终匹配分数，输入含点云文字条件化query、RGB视觉特征、交互及几何；更新路径明确，也易给出同特征无参数对照。

建议一项2×2：固定seed0、同家族source预选RGB和noRGB两份权重，各比较 frozen 与 rank-only TTA。原XYZT frozen/R30及当前已锁定scope仍报告为外部系统参照。每份fused模型都相对自身frozen计算适应增量；不能将RGB frozen相对XYZT的优势算进TTA增益。两源模型权重不同，因此两份适应增量的差是“源训练／RGB接口组合下的适应差”，不是严格的单一输入因果效应。

保持原22条偏好、原U96/93、平台独立初始化、beta1、16步和固定一次学习率预算；不新增G、不删已知错误标签。**损失必须用最终推理分数 logmass+gated-rank 的组间差，保持项也在该分数相对各自frozen q0的坐标上定义。** 不能沿用只看旧noun exact-logodds的loss而绕过新rank项，也不能把fused分数和旧概率坐标直接混加。分数尺度和源q0改变需披露，不宣称与R30的梯度强度完全匹配。

同信息不更新控制：用同一冻结fused的200维rank输入（含RGB、文字和几何）作为查询特征，复用原P/N组的最小L2坐标修正记忆、固定k1及q0锚定；各source模型在自己的最终分数坐标上构造记忆。只用原h288记忆会丢掉学生新增RGB信息，不能作为唯一“同信息”强对照。原R34 h-key QP可继续作次要控制。该记忆是待实现的匹配控制，不是已测结果或新颖性主张。

能力门仍核查偏好落实及全局最终选择，不把局部B>A升级成B优于全部候选。所有原负例和无反馈观测损害都保留，按平台及场景报告总体结果。

## 7. 论文机制与适用前提

[xMUDA，CVPR2020](https://arxiv.org/pdf/1911.12676)用主任务头和模仿头区分模态特有信息与互学目标；官方[架构](https://raw.githubusercontent.com/valeoai/xmuda/master/xmuda/models/xmuda_arch.py)具有dual head，训练[入口](https://raw.githubusercontent.com/valeoai/xmuda/master/xmuda/train_xmuda.py)联合源监督与目标跨模态KL。它要求点级投影和已训练的分支；不是source-free grounding TTA。这里可以借鉴“不要把外部模态的全部判断强写进无该信息的学生”，无需为了引用加入未训练模仿头，也不据此保证R25能学会Qwen证据。

[WildRefer，ECCV2024版本原文](https://arxiv.org/html/2304.05645v3)的三模态交互和解码通过监督训练建立。R25已有源训练接口满足基本前提，但其单帧局部ROI、全图池化和固定框，与原方法多帧／稠密交互不同；不能称复现。当前最小实验不需要重做它的源训练。

## 8. 停止与裁决

- 若零步frozen fused不能重放旧A输出，或新RGB路径与完整推理不闭环，先停止，不能用代码接通代替可用source。
- 若在固定预算下偏好仍不能落实，停止此接口，不根据DEV坏例增大层／换家族／换种子。
- 若RGB fused只有架构/源训练增益，而TTA不超过自身frozen，结论是没有新增适应收益。
- 若TTA不超过同信息fused-feature记忆，参数更新额外价值尚未建立。
- 若RGB的适应增量没有优于同预算noRGB，不能用本轮支持“缺RGB是之前主要原因”；仍需保留反馈错误、ROI污染和共享传播等解释。
- 不以Drone改善掩盖Quad退化。方案先在已暴露开发数据冻结；U当前轮评分后只作暴露诊断，不变回独立确认。FINAL继续保护，是否用新独立观测确认由root另行登记。

本轮裁决是**技术上可复用、信息机制可检验、收益未知；可用既有权重作一个有限因子实验，不启动新的源训练或模块扫参。**

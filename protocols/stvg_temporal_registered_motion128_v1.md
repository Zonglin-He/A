# R29：原全图上下文与物理注册目标RGB-motion，HC2 RC128

2026-10-20，草稿未预锁、未运行模型/新GT。目标时间/空间条件时间增益，空间实现/配置及E/Live41/42/43坐标完全冻结。沿用同Source Vid权重、65物理帧/两个33/32 offset、原八专家位置、原2080动作MBR。0新专家、网络训练、学生更新；本轮是观察能力probe，不是OPD或独立确认。

## 新信息、历史与限制

R28整幅输入换成移动crop，两模态一起改变，raw hard/expected均显著低于Native；该精确配置保持关闭。R8只替换过全图joint65 Swin上下文，R4融合后ROI pool、R21融合前motion-key prior、旧coupling B2 postfusion ROI query均不是目标RGB先经非线性Swin。未定位同条件注册RGB-motion结果，不称整个项目没做过双流。R21已存Source-exact原三模态pre-fusion src/pos/mask/views，但R28没有crop raw motion，不能称纯缓存实验。

唯一机制：原全图appearance/text fusion输入保持逐值不变、crop外原motion保持；目标RGB仅经原VideoSwin/input_proj2，按锁定crop几何注册到原motion网格，再通过原fusion/门控/原两pass产生时间反馈。融合后的appearance/text允许随新motion改变，不能声称最终memory完全不变。比旧ROI不同在于RGB先非线性重编码；但回原粗网格可能压掉细节，混合原/局部特征仍是预训练分布外输入。逐帧反注册不能逆转Swin已在crop坐标完成的时空混合、无法恢复真实运动或裁掉的交互背景。没有收益预测。

## 固定几何

直接复用R28已锁Live41的65个逐帧2倍框/min16/224方形INTER_LINEAR矩形，RGB哈希必须等于R28实际view。无Source指导新分支、crop/平滑/种子/时间参数搜索，仍全视频覆盖。像素过原make_batch归一化/两个offset，只有model.vid与input_proj2实际重编码；未用新ResNet输出。

注册坐标采用原有效feature cells在原图归一化空间的均匀分格约定，不声称精确CNN感受野。valid必须为左上完整矩形，padding排除。每原格与crop归一化矩形的面积交集/原格面积作为alpha，交集质心映射到crop坐标，以align_corners=False、bilinear、border读取新motion。输出=(1-alpha)原motion+alpha局部值；alpha=0分支原样保留，包括padding。覆盖率不额外调强度、norm或温度。原position/padding/frames保持原值，不把局部坐标直接赋给全图位置。

FeatureWarp匹配控制：从原native motion按同crop几何采样到实际crop motion grid，采样结果显式转换为实际新input_proj2的FP16 dtype，再用完全相同注册/面积混合/fusion/decoder。它没有新RGB，可能有非零效应，不应要求等于Native；其作用是隔离重采样与拼接。真正零控是原motion原样替回，原prefix/logits/q精确恢复。

## 输出与成本

7输出：原Native、Strong=HiResAuxMix、R28 MovingLive raw（已有负控制）；RGBRegistered raw及固定.5Native混合；FeatureWarp raw及同.5混合。主RGBRegisteredMix，唯一纯教师候选RGBRegistered，FeatureWarp为控制，不按结果晋升。全部使用同4条冻结空间轨迹，主要三seed指标父内均值，全部128曝光开发父/每父一query，TC109不选配置。

最多256新Swin offset编码/input_proj2，加首父原图和crop两个Source完整零控（4个额外Swin/ResNet offset）；actual attempted/completed按Source/Swin/ResNet/投影/fusion/两pass重放分账。原图控制的motion与R21src、boxes/logits/q须exact；crop完整控制时间logits/q与R28须exact，新独立Swin projected motion与该控制raw捕获须exact。共享非阻塞gpu.lock，GPU lease900秒含加载/IO/3秒清理预留、host6GiB/disk30GiB/启动GPU4GiB/运行5GiBreserve/allocator<=.8，失败完整保存CPU已返回张量/阶段和真实成本。不信号其它任务。

首两父完整矩阵（含Native精确重融合、RGB/FeatureWarp两条件、7输出、原图/crop控制和独立geometry/采样性质），按加载+实际控制+128×最大非控制父成本+60<900才继续，否则完整保存停止、不静默删臂/延预算。全128输出及raw新motion封存后才GT；网络/flags、空间/旧缓存pins全部前后不变。

## 审计与裁决

独立标量矩形/alpha/warp grid核验、随机local特征bilinear数值oracle、padding/outside不变、原appearance/text及pos/mask逐值不变、真实encoder入口核验与Native两pass exact；保存输入状态/原motion来源SHA、新raw motion、注册输入SHA、prefix SHA/logits/q及物理帧映射（完整成功prefix只在失败partial保留，不在所有最终预测复制）。全128至少抽固定cells/channels独立复算新注册与FeatureWarp，采样点预定不看GT；全部几何和空间exact，所有后验独立CDF、MBR每臂13+selected效用核；官方/Dense核3584最终输出，798720候选质量仅Dense逐项计算，不夸大为全候选双实现。

之后GT仅离线：fullv/t/起止MAE、固定GT支持及未观察actual节点sIoU不变、全部父正负/严重损失/LOO，原Native/Strong/R28旧分数复现，三seed先父内平均、128父bootstrap。RGBRegisteredMix hard>=2pp且CI正可冻结静态组件独立确认；pure RGBRegistered expected>=2pp且CI正，并对FeatureWarp expected优势CI正，才允许另立真实匹配Direct/freshFixed/Live转移。hard失败不单独否定分布资格。所有对照均报；若不超过FeatureWarp不得归因新RGB知识，不超过Strong不得称方法领先；两资格不通过则关闭本配置，不扫crop/alpha/混合/归一化救援。不把本轮成功（若有）称OPD成功或目标完成。

完成/失败同turn档案check/snapshot/check，GitHub allowlist公开代码实际协议全signed/unknown/负尾及实际远端双ref字节/hash/tree收据，排除媒体/GT框/文本/父ID/权重/原缓存。CURRENT/MAIN/PAPER与空间不晋升。尚未实施训练或运行真实模型，本稿需锁前工程复核。

锁前工程明确：两个Source控制的已完成CPU输出容器和全部已完成fusion views挂入partial；任何后续失败保留先前完整输出。Source控制的deepcopy decoder/head继承计数hooks，共享closure，因此总1552decoder/776head含原FP16和拷贝FP32调用。Scorer所读R28旧RESULTS逐字节pin。FeatureWarp local量化同实际crop投影FP16，注册算子在原src FP32计算；独立标量采样注册误差界16eps32×max(1,abs(reference))，local FP16误差界2eps16；Native精确零控不放宽。

锁前padding修正：FeatureWarp逐帧只从原motion有效矩形[:vh,:vw]采样，border夹在有效图像边缘，不能从完整含padding tensor的边缘夹取；独立scalar同式限定有效域。新增只改变invalid padding后local FeatureWarp及注册有效格逐值不变性质。修正发生在任何prepare/模型/GT之前，原CPU draft性质记录保留。

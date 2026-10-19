# R28：冻结空间轨迹引导的逐帧RGB观察，HC2 RC128

2026-10-20。用户授权继续时间/空间条件时间研究，空间算法、配置、Source/E/Live41/42/43坐标不变。只有现有TA-STVG Vid Source SHA5ab12c86…；不新增DINO/ReferDINO/其它专家、模型/头/网络训练。新Source前向只作为时间反馈，新的空间输出全部丢弃。本轮不是OPD，不做学生更新。

## 历史与检验缺口

逐帧移动RGB并非新理念：9月18日spatial_ssl cycle已逐帧native框宽高2倍裁到图内、最小16像素、resize224方形，实际同TA双offset前向并保存zero时间logits。旧HC1八源/100帧与Vid八源/变长、两个Source检查点，不是当前HC2/65帧/当前Live41；旧报告固定Native时间，只评空间SSL。当前16缓存的旧回执、Native/roundtrip数值与物理MAP核验96项pass，无新GT/任务分数/模型。旧panel文件全缺失，episode仅FakeTensor读取允许元信息，不称完整原输入或整模型复验。

10-04 UVTG/PE移动crop已有实测及负尾，不恢复该专家；R13则在全部时间使用同一个union矩形，101/128完整原图且主v+.0295CI跨0。当前未找到同条件HC2 Source-Vid/冻结Live41的逐帧RGB时间评价。R26/R27精确配置保持关闭，不重开Action强度/跨流层/位置扫描。

唯一新能力问题：在各帧分别观察冻结目标区域，是否提供强于完整画面/固定union窗的时间证据？不同于旧feature ROI；仍可能因错目标、裁掉交互上下文、抵消真实平移、放大轨迹抖动或强制方形形变而变差。变化涉及目标/尺度/上下文/运动坐标/宽高比，不能只归因细节增加。若没有充分新证据，不用旧空间正例证明时间增益。

## 固定视图、控制和预算

全部原RC128曝光开发父视频、每父一query，原65物理帧号和完整时间覆盖不变，原八专家位置/证据不变。两种指导：MovingLive以既有T20_Live_41轨迹；MovingSource以同一个Source初始化轨迹。41预选默认、不按GT选种子。每帧沿用旧cycle原规则：cxcywh乘原图W/H，crop宽高=max(16,2*bw/bh)，floor/ceil，裁到图内并执行同旧最小16边界规则，INTER_LINEAR resize224×224。几何提前由冻结框和图像尺寸锁定，不读取GT，不平滑、不扫context/尺寸/帧数，不从预测事件区间截片段。极端框仍按规则产生边缘patch并计数，不在线挑除。

随后同Source原生224预处理、完整文本/subject、两个offset33/32、FP16前缀与原两pass FP32解码。新框丢弃，仅时间logits→原两流envelope→原65/2080动作MBR。静态缓存原Native、Strong=HiResAuxMix与R13 UnionCrop0Mix都精确复用。新增MovingLive、MovingSource各自原q，另各自固定.5Native+.5q混合。主MovingLiveMix，唯一纯教师候选MovingLive，用于分布能力与后续转移裁决；MovingSource/Mix是指导对照，不选控制晋升。无confidence门控/局部参数搜索。

所有矩形按父锁定，相同矩形序列/同视频复用确定输出；实际像素SHA相同可复用，不重复编码制造刷新。不得因窗口是全图就假设resize224方形与原输入一致，只有实际RGB/shape都相同才可复用Native。最大256新视图Source+1个原图零控制=257完整Source请求；记录attempt/completed及真实ResNet/VideoSwin两offset调用，失败保存partial/已完成图像输出/账。不修改旧产物或信号其它进程。

GPU共享非阻塞gpu.lock，完整有限轮900秒含加载与清理，GPU启动4GiB/运行5GiBreserve、host6GiB/disk30GiB，allocator保留5GiB且<=.8。首两父完整双视图、原图控制、独立像素核验后，加载+原图控制+最大实际新视图时间×预锁所需视图数+最大非模型父开销×128+60 <900才继续；不达则保存并停，不删控制或偷偷延长。

## 工程与评分

原图首实际前向Source框/logits/q须精确复原。每父原RGB SHA须等于R13已存实际完整图像，保持同物理帧序列；首两父所有新视图像素用独立标量矩形+NumPy切片/OpenCV重算exact。锁所有实现、数据清单、parser、旧预测/空间/Source缓存及依赖；模型权重/flags前后不变、禁止标签文件访问、梯度无更新。

全128七输出及Source反馈全局seal后，先独立标量重算全部矩形、归一化/两流CDF/混合/MBR效用、旧Native/Strong/Union一致与512空间数组；之后才GT。3584个最终输出以官方与Dense双实现核对；全部2080候选质量由Dense逐项计算，3冻结空间seed父内均值，128父bootstrap，全部零交集/无参考/负例保留。报告fullv/t、边界误差、固定GT支持/未观测实际节点sIoU不变、正负/严重尾/留一、视图面积/移动抖动与实际成本。不得用更小KL/更高响应或相对差控制更好代替完整净增益。

分别比较MovingLiveMix−Native/Strong/Union/MovingSourceMix以及MovingLive raw后验期望质量−Native/Strong/MovingSource。若主hard>=2pp且CI正可冻结一次静态确认；若纯教师expected>=2pp且CI正，即使hard失败也可另立一次匹配实际Direct/freshFixed/Live转移。两者均不是OPD成功或用户目标完成；未超过Strong/Source控制须如实保留，不能宣称方法领先或WHERE→WHEN已成立。无任何一项足够能力则关闭该精确观察配置，不扫crop/seed/混合/温度救援。后续确认按父分离并披露历史曝光；当前128不是独立确认。

完成/失败同turn更新档案check/snapshot/check，GitHub allowlist公开实现、实际协议、全signed/unknown结果/负尾与双ref远端bytes/hash/tree收据。排除私有媒体、GT/原始框、权重、缓存、文本和父ID。空间/CURRENT/MAIN/PAPER不晋升，不新增automation/旧队列。

审计口径：2080个候选质量全部由 DenseTube 逐项计算；独立官方实现对照覆盖3584个最终输出，不声称全部候选质量双实现核验。MBR独立 brute utility 核验每臂13个预定点及选中动作，不声称2080动作全部双实现核验。Source原图/新视图各分 attempts/completed；实测lease秒与3秒清理预留分开保存；已返回Source原始CPU输出先入partial，随后检查资源和后处理；末端模型hash/flags后再检查预算，才写全局barrier。

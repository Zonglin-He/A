> 历史协议说明：本文件保留原OPD正式评估/骨干接口的恢复记录。当前主线已切换BoxFixed；正式Vid与CG仍暂停，不可用新默认配置恢复旧哈希队列。新BoxFixed与时间重放在其他骨干尚未验证，当前试验见[协议](boxfixed_temporal.md)。

# 完整迁移评估

**2026-10-20划分审计已纠正：当前使用TA作者实际公开包的HC1902条及对应修订框；Vid10303成员不变，评分框闭端点与used_segment末端按官方loader还原。raw3482旧计划及295条partial单独保留，不作本次完整官方结果。详见[划分审计](split_audit.md)。**

总任务按用户2026-10-20重新授权包含TA-STVG与CGSTVG。最新顺序明确为：先完成TA-STVG全部工作和GitHub同步，再开始CGSTVG正式任务。当前有限控制器只排TA两个方向，CG不自动接续。两个backbone各做VidSTG→HC2、HC2→VidSTG的Source／Full配对，共四组配对、八个主表设置。TubeDETR不在本轮执行范围。

| 目标域 | 完整官方划分 | 查询 | 父来源 | Source训练域 |
|---|---|---:|---:|---|
| HC2 | 作者公开包评估文件（论文称验证集） | 1902 | 227 | VidSTG |
| VidSTG | 官方测试划分 | 10303 | 732 | HC2 |

清单在推理前锁定；完整保留所有查询、无更新、零时间交集和负例。按实际视频SHA与物理帧网格排序，只为复用输入，不根据GT或成绩改顺序。每query临时状态与优化器重置，全部网络权重冻结。128条离线选参合法；完整集合含开发来源，主结果照常报告并单列已知开发曝光，不能把整个集合说成独立确认。

唯一数值配置仍为configs/default.json，seed41。HC为lr0.014、tau0.14、40步、初始化gamma0.5；Vid为0.01、0.5、30步、gamma1；alpha均1，双尺度0.05/0.25、32候选。CG直接迁移这些常量，不重新选参；其初始化校准常量也从TA既有源域设置迁移，不误称CG重校准。

单ReferDINO同一批八帧证据同时用于初始化与OPD。Strong使用各backbone自身的448时间后验与原生六层后验，原物理MBR不改；CG不借用TA时间logits。Source与Full共享同一源检查点和全部输入。Source也使用配对实验既定的原生时间物理MBR，这不是照抄论文中的域内成绩。

TA五行消融为Source、Spatial、Temporal、WithoutOPD、Full，对应无组件、仅空间、仅时间、初始化＋时间、完整。一次真实完整预测保存所有必要数组，后续五行评分共享这些封存输出，不重复执行五遍推理。主表按query等权；父等权结果和按父cluster bootstrap的query差值另报。vIoU@阈值严格沿用官方大于号。m_sIoU是固定密集GT帧集合的官方空间指标，与预测时间无关；另外报告实际特征节点、观察节点与未观察节点。边界误差按输入清单的显式fps换成秒，缺失不猜测。

## 工程加速与成本

Source、解析器和独立进程ReferDINO权重常驻CPU，按需转到GPU，避免每query反序列化与启动。模型内部跟踪状态不跨query保存。教师缓存绑定真实八帧RGB、原始文本、位置、物理帧号、模型/代码/配置/依赖指纹，并校验payload哈希。不同学生采样key不能复用适应结果。

只缓存文本融合之前的纯外观／运动编码，键包括真实归一化张量、RGB、帧网格、精度与模型实例；返回独立张量，Video-Swin保留全部层但按实际访问搬到GPU。HC本轮1902条的完整RGB网格均不同，禁用这项无收益缓存；Vid有2509个不同网格，使用4GiB有界CPU缓存。一次RGB解码仅在字节哈希、大小、网格完全一致时复用。

CG还将ROI整数坐标一次搬到CPU，替代逐scalar同步；原坐标round/ceil、切片、clone、mean与梯度不改。两个真实方向样例已对全部保存数组做逐位回归。不会用减小步数、降低精度范围或跳过难例换速度。

逐query记录真实端到端wall、适应时间、专家实际调用/缓存次数及student/teacher allocated CUDA峰值，成本包括实际执行的ReferDINO和Strong。缓存后wall与冷启动延迟分开，Source224组件时间不能冒充独立Source端到端测速。在两个方向完整封存评分后，按输入长度和视频哈希预先规定的规则，每方向最多选择16个不同父来源，分别实测Source独立推理及完整链路；关闭RGB、视觉和教师结果缓存，保留权重常驻，热身与加载单列。必须逐位复现已封存预测才发布耗时，未完成则留待测。

## 执行与恢复

```bash
# 仅核验并锁定计划，不启动模型
.conda/tubedetr/bin/python -B tools/run_formal.py
# 显式执行当前TA两个任务；共享原GPU锁，独立worker进程
.conda/tubedetr/bin/python -B tools/run_formal.py --run
# 仅在确认前次进程退出后，按完全一致代码/配置/清单恢复
.conda/tubedetr/bin/python -B tools/run_formal.py --run --resume
# 只从完成封存和评分的job生成公开安全表格
.conda/tubedetr/bin/python -B tools/report_formal.py
```

当前控制器顺序为TA-HC、TA-Vid；CG-HC和CG-Vid暂停，不在本轮自动队列中；不是循环任务，不建立automation。出错保存失败状态立即停止，不删结果、不跳过query、不自动重试。每个完整清单全部预测封存后才在独立CPU进程读GT评分。运行状态、原始清单和未脱敏结果保留本地outputs/formal；Git只公开配置、源码、清单指纹、状态和脱敏全signed结果。

外部TTA旧结果仅在checkpoint、完整样本集合、时间读出与指标都兼容时复用；不将旧小面板结果冒充本轮全量结果。定性图仅封存评分后生成，展示空间、时间、共同改善及失败，原RGB/GT与真实查询不上传Git。

两个TA任务评分完成后，控制器有限串行执行两方向独立效率补测、四类定性输出、公开报告与档案check/snapshot/check，结束状态为awaiting_github_sync。该状态不启动CGSTVG；必须先核验TA结果公开同步。

当前两个方向的清单哈希和离线标签哈希共同锁定；标签版本改变将拒绝续跑/评分。区间仍用官方半开运算，而GT框字典包含官方闭端点，不统一改成end+1。输入采样沿用本轮既定物理网格，并非TA官方loader的128点等距设置；Source与Ours配对相同，不将结果冒充论文域内推理复现。

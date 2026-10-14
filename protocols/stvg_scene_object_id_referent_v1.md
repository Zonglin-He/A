# 同屏稳定Object-ID spatial先验验证 v1

2026-10-14 人类追加“spatial branch再做一个这个”，并明确已有结果边做实验边上传GitHub。新scope独立保存，不覆盖前42教师/72fits/168指标真实有限结果。

## 实质差异与边界

前scope每条候选是独立四帧视频；本轮在同一帧同时显示候选框/稳定数字ID/固定canonical颜色，避免只把R1当单帧Mask logodds scorer。Source＋原marked/masked Top3的已封存相对运动假设复用：43union/18legal/25reject，最多Source＋前4nonSource截断到5；共17实际几何候选，4例只有Source，不能补GT候选或假造其他实例。所有Hypothesis中心=原anchor网格位置+Source相对运动，原四观察位置/wh保持；只是轨迹假设，不保证真实实例关联。无新tracker/光流/DINO/YOLO/SAM/模型下载。

此MVP同时改变同屏呈现、彩色ID渲染、最多5候选与最小native-ID输出；属于实际输入/任务协议screen，不能把收益单因归给“只同屏”。后续局部光流/运动prior loss是不同机制，本轮不混入。原Temporal Caption→own Query Alignment+Adam保持。

## A/B 固定真实输入与调用

原HC2/VidSTG各4历史曝光Source，clean canonical video CPU streaming decode原4帧并匹配旧Source RGB/frame/query/geometry hashes，不在旧marked.npy上叠框。每case5calls：base、ID循环置换（N1 visible1变2，只检验ID格式）、同域下一query、Source_removed、unmarked_none，共40冻结R1 greedy，max96tokens。每call一个4-frame448完整场景video，同预算；实际时间戳图上显示，encoder fps=1/mean原gap，非均匀spacing近似披露。不是完整连续tracking或官方原生检测实例分布。

native JSON只要求合法displayed id或0，reason可选；代码围栏无损去除，非法/截断保留unknown，不扫prompt。Source_removed剩余候选不保证错误，不能叫无目标；空pool与unmarked_none仅测试无可选ID应返回0，不表示视频中目标消失。所有job/像素/源码/runtime预锁后执行，40输出全seal后才GT评价。

## C 空间OPD条件接续

本新阶段A/B不默认重跑已完成72实际diagnosticfits。至少一条base选择稳定(与ID置换回映射同几何)的nonSource且格式合法/未截断，才启动本scope固定8query×Direct/IoU/DIoU各5步的有限真实Teacher-reference更新，unknown/不稳定保持No-update；是否启动只看GT-free元数据，GT不用于选择来源或参考。主要Frozen/T/新S/TS；同Source/canonicalframe/joint1792/原域config/32sample/同key。原Source与离线GT Oracle控制在接口和hash完全匹配时精确复用，明确哪些是旧真实fit，不能冒充新增fit。

若无稳定替代参考，则当前有限A/B screen封存负/neutral/unknown并关闭，C条件未触发，不称学生适应失败或待恢复义务。前72实际fit说明正确几何参考在这8例可被学习，但不证明新Teacher效果。GT Oracle任何诊断保持offline非部署，不混无标签结果。仅在本screen成功并另行固定独立名单时再考虑确认，不自动扩大。

## 资源/公开

单own GPU unit，RAM high/max/swap16/20/1GiB，GPU入场22GiB/RAM8GiB/盘底30GiB；只停本unit的failclosed guard与旧scope一致，不恢复人类停止旧queues，不设置全轮deadline，不建monitor、不晋升CURRENT。

实际root/source/scalar/完整像素view、全部匿名正负unknown/cost/code/protocol、逐远端bytes/SHA/Gitblob/mode/tree/ref、archive check-snapshot-check后仅当前有限scope关闭。官方STVG-R1将coords reformulate实例ID的来源：https://stvg-r1.github.io/ 与 https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee6630dcbcff857026e474fc857aa9f0-Abstract-Conference.html；不继承其完整检测跟踪成绩。

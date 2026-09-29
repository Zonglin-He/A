> Superseded by the latest teacher-independent native parameter rollout route. This completed result is preserved as history.

# S0.5：传播改善了未观察帧，但在线选择仍未验证

本轮已完成96个条件，复用S0的H与5帧Sa2VA masks，新增expert调用0、下载0；仅固定三步×.004 H_app更新，原生动态TTS/ASA/query/decoder保留。这里的oracle是离线候选上限，尚无部署selector、OPD或persistent adapter。

| Corruption诊断 | S0 | S0.5 | 配对变化 |
|---|---:|---:|---:|
| 未观察帧增益，同12来源 | +0.3145 pp | **+1.1513 pp** | **+0.8368 pp** |
| 观察帧增益，同12来源 | +11.9481 pp | +4.3608 pp | −7.5873 pp |
| 整条tube oracle增益，16来源 | +0.8993 pp | **+1.5800 pp** | +0.6807 pp |
| 六层候选∪轨迹oracle增益，16来源 | +1.3886 pp | **+2.1408 pp** | **+0.7522 pp** |
| 固定第三步整管增益，16来源 | +0.4344 pp | +0.7917 pp | +0.3572 pp |

主诊断的配对变化95%源bootstrap区间为[+0.2752,+1.4881]pp，并集变化为[+0.1471,+1.4383]pp。未观察帧按整管sIoU最佳候选读数，不逐帧拼GT管；全16来源未观察增益另为+1.3595pp，不能与旧12来源+.3145混比。源内五种corruption平均后再源宏平均；原16个开发曝光VidSTG父源/16query，同Vid-source checkpoint，不是未见测试确认。

结果支持这版“传播后扩展”改善候选支持：相较S0，信息对未观察位置的作用变大了。观察帧收益同时降低，这是把等权监督扩展到更多帧后的实际取舍；不能声称证据传播是唯一被隔离的因果因素。并集2.14pp比原1.39pp有提升，但没有达到附件举例的3–4pp，效应规模仍有限。

公式按附件实现：原始token前景/背景均值的cosine差，加所有前景reference token的最大cosine，各占.5。附件未指定E如何变成框，本轮在评分前固定：mask area投影到7行左右的原生grid、occupancy≥.5；参考前景/背景E类均值中点为阈值，未观察帧阈值mask取包围框，观察帧保留原mask框；沿用同一L1+GIoU。阈值校准包含local self-match，是本版启发式，不是HTR复现或已验证置信度。

实际66/96条件成功形成前景token bank，新增6022个未观察frame-cell目标。其余30条件保留S0稀疏证据：9条件原mask全空；21条件虽有像素前景，但以固定≥.5投影到低分辨率grid后没有前景token。本轮如实保留，不根据结果另调阈值或追加模型。直接传播目标在同11来源、相同有效未观察帧上sIoU为49.5142%，native为48.7243%；传播目标本身没有呈现观察帧专家那样的大优势。

正例Q05/Q10/Q13/Q14的整管oracle增益分别4.0217/3.9502/3.9002/3.0340pp。Q02和Q08的新oracle收益为0，相比旧S0丢失.3978/.9433pp；Q03 oracle只有+.4727pp、固定第三步−2.5373pp，Q15第三步−3.9574pp。固定第三步整管均值+.7917pp的区间[−.4440,+1.9596]pp跨零，16来源为9好/5差/2中性（.1pp），因此不能把oracle的结构性无害当成部署安全。

Clean同样改善：未观察同12来源+1.3792pp，整管oracle+1.7370pp，并集+2.3223pp。因此当前结果支持一般空间细化，尚不支持corruption特有的修复机制。

本轮结论：保留S0.5为有正信号的候选生成版本，停止本轮candidate-generation试探；不追加steps/radius/阈值搜索。下一项有意义的问题是“无GT偏好能否取到这些收益”，可据此讨论最小S1，但本轮没有启动。附件的leave-one-reference-out会生成不同fold的不同tube，后续必须先定义候选身份与偏好汇总，不能直接把相同步号当同一个候选求平均。Reverse-KL与persistent空间接口仍是待检验方案，不是当前方法。

运行GPU进程71.55秒（含模型加载、传播、288次下游梯度及2次实际编辑后的完整回插），无失败、无额外专家。六个CPU测试通过；960次双实现指标、576项独立loss、96组area投影、66组NumPy affinity、6022个框几何、1728项公开标量及240项配对统计通过。原生B0精确、text/motion/模型参数保持，全部预测封存后只读取原16条GT用于评分。代码与匿名标量结果公开；媒体、mask、H、原query/ID及GT坐标留本地。生产CURRENT与temporal方案不变。

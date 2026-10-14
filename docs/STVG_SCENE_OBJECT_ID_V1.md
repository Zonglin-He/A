# 同屏 Object-ID 空间验证｜固定八例真实结果

完成40次冻结STVG-R1请求及24个Teacher参考实际fit（120步），全部full-model重新解码后才评分。每域4例均为历史暴露开发来源；0新时间fit，旧QA时间区间精确复用。这里的Teacher参考来自学生候选池，不是独立几何真值。

## 预锁主结果

vIoU单位为百分数。T+S为新实际空间解码与旧锁定QA区间组合。Direct仅是本次用户授权诊断，不是论文比较门槛。

|方案|HC2|VidSTG|
|---|---:|---:|
|Frozen Source|22.210455|14.490136|
|Saved Projection|27.851281|22.016466|
|Saved QA temporal|31.286031|22.441929|
|Teacher Direct|22.210455|15.075741|
|Teacher IoU OPD|22.374139|14.133965|
|Teacher DIoU OPD|22.370390|14.113795|
|QA + Teacher Direct|31.286031|23.046034|
|QA + Teacher IoU OPD|31.416467|21.793376|
|QA + Teacher DIoU OPD|31.423857|21.767021|

## 候选及Teacher反馈

原43平移运动假设18四帧合法、25整条越界拒绝，不裁剪、不用GT补候选。本轮保留Source+最多4个原顺序合法备选，共17条；4例只有Source。clean RGB同屏标稳定ID，保持原四观察位置和每帧Source宽高。冻结BF16/SDPA R1，不使用DINO/YOLO/SAM/新跟踪。非均匀真实时间以文字标注，视频encoder使用平均间隔均匀近似。

40/40格式合法、0截断，ID换标8/8保持同几何；wrong-query 5/8偏好变化，其余不变。unmarked-none 8/8返回0，但它测试无可选ID，不表示画面没有对象。Source-removed仍可能存在正确备选，不能称负对象控制。7/8主选择Source，仅V03稳定选择ID2（旧窗口4）。C阶段触发来自GT-free稳定替代，随后固定全部8例×3算法，没有按GT筛例。

V03完整Tube vIoU从6.006629提升为Direct8.349050、IoU8.246636、DIoU8.233786；QA+IoU从6.891628升到9.228748（+2.337121pp）。这是真实局部正结果，应保留。其原四监督位置仅一帧有GT，参考ID2该帧IoU .158103而Source .445717；其余三帧unknown，不足以断言整条参考对象错误。参考稀疏IoU与最终学生完整Tube效果必须分别评价。

同屏提示同时改变多候选呈现、颜色/边框badge、候选上限和输出格式；不能把相对上一轮变化归因仅同屏。Source运动模板和固定宽高限制仍在，未运行光流/局部匹配/motion loss。

## 实际更新、控制复用与事后消融

每fit原1792 Query/LN接口、原域参数、同32 antithetic Gaussian种子、固定5步骤，无调参、GT最佳轮、carry或source cap。IoU和DIoU保留原OPD，DIoU仅替换奖励为IoU−center²/enclosingdiag²。7个Source-ID例仍按预锁主矩阵实际更新，因此存在自蒸馏漂移。

旧24个真实GT Oracle fit精确复用，本轮0新增Oracle训练；已对旧文件SHA、Source模型/帧/输入绑定再次核验。GT Oracle仅离线可学习性控制，不算无标签方法成绩。全部120个官方/独立指标最大误差1.2212453270876722e-15，24fit/120步骤奖励、权重、likelihood、Adam、Source和final cached/full读回通过。

附带Source-ID→NoUpdate只从现成输出作事后读出消融，在主矩阵指标读回后声明，不是预锁主方法或新确认；无新模型/优化调用。规则只用Teacher ID，不能把观察到的收益当新方法已验证。

|事后读出（Source-ID保持F/T）|HC2|VidSTG|
|---|---:|---:|
|direct|22.210455|15.075741|
|iou|22.210455|15.050138|
|diou|22.210455|15.046925|
|T_direct|31.286031|23.046034|
|T_iou|31.286031|23.026209|
|T_diou|31.286031|23.018286|

## 实际成本与证据边界

Teacher input tokens=29802, visual=20480, output=240；40请求端到端8.729875s，generate+sync 8.361589s，GPU peak allocated=16887108608bytes。上述请求时间不含cold load/官方资产hash/输入准备；学生阶段launch→exit=147.653407s（包含阶段加载及读写），二者不能相加成每query部署延迟。

own cgroup 16/20/1GiB与30GiB磁盘底限，finite worker退出0，0观测OOM。Luna独立source输入审计在初始key库存时反序列化query-bearing JSON，未访问/打印/比较原Query，但因此未独立clean-video解码；这个窄审计偏差披露，根实际准备已核clean decode RGB hash，根重新核全部job/runtime/response字节与输入/数学/指标。它不是新科学或GPU故障。

全部8例的正负neutralunknown、Teacher控制、监督/未监督帧、参数/输出位移、实际成本均附匿名JSON。16私人contact/256显示slots实际根像素查看，private RGB/Query/GT坐标/weights/state/optimizer不公开。保留Source/QA与所有历史失败，不恢复旧队列或更改CURRENT。结论仅限当前8历史例：ID提示协议可运行并出现一例局部增益；预锁OPD没有稳定双域平均收益，既不自动扩大，也不据此否定整个MLLM空间路线。

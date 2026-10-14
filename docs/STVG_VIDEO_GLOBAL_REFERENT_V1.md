# Video-Conditioned Global Referent Verification｜真实八例结果

本轮完成冻结R1四帧验证、Source/Teacher匹配48真实fit以及独立离线GT参考24诊断fit。旧Temporal Adam＋own Query Alignment精确保留，无新时间Teacher或时间优化。八例全部历史开发曝光，不是未见确认。

## 完整Tube结果

vIoU为百分数；离线GT各臂不得作为无标签方法成绩。Direct是当前授权诊断，不是论文比较门槛。

|方案|HC2 vIoU|VidSTG vIoU|
|---|---:|---:|
|Frozen Source|22.210|14.490|
|Projection (saved)|27.851|22.016|
|Original Adam + QA (saved)|31.286|22.442|
|Teacher Direct|22.210|14.490|
|Teacher IoU OPD|22.374|13.513|
|Teacher DIoU OPD|22.370|13.490|
|QA + Teacher Direct|31.286|22.442|
|QA + Teacher IoU OPD|31.416|21.140|
|QA + Teacher DIoU OPD|31.424|21.114|
|Offline GT Direct|28.615|23.093|
|Offline GT IoU OPD|27.628|23.043|
|Offline GT DIoU OPD|27.683|22.996|
|QA + Offline GT Direct|39.047|33.944|
|QA + Offline GT IoU OPD|37.716|33.633|
|QA + Offline GT DIoU OPD|37.767|33.576|

Source与Teacher各三算法预测精确相同：主Teacher选择全部Source，严格格式未合格者明确Source fallback。T+S为新实际full-model空间重新解码与锁定原QA时间区间组合，不是本轮新时间训练。

## Teacher与候选条件

Source＋marked Top3＋masked Top3共43去重窗口运动假设；四帧恒定中心平移保留每帧Source宽高，18合法、25越界整条拒绝、不静默裁剪。四例只剩Source，因此不能用8例宣称8次有效候选排序。原四观察位置保持；全上下文/原时间序，非均匀真实时钟文字标注但encoder采用mean-gap均匀近似。候选不是已证实同一实例跟踪。

42 calls严格完整JSON合格30/42，原生ID无损诊断解析42/42；诊断解析未重新选择方法参考。ID换标保持同几何7/8；wrong-query偏好改变8/8。Wrong query可能语义重合，Source始终第一段，尚未排除视频顺序偏好。

这轮主选择没有增加位置纠错参考，不能把格式修复视为几何收益。原四帧支持与恒定位移有额外覆盖限制，不能把这一配置负结果归因 solely 到R1没有空间能力。

## 实际更新及离线诊断

每fit joint1792(Query256/LN1536)，原域lr/sigma/tau，32 antithetic Gaussian、相同种子key、5固定步骤，无sourcecap/最佳GTstep选择/调参。DIoU仅reward替换IoU−center²/enclosingdiag²(λ1)。

方法actual optimizer steps=240；Oracle actual steps=120。24 Oracle参考fit仅在方法8来源全seal后开放GT，原四观察位置有20个可用合法GT框，12缺失未知，不补帧。Direct保持原计划4分母；OPD保留原valid-anchor mean语义，Oracle不同框数是诊断限制。

独立奖励/权重/likelihood/Adam/Source/fullreadout及逐指标算术读回另附根收据。source所有模型权重恢复，原预测/原teacher/window负结果不覆盖。

Oracle参考为真实GT框而非候选池GT最佳。它检验有正确稀疏几何监督时参数是否能执行位移；候选Oracle只作候选上限，两者不混称。Oracle fit内legacy GT_read=false仅表示fit函数未打开标签文件；其anchors已来自GT，整个Oracle阶段GT_read=true/offline_nondeployment。

## 成本与边界

Teacher输入tokens 48594、visual tokens 36864、输出tokens 1248；42 requests端到端(输入读入/处理/传输/推理) 35.255668s，模型generate+sync 33.519341s，peak allocated GPU 17900143616bytes。不包含cold load/资产hash/预处理准备阶段；worker launch-exit另外公开。

16/20/1GiB own cgroup和30GiB盘底；无新DINO/SAM2/YOLO/下载/监测/旧队列恢复/CURRENT晋升。Oracle首次边缘框float32反算tiny越界已保存，另存1e-7编码容差兼容，无框裁剪/科学参数改变；首Direct5完整快照精确接续。无观测OOM，未知未来安全性。全部正负neutralunknown/cost/源码均保留。

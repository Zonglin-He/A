# Video-Conditioned Global Referent Verification v1

2026-10-14，人类明确执行有限P0-A/P0-B/P1。旧单帧全图Top1退化不作为本机制停止门；本轮是四帧referent反馈与真实空间可学习性分离检验。旧两个有限FINAL保持，原Temporal Caption→own Query Alignment+Adam五步精确复用，不用ridge/trust替换。

## 固定来源与候选

原HC2/VidSTG各4历史开发曝光案例，TA-STVG两个原跨域source checkpoint。不是fresh confirmation，teacher训练重叠unknown。候选来源仅旧76网格已seal的marked Top3、masked response Top3和Source去重并集，分数采用旧实际float32值，降序及row-major ID稳定打破并列；不用GT shortlist。原第二observationposition为anchor；候选相对Source的中心Δ固定平移到原四观察位置，各帧保持Source w/h。43去重window ID假设：18四帧合法、25整条越界拒绝。越界不clamp、不移动观察帧、不得按GT补候选；4案例只有Source。候选是运动假设，不是已证实跟踪/同一实例。原四frame Source RGB/视频/frame map/checkpoint/state custody逐hash绑定。

## P0-A 四帧视频Teacher

冻结原STVG-R1-7B BF16/SDPA，不量化，不新模型/下载/YOLO/SAM2/DINO。每条候选独立四帧完整场景448letterbox、红框stableID、原时间序。每候选1次individual native JSON verdict(18calls)，每query三次joint candidate selection(base/ID反转/wrong-query)(24calls)，共42固定calls，上限80无搜索。wrong-query同域固定下一个query，可能语义重合，不把未改变选择直接当失败。Source-only ID swap从1到2，仅格式/ID理解不能检验排序。joint每候选四帧等预算，最多6条24视觉帧；实际token、模型时间与request-end-to-end分别保存。原四时间非完全均匀：图/文字提供实际时间戳，encoder fps使用端点mean gap，明确近似均匀pacing，不冒称完整运动/官方原生tracker精度。输出max128tokens、greedy、不prompt搜索。严格JSON合法ID解析，不用ABC单帧logodds当IoU。

base合法非0输出定义actualTeacher参考，不以GT/控制结果筛选；unknown/非法/截断采用明确Source fallback并保留raw结果。控制只诊断，不作为任意更新总门。当前仍有限diagnostic方法，不宣布已建立正式可靠性gate。

## P0-B / P1 实际参数适应

所有8来源使用Source和actualTeacher各三条真实fit：Direct、IoU-OPD、DIoU-OPD，共48fits；四原稀疏位置，joint1792(Query256/LN1536)，原每域lr/sigma/tau/32 antithetic samples，5steps，same key/noise/Source reset、无source RMS cap。DIoU只奖励改变为IoU−center_distance_squared/enclosing_diagonal_squared(λ1)，不改采样、KL、Adam或温度。Direct作为本轮授权诊断对照，不是论文必须比门槛。

无标签Teacher和student方法输出完整seal后才GT评分/Oracle reference生成。另24fits使用原四位置可用GT几何参考（missing保持unknown、不补帧），同Direct/IoU/DIoU，为离线非部署正参考诊断。Oracle标签不得进入无标签Teacher选择、方法训练、调参、回退或正式成绩。不存在任何GT的案例保留empty no-op。报告实际每arm参考数/optimizersteps，不以宣称的72fits冒充全部有更新。

主F/Projection/原T及S/T+S真实最终full-model re-decode，Source全模型恢复/缓存完整decoder逐值一致；P1固定最终轮，不挑最佳GTstep，不按GT选lr/tau。完整Tube vIoU/tIoU/sIoU及Source负尾、观测/未观测帧、参考quality、候选Oracle与正确参考实际参数移动分开。旧Temporal exact复用需接口/source/features/hash验证，不新temporal教师调用。

## 资源与关闭

单一own GPU unit、MemoryHigh16GiB/Max20GiB/SwapMax1GiB；teacher入场GPU22GiB/student8GiB、RAM8GiB、盘底30GiB，own RAM18/anon14/host余6/GPU余4提前保存并只停ownunit；三次采样错误failclosed。无全轮deadline、无限搜索、监测或旧queue恢复，不晋升CURRENT。

root独立source/math/完整预测seal后评分，全部匿名正负neutralunknown、实际PDF/私人像素view、public远端bytes/SHA/Gitblob/mode/tree/ref验证与archive check-snapshot-check完整后本有限scope才能FINAL。prepared/worker退出/合同pass不等于科学收益pass。

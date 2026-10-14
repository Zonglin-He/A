# STVG 真实复验：优化稳定性与全图空间证据

本轮先完成原8个历史开发样本（HC2/VidSTG各4）的低秩/trust/offset比较和保守空间更新，再执行新Box-scale全图资格。GT只在相应预测/教师输出完整封存后离线读入；没有用GT决定候选、参考、步长、参数或最优步。历史开发来源不是独立确认集。

## 时间与空间优化已实际执行

40个CPU时间fit（5模式×8）、8个新Direct空间fit、8个新保守OPDfit，80个新空间optimizer步骤；8次Source真实重捕获、16次新完整模型final readout。原Adam参数/loss/interval逐步与旧结果精确一致；旧Source/OPD为封存exact reuse；时间5种模式新增teacher调用和model前向均0。

| 时间分支，固定Frozen空间框 | HC2 vIoU% | VidSTG vIoU% |
|---|---:|---:|
| Projection |27.851281|22.016466|
| 原Adam |31.286031|22.441929|
| 一次ridge |23.079210|23.187334|
| 五轮残差ridge |22.421633|23.187334|
| ridge+full-span trust |22.213819|23.187334|
| per-offset target+geometry |22.554401|23.187334|

原Adam固定目标7/8末步增大，子空间外参数分量max .542718。低秩四模式max<=1.71e-14；trust/per-offset全合法区间reverse-KL maxima .00999636597/.00999683877，每步真实objective非增。mean feature rank2–3，stackedoffset rank3–6，同512参数。P2同时改变offset features及targets/loss，是联合offset机制比较，不能独立归因其中一个。低秩first-update对离线GT下降方向cosine：HC−.249834/Vid+.124126；peroffset−.085785/+.091329。梯度只诊断。

空间Native时间S0/Direct/oldOPD/conservative：HC22.210455/21.556488/21.212784/21.982775%，Vid14.490136/14.490136/13.512786/14.701888%。预锁trust时间+conservative为HC22.053446/Vid23.489750%，不能说稳定双域替换旧T。未监督GT帧sIoU变化：oldOPD HC−.208376/Vid−2.801260pp，conservative HC−.363659/Vid+.499251pp。40更新全部缩步接受、184尝试、14160源帧坐标RMS<=.01，真实零全reject；拒绝恢复仅源/CPU mock evidence，未被实际GPU路径覆盖。304官方/独立评分max误差1.221245327e-15。Direct只诊断，不是论文必须benchmark。

## 新全图参考实际P0

每例固定第二个原observationpos一帧，Student只提供w/h，生成76个合法同尺度coarse窗口完整覆盖图像，0半步、多scale、GT shortlist。R1完整画面radius8局部blur差分及固定红窗ID1两协议共160categorical请求。原160保持immutable。随后公开声明追加24同位置Source控制（8原图repeat、8Sourceblur、8Sourcemark），全24已seal；8原图跨teacher load logits/probs/logodds精确相同，未改变Query、anchor或Student形状。

8锚点只有5有GT（HC3/Vid2），3unknown未GT换帧/未填0。以下按有GT的锚点平均，IoU以百分数表达；不是Tube vIoU。

| 完整coarse网格+原Source候选 | HC2 n=3 | VidSTG n=2 |
|---|---:|---:|
| 原Source IoU |40.734415|66.273113|
| union候选Oracle IoU |50.603316|70.144486|
| 遮挡选择IoU |24.290323|36.530342|
| 标记选择IoU |35.737548|45.388188|

允许Source位置后，遮挡5valid中2正/2负/1neutral，标记0正/4负/1neutral。全部76候选排名/反馈/GT质量、Source控制、3unknown完整保存。Oracle说明候选位置在部分例子有潜力，但Teacher未提供稳定更好参考；图像响应/可见性分数不是bbox真值。

CPU32 antithetic局部Gaussian两参考协议共16探针中2组IoU全部0；DIoU λc1全部非平坦。这只是奖励可辨性诊断，0新student fit/optimizer/backward，不是DIoU效果或参数可达性。当前固定单帧recipe未触发后续新Direct/IoU/DIoU fits、4帧关联或完整新T+S扩大，不能从这个有限负结果推导global evidence或STVG-R1路线不可能。

实际R1为原STVG-R1-7B revision c0db00d019bf746f54b67770b82261b2161fed59，8,292,166,656params、BF16/SDPA未量化、15本地原资产逐streamSHA核（weights16,584,414,544bytes），不新下载。160 model+sync17.346178秒、owned worker总wall56.103196秒；24control model+sync2.977674秒、worker wall28.779893秒。完整每request预处理/IO/transfer延迟和gridprep未计时为unknown，不能将model-only写成完整latency。GPUpeak16,735,618,560bytes，own unit实际16/20/1GiB限制pass，未观察到OOM；single-image vs video-training、SDPA sliding-window实现warning均披露，不继承原论文精度。

## 验证与边界

根独立进行了所有源/缓存/teacher绑定、公式/真实输出/scalar再读、全部160输入pixel重建、8完整固定私有参考contact实际pixelview和5页PDF实际view。环境/reader metadata分派错误保留原件，只修新工程入口，0因此追加science模型或GT更新。旧停止队列保持停止，CURRENT SHA146a2caf1633544a844ea33106c83c022c0bebad5f3f49a87bb5f81e6f8568d4，不自动32/域confirmation/main/carry/paper正文或schedule。公开同步及档案核验收据以各scope动态closing为准。

原始归因背景：[RISE](https://arxiv.org/abs/1806.07421)、[D-RISE](https://openaccess.thecvf.com/content/CVPR2021/html/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.html)。DIoU中心距离背景：[DIoU](https://arxiv.org/abs/1911.08287)。这些文献不能替代本轮真实MLLM定位资格。

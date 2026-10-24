# TubeDETR空间sIoU目标的独立参数开发

2026-10-24用户明确要求扩大TubeDETR参数搜索，目标是空间sIoU相对Source转正。此授权覆盖此前“不调参”的限制，但不覆盖已经封存的固定方法实验。新目录tools/tubedetr_spatial_search、results/tubedetr_spatial_search、outputs/tubedetr_spatial_search与旧实验隔离。

Source严格保留原生框和MAP时间；不是把额外MBR改名Source。空间选择目标改为原两域各128独立父来源的等query dense sIoU，所有GT标注帧、插值、未采样区间零值与现有评分完全一致。时间固定为各query既有SimplePath Selected_interval，不调原型/MBR/时间参数，不通过时间收益掩盖空间损伤。未观察帧和观测帧空间分项、尾损和全部负配置均保留。

首次预算每域2048个去重配置、seed20261024。覆盖初始化gamma0到2；八种affinity温度/top-k几何；观察/未观察appearance权重0到4、连接权重0到4、Source保护0到16、候选准入margin；观察/未观察输出混合0到1；宽高0.8到1.2数值校准。包含原生Source、原固定SimplePath逐位回归、初始化、旧Full混合、仅Source宽高校准等独立控制，随后预定固定seed联合采样。不是声称穷尽连续参数空间。新增保护/混合/宽高校准是明确记录的数值后处理扩展，不冒充原固定上游方法，也不将仅校准控制的收益归给传播。

仅复用旧完整seal的原128原生post-fusion空间特征、原八个专家框/原生mask和既有完整五行预测。每条来源先核payload/receipt/Source/帧/八位置；保留所有原生时间和固定新时间。每个域全部2048×128预测先完整seal并核全部payload/receipt，之后另CPU进程才读取标签。推理进程有annotation guard，GT不参与单query候选/准入/回退，不能逐query用GT挑赢家。最大化总体128 sIoU选择一套域级参数；必须披露开发曝光与搜索偏差，不把选择区间当独立确认。

入口`.conda/tubedetr/bin/python -B -m tools.tubedetr_spatial_search.run prepare`首次冻结计划；`run`依次CPU predict/score两域，所有GPU不可见，默认一条CPU数学线程，不影响当前LLaVA或修改其160项绑定源码。状态/进度/失败在outputs/tubedetr_spatial_search，独立runtime/tubedetr_spatial_search.json；日志logs/tubedetr_spatial_search_<ds>_<predict/score>.log。

两域锁定后再验证所选完整10303/1902总体。若所选配置只需既有Full/Source混合或数值校准，可CPU重新生成完整预测seal后独立评分；若需要重新计算完整空间geometry，则新GPU捕获必须等待当前LLaVA全部GPU工作完成并获得共享gpu.lock，不并行新主干，不恢复已删除特征缓存。全量结果不反过来选参数；完整核验发布前不称任务完成。若开发仍未转正，保留失败并由Root分析追加独立冻结的搜索阶段，不修改已评分计划。

# R48 独立审计与唯一评分

本目录只面向 R24 已暴露 D208/V64/M64 的 336 条缓存。禁止读取 FINAL384 的清单、预测、媒体、点云或标签。

`audit48_independent.py` 需要 root 的独立 CPU 审计授权，并先校验代码、协议、输入、caption-only 解析封存和五臂预测封存。它从已封存依存分析独立执行冻结主体规则，复核官方类别白名单、同义词优先级、带首空格文本的 literal positive-map、同一字符跨度 remap、实际 T 上的 CPU64 softmax、所有候选分数及 1680 个选择框。仅官方两臂允许使用标注类别。该步骤不读取标注框。它验证程序执行与数值接口，不等于给文本主体做语义金标；语义核查由预先封存的盲审另报。

`score48_independent.py` 需要不同的单次任务评分授权，绑定独立 PASS、同一预测 seal 与评分程序 SHA。其后才读取历史 P5 已注册 metadata 的精确 object_index，执行 Drone 高度偏移及 float32 GT 转换，并评分未改变的源 6D 框。五臂所有 1680 IoU、所有严格退化比较、全部 split/platform/scene 和 UNKNOWN 均保留。该步骤没有任何反馈回流或参数更新。

缓存没有保存历史 input_ids。因此重建 tokenizer 的 T、offset、current mask 与当前缓存一致，是输入桥接证据；不能声称历史 token IDs 已逐位证明。官方两臂是标注类别辅助的跨度诊断，不是部署方法、oracle 或完整官方模型复现。

运行入口均必须由 tools/night_guard.py --resource cpu 串行执行，且分别等 root 的审计与评分授权。本文档与代码本身不授权执行。

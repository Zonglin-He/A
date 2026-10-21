# HC64缓存时间后验融合：固定一次验证

## 主张与约束

空间、ERef、完整框传播、对应可靠性、alpha.5/context1.5、辅助轨迹及置换全部不变。唯一变化为时间读出：六层分别通过原冻结head形成原物理轴的区间后验，然后平均为qG/qT/qP；比较Global6=qG、GlobalTube6=.5qG+.5qT、GlobalPermuted6=.5qG+.5qP。不平均logits，不重放decoder，不增模型或教师。

Source_Native224为未适应Source原生框与原生224最后层时间，完整方案相对Source评价总体收益；GlobalTube6−Global6评价空间反馈，−GlobalPermuted6评价空间位置。Strong仅旧参考，不设必须超越门槛。另固定同一A框轨迹，只换各时间区间；该行Source区间的固定空间结果不能称未适应Source总体结果。

## 证据与执行

复用已封存outputs/tube_temporal_hc64，严格验证全部旧source/input/receipt/seal哈希。不改旧回执或代码指纹。调用external/TA-STVG/models/net_utils.py原MLP，按原Source loader优先model_ema的规则，只提取temp_embed四张量；不构造重Source网络。保持每offset完整六层head批量形状和精度，G/T/P末层logits、后验和MBR必须逐位恢复旧值，反序列化几何必须恢复原B/C/D动作/框/分数，否则保存失败并停止。

独立脚本scripts/tube_fusion/run.py --run，独立outputs/tube_fusion_hc64。全64输出与receipt先封存；scripts/tube_fusion/score.py --run-after-seal才读原独立标签。保留全部signed变化、严重损伤、起止MAE秒、tIoU/vIoU、长度与10000父来源配对CI。HC64为已曝光开发数据，不称独立测试；默认配置和正式Vid/CG暂停不变。

缓存验证只运行原head6批/query（G/T/P×两offset）、物理后验、原MBR、三次最终BoxFixedDP及工程恢复检查。记录head/load/IO/后处理/全wall；不能当成部署成本。实际部署条件分支仍需原生224、同八专家、传播和两个offset各一次时间decoder，旧预算不被隐藏。

## 依据与边界

[E3M原论文式14](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11011.pdf)把原始全局与空间聚焦后的匹配项相加，支持保留两类信息的设计原则；它不是本轮概率等权融合或TA冻结重放的性能证明，不引入CLIP/新检测器或EM迭代。若真实融合相对关闭反馈及置换没有优势，本轮停止这条反馈方案，不扫权重。

## 已完成与停止决定

全64先封存后GT；指纹fa9f761b40c1b4b66272ccb1d221567ca8aa4d0faf71e5c9292e8a882c6cc947。Source/Global6/GlobalTube6/GlobalPermuted6完整vIoU26.8938/29.6952/29.7235/29.6977%。融合−Source+2.8297pp [.9075,4.8356]；融合−Global+.0283pp与融合−置换+.0258pp均区间跨零，tIoU分别−.1532/−.1140pp，未证明反馈优势。固定空间两项v差−.0137/−.0161pp。按锁定规则停止反馈方向，不扫权重；这不否定完整方案相对Source的整体收益。详见[分析](../results/tube_fusion_hc64/ANALYSIS.md)、[所有指标](../results/tube_fusion_hc64/report.json)及[正负表](../results/tube_fusion_hc64/signed_changes.tsv)。

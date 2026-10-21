# 有限双向调参与新全量协议

用户在2026-10-22授权两方向调参、根全程盯调参，结束后全量由一个Luna max巡检、异常根修复续跑。本协议覆盖上一轮停止调参限制；旧formal8405与CG仍暂停，不恢复旧方法队列。

具体机制、每阶段决策、全部配置/正负/复核、缓存回归见[本轮结果](../results/boxfixed_tuning_v1/ANALYSIS.md)、[预锁协议](../results/boxfixed_tuning_v1/protocol.json)、[逐阶段记录](../results/boxfixed_tuning_v1/decisions.jsonl)。每方向64开发父，实际14/17配置≤18，原Source/八观察/完整框候选/gamma/精度/指标不变。四参数为最终时间温度、观测权重、几何保护及连接；非OPD学习率/步数。

HC开发选择(1.5,.75,.2,.25)、Vid(3,1,.2,.25)先锁后32×2一次复核；完整v/t均下降，Vid未观测及严重损伤亦失守，按事前规则完整回退两域(1,.5,.1,.25)。不据复核改门槛、挑第二组或扩大搜索。数据全域已有历史开发曝光，不能称独立测试；事件时长差是观察到的竞争解释，不是GT时长在线门控。

新全量profile是[configs/tuned_boxfixed_v1.json](../configs/tuned_boxfixed_v1.json)，文件名是本轮工作流标识，数值为明确fallback，不能把Final叫作调参获益版本。生产默认Strong未变，新队列显式用224/Global6旧hull/原MBR/BoxFixed，无448、条件decoder、OPD、训练或新专家。原正式结果保留不可混用。

全量固定HC1902/227父与Vid10303/732父，Source与Final同输入/源权重/指标配对，全部query不得跳过。Final与Untuned此时完全相同，作为alias一次计算/一次评分，既保留过程可比，也不制造第二次运行。每域全seal后独立CPU固定官方标签，主均值按query，区间按parent bootstrap并保留query计数加权；数据包含开发曝光如实披露，不根据全量结果选参。

全量唯一plan fingerprint为`2ebd134dfd055c8e17e01859b706d56d187b15b8907a8129c49669539d84ea6e`，位于私有outputs/boxfixed_full_v1。缓存索引192输入按全部metadata除路径、runtime/权重绑定，每文件SHA验证。RGB512MiB LRU按视频内容/精确网格；视觉4GiB LRU按实际归一化tensor/mask/duration/autocast；不共享跨query文本/fused/decoder。两域miss/hit共4已有query工程回归全部Source/E/位置/六层/31几何张量逐位、0专家。记录命中、key/transfer、Source/专家/证据/后验/DP/IO及当前成本，cache原成本不计新耗时。Source与专家分进程allocated峰值不冒充全卡NVML总峰值。

```bash
# 仅首次新队列，已有完整资产/锁定profile时
.conda/tubedetr/bin/python -B scripts/boxfixed_tuning/full.py --prepare
.conda/tubedetr/bin/python -B scripts/boxfixed_tuning/full.py --run
# 仅核原controller及全部自有子进程实际退出、指纹一致后
.conda/tubedetr/bin/python -B scripts/boxfixed_tuning/full.py --run --resume
```

controller.lock及PID/startticks防重复；每query预测+receipt原子提交，resume核plan/源/权重/profile/清单/缓存索引与已有预测SHA，不放宽fingerprint。孤立未提交文件封存后重做未完成query，已完成预测不重跑。Luna只读本队列plan/state/progress/log与/proc/GPU/RAM/磁盘/gpu.lock，写私有monitor回执，半小时复用，健康静默。异常根保全证据、核退出后自行修复；不得改冻结方法、参数、采样、精度、评价或信号无关服务。代码修复需旧源封存与逐位回归/明确迁移，否则保留失败报告。

全量完成时保存所有Source/Final signed、严重损伤、时间/空间/阈值、parent聚类区间、真实缓存/冷热成本；更新档案check snapshot check，代码/配置/结果及负发现公开GitHub两分支freshCAS，核远端tree/bytes/hash后暂停巡检。不得上传数据、GT坐标、视频query正文、权重或原始特征/mask/状态；不得自动启动CG。

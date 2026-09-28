# 固定16源已存监督前缀与原生支持审计 v2

本轮为 CPU 已存证据审计，不是新推理或新的效用实验。问题：此前最终 head FP32 固定三步源监督出现原生定位负尾；原监督 CE 与原生读出究竟在哪些已保存的语义、时间、框支持和 token 分母上不同？全部原 PANEL16 Vid 训练父源各1 query，读取同 B1、旧 BF16 supervised3、新 FP32 supervised3 三状态；保留全部正负与无效几何。不得按结果换样本。

- 输入只限 source_fp32_head_control_v1 的已封存原件、同16 SOURCE_RECORDS、既有评分结果，以及先前28199单源完整支持/审计。官方 target builder、joint_loss、native generation 与 shared-reference cached 代码仅静态读取。
- 新 model forward/backward/optimizer/native prediction/GPU allocation 均0；不打开视频、target 输入/GT、额外标签池或 checkpoint 权重。仅本地 tokenizer 可加载到 CPU。源标签仅已授权诊断，不进入新更新。
- 逐 query 检查 GT response、实际 frame roster、语义 reference、time tokens、frame-box 支持与三个已存 native 输出。native 1基文本、0基 positions/interval、实际物理 frame ID 必须一致；格式与无效零框保留。
- 用未修改官方纯 target-builder + 原 loss masks 在 CPU 重建**标签侧**完整监督目标。使用 masked dummy prompt，因此不是完整多模态 prefix/position/attention 重放。非 dummy 的原始完整 support 只在28199有封存原件；其余15只有 hash 与 token counts，明确缺项，不自动补 GPU。
- 以独立闭式计数及第二解析器核对类别：NTP/MTP 是执行位置类别，不能命名为时间/坐标。事件分支含 semantic+time；空间分支含 GT box row，包括 time-anchor/box delimiters/coordinates。MTP null padding仍监督。记录全部 label 对应 token 与类别，不截断支持。
- 已保存全部新3 step、最终两种精度 CE，以及旧监督最终两种精度 CE 的 NTP/MTP 分母、分子重聚合。预定容差 abs <=2e-6（原 FP32 chunk sum 与分组 sum 归约次序）；计数/身份/整数 token 一致要求精确。分支 CE 是两组 loss 的 count 加权均值，task 是两分支均值相加。不能用组均值推断 endpoint 或 coordinate 子类 CE。
- CPU controls 在审计前执行：变长 reference/box支持的独立计数，拒绝跳号/倒序，混合 NTP/MTP 加权而非简单均值。真实16审计不得热改阈值；失败写 FAILURE 并保留，隔离修复。
- 独立 root readback 重新解析48已存 native、重新计算分母与 CE 加权变化，核对 report；不重算新指标或读取评分标签以造新效果。
- 输出 source_prefix_support16_v2；新文件总量 <=20MB，始终保留8GiB。CPU耗时独立记账；GPU累计保持38820.55116519004秒、cap=null。CURRENT/旧暂停/历史 optimizer 勘误/多次目标开发曝光/Vid-only 与 HC transfer 保持。

判断：若语义一致而时间/框支持多不同，只支持监督与原生条件差异的具体范围，不直接称任务根因。若支持完全相同仍损害，保留读出/数值/充分性竞争解释。即使端点监督占比较小，也不是梯度贡献或重新加权有效的证据。现有信息不足则明确缺项。完成后才能提出一个有区别的源协议；不得自动新 target/64/精度或步数网格。

版本修复：v1 CPU 审计在首源物理区间检查因把 event_interval 字典当列表发生 KeyError(0)，尚未生成报告，旧脚本/锁/FAILURE 保留。v2 仅使用实际 begin_fid/end_fid 字段、加半开区间结构控制并写新目录；全部指标/阈值/数据/支持不变，0 GPU 重跑。


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.

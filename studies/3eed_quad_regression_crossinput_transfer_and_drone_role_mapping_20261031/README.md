# R14：Quad 跨输入回归迁移与 Drone 主体/映射诊断

本包保存已执行协议、完整匿名标量、负例、修订历史和审计投影。结论以 [REPORT](REPORT.md) 和 [DECISION](DECISION.json) 为准，本文不自动宣布方法有效。

Quad 使用冻结 Waymo-only source100。A 为 R13 Quad76 的全部既有不等式 G，B 是同场景一一配对的不同图像及点内容76。B 只用于原源特征/固定自身 q0 读出，不构造 G_B、不参与 A loss/LR/gate。真实末回归层 half/full 更新与精确 frozen-feature kernel 对照并列，另有 additive 和 source-size-normalized 直接迁移。即使 B 未参与拟合，末仿射头更新仍与跨输入 kernel 理论等价；数值差不作为独有学习贡献。

Drone 将旧32开发例与近期内容不同32分开报告；64并集只作描述。冻结角色解析与原3D候选映射均不构造新框、不做参数更新。完整 JSON 的单层 codefence 格式修复发生在 GT 前；工程001、formal001中断、工程002及formal002预算/引用契约瓶颈、003仅工程预检与004的GT前授权/审查等历史以 [计数记录](HISTORY_COUNTS.json) 留存，不公开 raw calls。002的visualcap64下61/128未EOS、0valid visual与combined全source是无GT工程覆盖瓶颈，不能据此宣称语义方法失败；若另有002任务标量，其64例作为相关重复观测另表，不计入最终64分母。001进度记录的20reuse不等于实际闭合调用总量；闭合inventory为39=15new+24reuse，二者分别留存。003仅2例工程预检（10new+4reuse），4/4EOS后仍出现reference槽类型把合法target一起丢弃的接口问题，未启动formal003；004只局部UNKNOWN处理reference内部非string槽，主target严格契约和模型/提示/几何保持锁定。首次004工程的主体分支虽通过，却因runner局部变量shadow把实际2case误记为seal n_cases4，故未启动formal004；10new+4reuse与原错误seal保留为元数据失败，变量名窄修不改变算法，004b工程及后续formal另计。最终attempt由root readiness与实际score/prediction/plan哈希绑定，本文不偷换任何未完成attempt为已通过结果。

All5912 targetVAL 记录有 R7/R8 历史模型/评分暴露。本轮独立封存不消除旧标签暴露；distinct content 不等于已证明同一物体、不同实例或独立场景。三场景依赖、零信号、NA、全部正负变化均保留，alias 标量不是独立样本数量的扩增。

仅代码/协议/标量可审计导出：未随包发布原图、点云、mask、features、原始GT/预测坐标、raw IDs/captions/calls、权重及私有输入注册。精确预测重放仍需合法取得官方数据、固定资产以及原 source100 checkpoint/私有注册；本包不独立包含全部运行资源。可执行项目代码依赖已纳入 code/，官方3EED checkout与兼容补丁见 UPSTREAM_PROVENANCE.json。

公开标量包含 Quad 全76与 Drone 全64（旧32/近期32清楚分开），没有删除失败、零变化或负例。后续 build 后仍需独立 privacy 审查，再由 root 单独发布；本脚本没有上传功能。

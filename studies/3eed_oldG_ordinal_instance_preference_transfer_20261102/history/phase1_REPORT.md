# R44 阶段1：旧 G 的不同实例组局部提名，直接使用不更新

## 总体结果与预先数值门

独立算术和 624 个任务值评分通过。受资格约束的固定组代表直接选框，A208 平均 mIoU 为 **24.3321%**，相对 source 的 **19.6636%** 提高 **4.6686 pp**；两个平台均为正。

| 方法，mIoU % | Drone | Quad | 两平台平均 |
|---|---:|---:|---:|
| Frozen source | 12.7430 | 26.5841 | 19.6636 |
| 原 unrestricted 旧 G | 15.9786 | 33.4761 | 24.7274 |
| Eligible 固定 source 组代表 | **15.6545** | **33.0098** | **24.3321** |

Eligible 相对 source：Drone 11 条改善、0 条恶化，Quad 14 条改善、1 条恶化；剩余为不变，所有行仍保留。即使已看到任务评分，也没有据此丢弃 1 条不利偏好或零 IoU 差的样本。任务改善不是独立身份 gold。

预锁 `N>0` 且全体 balanced mIoU 净增量 `>0` 的数值门已 **PASS**，没有加评分后的覆盖或区间门。这个结果只支持 root 考虑另批有限阶段2；不自动授权训练，不证明每条偏好正确、不证明新场景稳健，也不替换精度更高的原 unrestricted G。受约束版本比原 G 低 0.3953 pp，这个覆盖／固定代表代价保留。

## 执行与边界

本阶段只运行固定旧 G 的同信息直接选框，不训练、不调用新 G/VLM、不使用 GPU，也不读取 FINAL/U/T。原 A208 全体保留：32 条满足两个不同完整实例组的资格（Drone14、Quad18）；169 条 G 第一名与 source 相同，没有纠错提名；7 条 G 提名没有唯一完整映射，UNKNOWN/source fallback。没有按评分、覆盖或个例改门。

所有三臂 query、原 source 框、全 F64 分数、组成员、固定代表和每条拒绝原因先封存，随后由独立代理完成算术复核和 624 个任务 IoU 评分并通过。预先的数值进阶门仅为 eligible N>0 且全 A208 两平台等权 mIoU 相对 source 净增量>0；通过只允许 root 考虑另批有限阶段2，不代表反馈全正确、稳健或已采纳。阶段2训练当前没有授权。

## 固定规则

使用原 R24 的 source 概率和原 bank272 完整 evidence：F64=s0+2vision+2position+2relation，qG=argmax F64、q0=argmax s0。只有 qG≠q0、F64(qG)>F64(q0)、两者各自恰有一个非空 R37 完整组，且两个 mask 组不同、成员不交叠时，才接受一个局部组间偏好。

直接输出正组固定原 source 最高分代表（同分最小 query），不挑 GT 好框，也不把重复 query 当负实例。其他情况均不生成偏好。原 source、原不受限 G 和新的 eligible-group direct 三臂共同保留；没有将二选一偏好升级为其他所有候选均负。

不同 mask 组是现有检测/几何的实例假设，不是物理身份 gold。固定 source 代表避免用 GT 选择框，但任务 IoU 改善仍不自动认证语义实例纠错。

## 测量正控

原 R24 A208 的 256×6 source 框、256 source 分数已与 R37 准备数组逐行通过 shape/dtype/value/bytes exact 门：208/208 全通过，无容差、换 source 或重映射。

本阶段 unrestricted oldG 的 query 和框进一步与历史 R29 封存的对应 legacy_G **208/208 精确相同**，不是仅复现均值。F64 与 F32 teacher argmax 有 3 条数值差异，作为桥接诊断保留；主规则继续固定历史 F64，不按任务分数选择精度。

## 封存与证据

预测 seal SHA：`7c52a420dcced4c426caf34056df7f6c0dfc477dd0ad7b7373347926c68ab733`。

- `ROOT_EXECUTION_AUTHORIZATION.json`、`PLAN_LOCK.json`：阶段1范围、代码/输入哈希和预先数值门。
- `crossplatform/round44/direct208_attempt001/`：三臂原框、全部 F64、接受／拒绝与历史 oldG 重放。
- `../qualification/RESULT.json`：原 source exact 门，只证明兼容，不证明 G 或身份可靠。
- `../review/REVIEW.md`：历史已做过 qG>q0 成分、局部 RankNet 与未来 keep 耦合的限制。

独立结果在 `../independent_phase1/`：`READOUT_AUDIT_RESULT.json`、`TASK_SUMMARY.json`、`SCORING_RECEIPT.json`、`ALL208_TASK_ROWS_PRIVATE.json`、`ALL_NEGATIVES.json`。全部 62 条负参考比较保留；它们不是 62 个独立样本。全 A208 已暴露，结果是阶段进阶的开发依据，不是独立确认。

本阶段没有启动参数 TTA，后续 keep 集、偏好分母、memory/最终读出和有限预算必须另锁。若使用 A\eligible 作 keep，集合将成为 Drone90／Quad86，不能继续把它称为原 R30 U189，也不能将与旧22反馈的差别说成只有监督信息一个因素。

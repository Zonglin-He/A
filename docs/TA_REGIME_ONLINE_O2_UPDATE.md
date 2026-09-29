# O2：Regime-Coherent Online Transfer

**出现了局部正迁移，但整体增益很小，尚不能说明 homogeneous regime 已解决 online transfer。**

保持 O1 的方法完全不变：768D linear slow state + common bias，原始 temporal hidden 特征，SGD .001、专家位置单步、alpha1、25% expert budget、原候选和 UniversalVTG critic。只改变 stream construction：旧 C3 前16个曝光 VidSTG source，沿用 O1 的 source-hash 顺序，每个 source 在五条独立流各出现一次。每流固定一种5% transient burst，位置1/5/9/13使用专家；每流开始零重置状态。5%是原 full-input-video burst 占比，不是重新提高图像退化强度。

## 主结果：只看非专家位置

| Condition | nonexpert | 选择改变 | ΔtIoU (pp) | ΔvIoU (pp) |
|---|---:|---:|---:|---:|
| Frame Drop 5% | 12 | 3 | +0.0730 | +0.0214 |
| Frame Freeze 5% | 12 | 0 | 0 | 0 |
| Motion Blur 5% | 12 | 0 | 0 | 0 |
| Occlusion 5% | 12 | 1 | +0.5737 | +0.2056 |
| Exposure 5% | 12 | 0 | 0 | 0 |
| 五类 macro | 60 cells / 12 sources | 4 | **+0.1293** | **+0.0454** |

比较都是 Online Slow-Fast − Budgeted Rerank。非专家位置 Budgeted 就是 Frozen；macro tIoU从28.8343%到28.9637%，vIoU从17.1304%到17.1758%。同一source先平均五条件，再bootstrap12个非专家父源，不能把60 cells当60独立源。macro条件95%区间：t [0,+0.3734] pp，v [0,+0.1319] pp；均包含零，而且只是固定共享状态轨迹上的描述统计，不是重复在线实验。

四臂 all-arrival macro t/v：Frozen35.0408/15.8454%，Budgeted37.0651/16.8205%，Online37.1621/16.8546%，Full41.1714/19.0016%。Full逻辑专家预算80，Budgeted/Online各20。Full只是较高预算参照，不是oracle。

## 正案例和零变化要一起保留

60个非专家位置，vIoU改善2、下降0、不变58。4个选择变化中另2个改变区间但t/v仍为零；本轮无Online−Budgeted的>5pp损害。

- Frame Drop：流内位置3/Q09，t15.7664→16.6423%，v5.1749→5.4316%。Full反而为t14.0146/v4.6161%。
- Occlusion：流内位置8/Q16，t28.8770→35.7616%，v10.6912→13.1584%；Full为t31.9527/v11.7972%。
- Frame Drop位置2/Q13和位置15/Q03改变选择，但t/v仍0。

这两个收益均不是精确复现Full选择；Online与Full top1一致仍1/60，与native相同。最终判断应以task指标为准，不能把agreement作为成功替代指标。

## 对机制的判断

20/20次专家更新都降低pairwise训练loss。各流末||w||：drop .01136213、freeze .00189893、blur .00147897、occlusion .00748164、exposure .00139716。三个零收益流实际没有改变任何nonexpert选择。

因此保留两个向未来query迁移的具体正例，但不能据此宣称“Slow–Fast已经有效，只差condition-aware memory”。本轮也不能反过来宣称global linear state不可迁移：O1.1放大过的旧状态不等于O2四次原LR写入后的scale充分。O2仍只有4/60选择改变。

O1是32源/8写入/混合条件；本轮按请求16源/4写入/每流固定条件，方法参数相同，但两轮均值之差不能唯一归因于coherence。现在可以记为“弱局部正信号，尚未建立稳定收益”，不自动启动prototype、normalization、LR/alpha sweep或gate。

## 实际运行与核验

- 80 frozen capture，160 offset forwards；像素/预处理/框/两offset logits与旧基线精确一致，模型state不变。
- GPU进程92.961秒（含加载和CPU解码）；无新expert计算，复用20 sparse +60 Full caches；20 CPU SGD。
- 80到达/5零重置/20更新独立NumPy复算通过，w最大绝对误差6.51e−19；非专家teacher读取0。
- 先seal五条online轨迹，再读取60 Full reference，再seal四臂，最后读取原16键GT评分；无GT进入更新。
- 640次候选双实现metric检查，80次critic公式重建，486项公开均值/区间核验，原3项CPU合同测试均通过。
- 最初直接执行无可执行权限的CUDA wrapper在Python启动前退出126；改用bash调用现有脚本后完成，原launch_failure.log保留，无系统权限修改。
- 所有数据为历史曝光开发来源。原O1/O1.1、生产CURRENT未变，未启动后续实验。

完整四臂、候选级指标、所有正负/零案例、状态尺度诊断见 REPORT.md / ROWS.json / SUMMARY.json / READOUT_DIAGNOSTIC.json。原始媒体、标签、hidden和状态张量保留本地。

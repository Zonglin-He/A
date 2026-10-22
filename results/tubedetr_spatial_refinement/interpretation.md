# 对本轮空间结果的解释

这是完整两域128封存评分之后的解释补充，不改变冻结计划、选择、预测或原终态文件。

1. **收缩不是充分解释。**Vid保留Source宽高、仅加入alpha=.25中心修正，dense降至48.1367，低于旧参数配置48.5304；平均改动帧从5.30增到24.62。候选更接近Source后，原保护代价也变小，原评分选择了更多替代。该干预同时影响候选及其原公式分数，不能断言全部损失只由改动数量导致；但它反驳了“直接保尺寸就能修复”的简单判断。
2. **相对校正有小信号，幅度不足。**Vid residual alpha1/beta.5相对旧参数方案，dense +.0521pp、未观察+.0518pp、v+.0467pp；所有描述性区间跨0。相对Temporal，未观察+.0494pp、v+.0261pp，但整体dense−.0340pp，事前护栏失败。未观察与整体指标使用不同支持范围和分母，不能只报告正的未观察指标。
3. **时间支持影响空间收益。**同一Vid残差参数下，Spatial（Native时间）dense49.1277高于Source48.6164；Full（Global6时间）dense48.5825则仍低于Temporal。两种区间会改变可修改节点、评分归一化、路径及dense插值，不能把Native条件的空间收益冒充Global6完整方法收益。此次未修改时间分支。
4. **HC没有新的改进。**旧参数gamma.025/admission.7/guard.4/link1仍最好，dense57.6310、未观察57.5716、v27.7187。新的两种候选及标量复查均未超越它；旧方案自身相对Temporal只有很小点估计收益，区间跨0。
5. **研究边界。**本轮不引入第三种机制，不扩网格，不继续利用已曝光128做无上限筛选。全部29配置、3712条预测与receipt均完成核验，旧生产代码及全量不变。没有新神经网络/教师调用或全量。当前不能声称已获得稳定空间提升或跨域泛化优势。

可复查材料：plan.json、各域all_metrics/decisions/selected_profile、analysis.json、integrity.json、report.md和completion_seal.json。参数导出configs/tubedetr_spatial_refinement_128.json只记录开发状态，不改生产配置。

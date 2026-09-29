# 2026-09-29 O1.1 Residual Scale Readout — completed

只复用O1保存的24个非专家到达的arrival w、phi、native score，读出S=ell+alpha phi@w，固定alpha1/8/16/32。没有新训练、GPU、专家调用或在线trajectory，也没有读取新GT文件；最终task值来自O1先前双实现核验的逐候选指标。原32流/8专家写入/24评价源、Vid-source TA与UniversalVTG不变，属于已曝光开发数据的事后固定轨迹诊断。

| alpha | 相对Frozen改变选择 | 与Full Rerank完全一致 | tIoU % | vIoU % | Δt pp vs alpha1 | Δv pp vs alpha1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1/24 | 1/24 | 46.2867 | 17.9538 | 0 | 0 |
| 8 | 16/24 | 5/24 | 46.4284 | 17.5656 | +.1417 | −.3882 |
| 16 | 17/24 | 4/24 | 46.6493 | 17.6065 | +.3626 | −.3473 |
| 32 | 19/24 | 3/24 | 46.0061 | 17.3296 | −.2806 | −.6242 |

参照Full Rerank t50.0614/v18.6286%，它读取当前query专家，不是等预算方法或oracle。α8/16/32新增Full匹配4/3/3，α32还丢失原有1个匹配；多数改变并未恢复Full选择。放大解决了选择几乎不动的问题，但没有任一放大档的平均vIoU收益，不能把O1的零收益完全解释为residual没有话语权。

配对Δv条件95%区间：α8 [−1.1456,.1919]，α16 [−1.0844,.2189]，α32 [−1.5892,.1381]pp；所有放大档t/v区间均跨0，不把本次负均值写成普遍显著损害。区间仅重采样已封存24结果，不重模拟trajectory，不代表顺序泛化。v gain/loss/unchanged分别5/8/11、6/8/10、6/9/9；>5pp v损害1/1/2例。

保留局部成功：pos18/Q12/occlusion5，t87.302→92.593%、v30.964→32.592%，向Full移动并改善。也保留teacher一致但变差的pos15/Q15/drop5：t66.019→56.198%、v45.909→39.351%。pos16/Q07/freeze5在α32进一步恶化，v9.331→2.000%，并未选择Full。不能把Full agreement当GT correctness。

执行顺序已核对：96选择先seal，随后仅读已缓存Full评分并seal agreement，最后才消费已缓存GT-derived metrics。输入SHA前后不变，α1逐值复现O1；24 independent math.fsum dot最大误差5.55e−17，NumPy重建96选择/30均值与bootstrap区间及全部agreement/损害数通过。哈希验证会读取文件bytes，不消费其teacher/metric值进行候选选择。没有改源状态、模型或CURRENT。

结论限定为：scale解释决策不动，但单纯放大当前固定slow信号不足以改善tube任务；当前信号的跨query偏好质量仍未建立。不能由此判所有global linear/normalization/Slow-Fast都不可行，也不等价LR×alpha。没有自动选alpha或启动normalization rerun；本轮仅O1.1，prototype/memory/gate/LR grid/空间/C0.6均未扩展。

[完整报告](../results/tastvg_o11_scale_readout/2026-09-29/REPORT.md)，[全部24×4结果](../results/tastvg_o11_scale_readout/2026-09-29/ROWS.json)，[协议](../protocols/tastvg_o11_scale_readout_v1.md)。

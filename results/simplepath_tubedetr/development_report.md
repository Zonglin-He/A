# TubeDETR SimplePath/Native2D：固定双域128诊断

这次按main a238705a固定方法在原两域各128查询/128父来源执行，无新调参、无query替换。全部256预测/receipt哈希、request/plan绑定、有限数组、评分均值复算通过，完整封存先于独立CPU评分；总both128_gate在首个full receipt之前。完整全量仍在运行，以下不代替全量主表。

|域|方法|m_vIoU|dense sIoU|native/readout tIoU|
|---|---|---:|---:|---:|
|Vid128|Source|16.1008|48.6164|28.5176|
|Vid128|Spatial|15.5799|43.7212|28.5176|
|Vid128|NativeMBR|19.7760|48.6164|35.3238|
|Vid128|Temporal|18.5127|48.6164|33.4729|
|Vid128|Full|17.8746|43.8512|33.4729|
|HC128|Source|14.2886|57.5967|23.1933|
|HC128|Spatial|13.7090|52.0774|23.1933|
|HC128|NativeMBR|27.3462|57.5967|47.5084|
|HC128|Temporal|26.7979|57.5967|46.0379|
|HC128|Full|25.0904|51.4346|46.0379|

Source保留原生MAP，Spatial用新空间轨迹但原时间；NativeMBR仅对同一真实原生末层先验做物理MBR并保留Source框；Temporal施加一次固定原型反馈后MBR、保留Source框；Full同时使用新空间轨迹。NativeMBR是读出控制，原型反馈效应应相对该行判断。

Full相对Source的m_vIoU增量为Vid+1.7738pp、HC+10.8018pp；按父bootstrap描述性95%区间分别[-1.3873,4.7375]、[7.7729,13.8730]pp。dense sIoU分别-4.7652pp、-6.1621pp，未观察帧-4.7869pp、-6.2235pp。Full相对NativeMBR的时间变化分别-1.8509pp、-1.4705pp，m_vIoU分别-1.9014pp、-2.2558pp。固定空间路径和原型反馈在这些开发面板未优于相应控制，不能把相对原MAP Source的正增量全归新模块。

Full相对Source vIoU下降>5pp分别23/13条，时间或v任一下降>5pp的并集34/15条。五行全部正负逐query及父ID哈希匿名标量见各域all_signed_development_results.json；详细聚合、区间、损伤计数见development_metrics.json。没有删除负例。面板与先前开发/全量研究均已曝光，区间不是独立未曝光确认。

参数锁/原生模型/教师cache-only保持不变，新专家调用0；不因128结果进行性能筛选、扩大网格或换query。两域128完整后，已按固定顺序进入Vid10303全量，完成后HC1902；两域完整seal后各自独立评分，Root最终核验发布仍待。缓存加速运行计时不能充当冷部署成本。

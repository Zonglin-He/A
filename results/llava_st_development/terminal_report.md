# LLaVA-ST：两域128锁定后的完整全量终态

VidSTG10303 query/732父来源与HC1902 query/227父来源均完成预测封存及独立CPU评分；控制器正常退出，shared gpu.lock已释放。128选参锁和总锁先于任一全量，未用全量反馈调参。原预测和final seals保持原字节，不回写其中publication_pending字段。

| 全量指标(%) | Vid Source | Vid Ours | Vid差值pp | HC Source | HC Ours | HC差值pp |
|---|---:|---:|---:|---:|---:|---:|
| m_vIoU |12.433020|15.363700|+2.930680|9.704737|11.794343|+2.089605|
| dense sIoU |27.428778|29.882846|+2.454068|38.415365|42.084354|+3.668989|
| 未观察sIoU |27.411011|29.611171|+2.200160|38.339156|41.843245|+3.504090|
| 原生tIoU |42.677984|42.677984|0|21.730593|21.730593|0|
| 观察sIoU |28.468249|42.695506|+14.227258|41.690931|52.090489|+10.399558|

主表、dense及未观察的分母分别为两域完整10303/1902；观察指标仅在至少一个实际观察帧落入GT评价支持时可定义，分母为8593/1857，不能与完整总体混同。两域原生无效生成均为0。

配对父来源bootstrap95%区间(pp)：Vid v[2.4310,3.4442]、dense[1.6487,3.2660]、未观察[1.3987,3.0075]；HC v[1.7368,2.4369]、dense[2.9806,4.3488]、未观察[2.8198,4.1801]。均值增益不消除个体损伤：Vid有1397/10303条、HC有134/1902条vIoU相对Source下降超过5pp。所有改善、退化和并列均保留在逐域all_signed_full_results.json.gz中，而非仅发布有利样例。

两域128开发已曝光且包含在全量中，置信区间未校正开发选参；同一merged checkpoint共享于两域，训练来源包含VidSTG衍生指令，不能声称严格双源迁移、未见域或独立未曝光确认。当前方法是旧BoxFixed空间适配；Source/Ours原生时间完全相同，无Global6/MBR，时间提升为0。此终态不是新版SimplePath/Native2D结果。

Vid冻结profile67c1237741b6d10d：gamma1/admission.05/guard.025/link.5；HC3addfb6c2863f954：gamma.5/admission.05/guard.1/link.5。原full_metrics中parameters为null是原full_score调用未传局部params的序列化字段；实际推理使用总选择锁，Root已逐预测receipt核验selected_profile_id及plan/request绑定。原指标不改写；[终态核验](terminal_verification.json)补充明确参数与证据绑定。

Root在Vid终态首次核全部10303预测与receipt及有限数组，HC终态首次核全部1902；最终沿用Vid首核回执并核小seal/metrics哈希未变，避免重复扫描。两域评分详情重新聚合与公开均值、严重下降数一致；没有新GPU前向或GT重评分。完整总seal及域seal全部引用hash核对通过。发布完成须结合晚于本终态的outputs/maintenance/llava_st_final_publication.json回执；旧准备/资格/128发布不代表此处全量已发布。

缓存原资产继续保留。下一轮按用户授权先TubeDETR再LLaVA-ST，固定main a238705a新版方法，独立协议、原生时间与pre-text表示适配及真实冷/缓存资格必须先通过，不恢复旧网格或伪造后验。

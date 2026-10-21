# 内容边界CDF奖励资格

根据完整128范围诊断，区域Dice在教师较好时增加小框偏好、牺牲完整范围。本轮不改变观测、教师或网络，只检验四边内容边际CDF能否比区域重叠更接近STVG矩形定位目标。不是已确认的新方法或OPD优势。

固定两个信号：原生单ReferDINO选中软mask，以及已经封存的另一帧教师框支持传递。各自在有效图像中按x/y累积质量，候选左右上下四边的CDF分别对齐固定[.025,.975,.025,.975]，奖励为四项绝对误差均值的相反数。无L1/GIoU混合，不按GT改分位、不阈值化mask。由二维积分图连续插值边界质量、精确裁去padding，保留Direct连续框梯度。CDF在图像外饱和，无法额外惩罚越界扩框，明确保留此限制。零支持奖励为0、并列top1平均regret，不删除空帧/query。

只原Vid128/原八帧/原32候选，0新教师、0GPU、0学生fit、0参数搜索。跨帧支持由封存互近邻索引及原教师框还原，不重新匹配，不重新加载图像/教师；还原后旧TransportDice须逐query误差<=1e-6。CDF double积分，父终态与缓存/代码hash绑定。工程检查覆盖padding、分数像素积分、越界饱和、零支持、Direct有限差分和对应重建；61项CPU检查通过只代表实现验证。

全部128奖励封存后独立CPU读GT，报告Geometry/Layout3/TransportDice/两个CDF的排序、Spearman、top1regret和GT矩形覆盖/面积偏移/框内精度，教师较好/不更好/无观测GT组只诊断。Geometry/Layout3/TransportDice原统计须精确回归。资格规则事前不变：排序不劣几何、regret不高且至少一项改善。资格属已曝光开发、不作独立确认，不通过不做适应，通过也须另外先冻结同奖励OPD/Direct公平验证，不自动适应或全量。

完整v2结果128/128：MaskCDF/TransportCDF排序70.28%/71.58%，regret0.128684/0.140112，均未通过几何参照门槛。覆盖上升同时框内精度下降，未做适应或分位搜索。首版将bank总数误写34，真实为45（policy32+13诊断参考），尚未写reward或读GT即失败；v1保存outputs/tubedetr_cdf_failed_v1，v2计划绑定engineering_recovery.json并保持输入/评价规则。

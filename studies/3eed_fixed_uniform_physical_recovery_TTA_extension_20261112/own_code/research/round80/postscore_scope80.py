"""Descriptive scope/optimization readout from already audited saved scalars."""
from core80 import *
d=read(H80/'DECISION.json');g=d['cohort_gains'];pool=d['pooled2304_balanced_mIoU_percent'];effects=read(H80/'TASK_EFFECT_AND_TAIL_ROWS.json');updates=read(H80/'UPDATE_SUMMARY_ROWS.json');assert read(H80/'independent_scalar_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_')
d.update(retain_VAL1280_positive_result=True,VAL1280_positive_result_not_independent_confirmation=True,TRAIN1024_negative_retained=True,uniform_development_candidate_scope='fixed_exposed_DEV336_and_VAL1280;not_validated_on_TRAIN1024',all_auxiliary_steps_descent_guaranteed=False,do_not_attribute_all_negative_transfer_to_auxiliary_task_misalignment_without_step_diagnostics=True);dump(H80/'DECISION.json',d)
v=next(r for r in effects if r['cohort']=='VAL1280' and r['platform']=='balanced_tail');uv=next(r for r in updates if r['cohort']=='VAL1280' and r['arm']=='uniform_sgd1');ut=next(r for r in updates if r['cohort']=='TRAIN1024' and r['arm']=='uniform_sgd1');summary={'state':'PASS_R80_DESCRIPTIVE_POSTSCORE_SCOPE_AND_OPTIMIZATION_READOUT','VAL_gain_vs_frozen_pp':g['VAL1280']['vs_frozen_pp'],'VAL_gain_vs_direct3_pp':g['VAL1280']['vs_direct3_pp'],'VAL_correct_minus_wrong_pp':g['VAL1280']['vs_uniform_wrong_pp'],'TRAIN_gain_vs_frozen_pp':g['TRAIN1024']['vs_frozen_pp'],'TRAIN_gain_vs_direct3_pp':g['TRAIN1024']['vs_direct3_pp'],'TRAIN_correct_minus_wrong_pp':g['TRAIN1024']['vs_uniform_wrong_pp'],'pooled_gain_vs_frozen_pp':pool['uniform_sgd1']-pool['frozen_bi'],'pooled_gain_vs_direct3_pp':pool['uniform_sgd1']-pool['frozen_direct3'],'VAL_auxiliary_decrease_n':uv['auxiliary_decrease_n'],'VAL_auxiliary_n':uv['auxiliary_before_available_n'],'TRAIN_auxiliary_decrease_n':ut['auxiliary_decrease_n'],'TRAIN_auxiliary_n':ut['auxiliary_before_available_n'],'VAL_gain_excluding_largest_positive_pp':v['remaining_gain_after_largest_positive_pp'],'not_new_hyperparameter_selection_or_experiment':True};dump(H80/'SCOPE_AND_OPTIMIZATION_READOUT.json',summary)
text=f'''

## 范围与优化读数补充：保留VAL正证据，区分未解原因

VAL1280的正结果不能被TRAIN的负结果抹去：相对冻结{g['VAL1280']['vs_frozen_pp']:+.6f}pp，相对同信息直接平均{g['VAL1280']['vs_direct3_pp']:+.6f}pp，相对均匀错配{g['VAL1280']['vs_uniform_wrong_pp']:+.6f}pp。Drone/Quad均有正mIoU增量，共49条改善、40条恶化；去掉最大单例正贡献，剩余仍{v['remaining_gain_after_largest_positive_pp']:+.6f}pp（原分母），不是只依赖一条大正例。它支持“固定已暴露VAL范围存在参数更新增量”，不等于新的盲测确认。

TRAIN1024则为相对冻结{g['TRAIN1024']['vs_frozen_pp']:+.6f}pp，Drone+0.083802pp/Quad−0.430266pp；共有40条改善、39条恶化，不能用改善条数多数取代连续IoU总量。它限制的是跨recording条件下的重复性。将2304混合会得到相对冻结{summary['pooled_gain_vs_frozen_pp']:+.6f}pp、相对直接平均仅{summary['pooled_gain_vs_direct3_pp']:+.6f}pp，故分别报告两组而不以合并微小正数维持统一主张。

此外，本固定步长并不保证辅助损失下降：VAL的{uv['auxiliary_decrease_n']}/{uv['auxiliary_before_available_n']}步下降，TRAIN的{ut['auxiliary_decrease_n']}/{ut['auxiliary_before_available_n']}步下降，全部样本仍有非零参数变化。这是已保存端点的描述性优化读数，不能据此事后按GT或辅助下降筛出“有效样本”重算主成绩。当前一步保持项初始梯度为0，且未启用回溯，不能把所有负迁移都归为“辅助目标与grounding不一致”，步长／非线性优化同样是尚未隔离的解释。没有在本轮补做步长选择或新损失。

冻结双向模型本身在TRAIN1024已低于原始3EED（16.521759% vs17.993519%），而在VAL1280高于原始模型（18.653630% vs18.148518%）。因此静态source准备的泛化和TTA增量是两个问题，不能将全部差距归因于测试时更新。这里只给出测量差异，不把它直接解释成RGB错误或某类语义失效。

保留DEV/VAL正信号，收缩“该固定方法稳定跨场景有效”的主张；预设的进一步机制消融资源条件没有满足，当前不启动。条件仅用于决定本轮投入，不是否定VAL正结果的数学标准。正确绑定在VAL有效、在TRAIN无优势，目前尚不足以把跨模态恢复的因果贡献推广到两个范围。所有新方法预测、固定对照和负例均不改变。
'''
with (H80/'REPORT.md').open('a') as f:f.write(text)
numeric=read(H80/'REPORT_NUMERIC_AUDIT.json');numeric['scope_and_optimization_readout']=summary;dump(H80/'REPORT_NUMERIC_AUDIT.json',numeric);print(summary)

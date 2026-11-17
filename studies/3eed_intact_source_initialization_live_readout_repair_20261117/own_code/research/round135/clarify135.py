"""Post-score interpretation only; preserves registered decisions and every output."""
from common135 import *
import numpy as np
def main():
 Q=H135/'scores_attempt001';v={r['arm']:r for r in rd(Q/'AGGREGATE_ROWS.json') if r['aggregation']=='design_weighted' and r['group']=='all_six_equal'};paired=rd(Q/'ALL_PAIRED_ROWS.json');process=rd(H135/'PROCESS_ROWS_PRIVATE.json');summary=rd(H135/'SUMMARY.json');d=rd(H135/'DECISION.json');decision_hash=sha(H135/'DECISION.json');metrics_hash=sha(Q/'AGGREGATE_ROWS.json')
 def val(k,metric='mIoU_percent'):return v[k][metric]
 def gain(a,b):return val(a)-val(b)
 actual={k:{'fitted':len([r for r in process if r['kind']==k]),'nonzero':sum(r['parameter_L2']>0 for r in process if r['kind']==k)} for k in ['tta_A','tta_F','wrong_A','wrong_F']}
 out={'registered_A_protocol_gate_passed':d['TTA_development_candidate_retained'],'registered_primary_unchanged':'tta_A','not_global_best_same_information_method':True,'A_TTA_minus_complete_F_zero_pp':gain('tta_A','intact_live_F'),'A_TTA_minus_F_direct_pp':gain('tta_A','direct_F'),'F_TTA_minus_own_zero_pp':gain('tta_F','intact_live_F'),'F_TTA_largest_positive_contribution_exceeds_net_gain':next(x['largest_positive_contribution_pp'] for x in summary['comparisons'] if x['method']=='tta_F' and x['control']=='intact_live_F')>gain('tta_F','intact_live_F'),'A_TTA_minus_legacy_R134_Acc50_pp':val('tta_A','Acc50_percent')-val('legacy_R134','Acc50_percent'),'old_zero_to_intact_A_gain_pp':gain('intact_live_A','legacy_zero_F'),'old_TTA_gain_pp':gain('legacy_R134','legacy_zero_F'),'new_A_TTA_gain_pp':gain('tta_A','intact_live_A'),'new_A_vs_old_R134_gain_pp':gain('tta_A','legacy_R134'),'A_zero_fixed_vs_live_pp':gain('intact_fixed_A','intact_live_A'),'A_reset_only_pp':gain('reset_live_A','intact_live_A'),'fixed_context_after_reset_compensation_pp':gain('reset_fixed_A','reset_live_A'),'complete_F_live_minus_fixed_pp':gain('intact_live_F','intact_fixed_F'),'actual_updates':actual,'interpretation':'engineering restoration passes; A-protocol adaptation improves over its own zero/direct/ABmean/wrong but remains below complete-F no-update; F adaptation has a small fragile incremental gain. No global/robust TTA superiority claim, target-based output switching or new run.'}
 summary['scientific_disposition_postscore']=out;dump(H135/'SUMMARY.json',summary)
 text='''
## Post-score interpretation: stronger same-information baseline and metric tradeoff

The registered A-protocol gate and DECISION.json are preserved exactly. Passing
that gate supports A-output development only; it does **not** establish global
superiority over every same-information readout. Complete no-update F_AB is
13.580861%, above A-output TTA13.498989% by0.081872pp. Complete F direct readout
13.544418% also exceeds A-output TTA. These stronger controls were predeclared,
run and retained, and must not be hidden by the narrower A-protocol gate.

F-output TTA13.606614% is0.025752pp above its own complete zero and0.062195pp
above its direct readout. Its25improvements/26harms and largest positive case
contribution0.053304pp (larger than net gain) make this a fragile secondary
signal. It was not promoted over the registered A primary by target-score argmax.

Only522of749 qualified A fits actually changed parameters;85changed Top1 and
46changed IoU (29better/17worse). F has749nonzero updates,98Top1 changes and
51IoU changes (25better/26worse). Active-task count is not actual-update count,
and actual-update count is not effective correction count.

The factorial result identifies reset as the dominant A-only starting loss:
reset alone with live head costs0.783208pp. Fixed source-A context alone costs
0 at intact zero-A, while after reset it offsets0.099555pp of damage. Thus
initialization and readout losses are not additive. Under intact source state,
full live F gains0.186398pp over A; stale fixed context loses0.065458pp versus
live F. The old reset+fixed path instead loses0.080292pp when adding F.

Old zero12.630518→intact A13.394463 restores0.763945pp. Old R134 already added
0.243470pp to that weakened zero; new A TTA adds0.104526pp to the intact zero.
Consequently final A improves0.625001pp over R134, and **that entire0.625001pp
must not be attributed to stronger TTA**. Against old R134, A's Acc50 decreases
from5.170967% to4.644834% (-0.526133pp), despite mIoU/Acc25 gains. All signed
metric/direction/scene comparisons remain in the public tables.

Disposition: retain the intact initialization/live-reader/no-B engineering
repair. Keep A TTA only as the predeclared A-protocol development candidate;
retain complete frozen-F and the secondary F-update evidence as separate
controls. This exposed-cohort experiment does not establish a stable globally
best TTA method. No prediction, threshold, step, readout or registered gate was
changed after scoring.
'''
 p=H135/'REPORT.md';p.write_text(p.read_text()+text);assert sha(H135/'DECISION.json')==decision_hash and sha(Q/'AGGREGATE_ROWS.json')==metrics_hash;print(out,flush=True)
if __name__=='__main__':main()

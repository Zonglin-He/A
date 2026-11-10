"""Postseal rank diagnostic avoids treating softened probability gaps as rescue."""
from common61 import *
H=ROOT/'research/round61';P=ROOT/'crossplatform/round61/intervened336_attempt001';tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');labels=read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json');assert read(H/'independent_diagnostic_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');rows=[]
def rank(scores,valid):
 order=np.argsort(-scores,kind='stable');positions=np.flatnonzero(valid[order]);return int(positions[0]+1) if len(positions) else 257
for i,(t,l) in enumerate(zip(tasks,labels)):
 z=load(P/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'));valid=np.array(l['candidate_IoUs_PRIVATE'])>.25;base=rank(z['scores'][ARMS.index('frozen_ABC')],valid);recoverable=base>1 and base<=256
 for j,arm in enumerate(ARMS):rows.append({'row':f'DEV336_{i+1:05d}','platform':t['platform'],'arm':arm,'recoverable_ABC_error25':bool(recoverable),'best_good_box_rank_before':base,'best_good_box_rank_after':rank(z['scores'][j],valid)})
summary=[]
for arm in ARMS:
 rr=[r for r in rows if r['arm']==arm and r['recoverable_ABC_error25']];summary.append({'arm':arm,'recoverable_n':len(rr),'best_good_rank_improved':sum(r['best_good_box_rank_after']<r['best_good_box_rank_before'] for r in rr),'best_good_rank_worsened':sum(r['best_good_box_rank_after']>r['best_good_box_rank_before'] for r in rr),'best_good_rank_unchanged':sum(r['best_good_box_rank_after']==r['best_good_box_rank_before'] for r in rr),'median_best_good_rank_before':float(np.median([r['best_good_box_rank_before'] for r in rr])),'median_best_good_rank_after':float(np.median([r['best_good_box_rank_after'] for r in rr]))})
dump(H/'RANK_SUMMARY.json',{'state':'R61_POSTSEAL_DESCRIPTIVE_RANK_DIAGNOSTIC_NO_METHOD_CHANGE','GT_used_only_for_diagnosis':True,'not_probability_margin_proxy':True,'metrics':summary});dump(H/'ALL_RANK_SCALARS_PUBLIC_SAFE.json',rows);print(json.dumps(summary),flush=True)

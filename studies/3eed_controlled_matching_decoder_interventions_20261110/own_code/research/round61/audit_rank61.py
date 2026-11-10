"""Independent ordinary-sort rank diagnostic check from sealed scores."""
from common61 import *
H=ROOT/'research/round61';P=ROOT/'crossplatform/round61/intervened336_attempt001';tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');labels=read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json');rows=read(H/'ALL_RANK_SCALARS_PUBLIC_SAFE.json');summary=read(H/'RANK_SUMMARY.json')
for i,(t,l) in enumerate(zip(tasks,labels)):
 z=load(P/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'));valid={j for j,v in enumerate(l['candidate_IoUs_PRIVATE']) if v>.25};expected={}
 for j,arm in enumerate(ARMS):
  ordered=sorted(range(256),key=lambda k:(-float(z['scores'][j,k]),k));expected[arm]=next((r+1 for r,k in enumerate(ordered) if k in valid),257)
 for j,arm in enumerate(ARMS):
  r=rows[i*len(ARMS)+j];assert r['row']==f'DEV336_{i+1:05d}' and r['arm']==arm and r['best_good_box_rank_after']==expected[arm] and r['best_good_box_rank_before']==expected['frozen_ABC'];assert r['recoverable_ABC_error25']==(1<expected['frozen_ABC']<=256)
for m in summary['metrics']:
 rr=[r for r in rows if r['arm']==m['arm'] and r['recoverable_ABC_error25']];assert len(rr)==m['recoverable_n']==126
 for key,cmp in [('improved',lambda a,b:a<b),('worsened',lambda a,b:a>b),('unchanged',lambda a,b:a==b)]:assert m['best_good_rank_'+key]==sum(cmp(r['best_good_box_rank_after'],r['best_good_box_rank_before']) for r in rr)
out=H/'independent_rank_audit_attempt001';assert not out.exists();out.mkdir();r={'state':'PASS_R61_INDEPENDENT_EXACT_RANK_RECONSTRUCTION','n':336,'rank_rows':len(rows),'recoverable_n':126,'new_predictions_or_task_scoring':False,'summary_sha256':sha(H/'RANK_SUMMARY.json'),'program_sha256':sha(__file__)};dump(out/'AUDIT_RESULT.json',r);print(r)

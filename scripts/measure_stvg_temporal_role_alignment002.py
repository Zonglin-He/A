"""Postseal official/independent scores and paired four-candidate rank diagnostics."""
from pathlib import Path
import sys,math,statistics,itertools,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_role_alignment002 import *
def main(phase):
 import torch,numpy as np
 from scripts import review_stvg_sparse_r1_v2 as truth_reader
 from scripts import score_stvg_native_instance_nativeclock005 as scorer
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 check_runtime();b=BASE/phase;root=read(b/'ROOT_READBACK.json');seal=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert root['status']=='pass'and root['count']==64 and not root['GT_read']and seal['status']=='complete'and seal['count']==64 and seal['GT_read']is False and seal['scorer_sha256']==sha(__file__)and seal['runtime_sha256']==sha(BASE/'RUNTIME.json');assert not(b/'GT_EXPOSURE.json').exists()
 for x in seal['files']:assert sha(ROOT/x['path'])==x['sha256']
 records=vid_roster();assert len(records)==64;cases={r['probe_alias']:load_case(r)[0]for r in records}
 write(b/'GT_EXPOSURE.json',dict(status='all64_predictions_and_root_sealed_before_official_labels',GT_read=True,prediction_seal_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
 truth,_=truth_reader._truths_after_seal(cases);official,independent,_=truth_reader._metric_functions();rows=[];ranks=[]
 refs={r['case_id']:{a['position']for a in r['anchors']}for r in read(COORD/'REFERENCE_LOCK.json')['records']}
 for rec in records:
  cid=rec['probe_alias'];c=cases[cid];d=torch.load(b/'student'/(cid+'.pt'),map_location='cpu',weights_only=False)
  for arm in ['F','T',phase]:
   metrics,audit=scorer._score_student_arm(d['arms'][arm],c,truth[cid],refs[cid],official,independent);rows.append(dict(case_id=cid,query_type=rec['query_type'],arm=arm,metrics=metrics,metric_audit=audit['metric_audit']))
  if is_question(rec):
   old=p0_responses(rec);new=torch.load(b/'teacher'/(cid+'.pt'),map_location='cpu',weights_only=False)['slots'];rw0=[ta._alignment_reward(x['canonical_id'],r)if r is not None else None for x,r in zip(c['temporal_candidates'],old if old else[None]*len(c['temporal_candidates']))];rw1=[ta._alignment_reward(x['canonical_id'],r)if r is not None else None for x,r in zip(c['temporal_candidates'],new)];qualities=[]
   for candidate in c['temporal_candidates']:
    m,_=scorer._score_student_arm(dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']),interval=candidate['interval_frames']),c,truth[cid],refs[cid],official,independent);qualities.append(m['m_tIoU'])
   pairs=[];unknown=gt_ties=0
   for i,j in itertools.combinations(range(len(c['temporal_candidates'])),2):
    if any(x is None for x in [qualities[i],qualities[j],rw0[i],rw0[j],rw1[i],rw1[j]]):unknown+=1;continue
    q=qualities[i]-qualities[j]
    if abs(q)<=1e-10:gt_ties+=1;continue
    scores=[]
    for rw in [rw0,rw1]:
     delta=rw[i]-rw[j];scores.append(.5 if abs(delta)<=1e-10 else float(delta*q>0))
    pairs.append(scores)
   ranks.append(dict(case_id=cid,actual_candidate_count=len(c['temporal_candidates']),missing_candidate_budget=4-len(c['temporal_candidates']),known_pairs=len(pairs),unknown_pairs=unknown,GT_tied_pairs=gt_ties,P0_accuracy=statistics.mean(x[0]for x in pairs)if pairs else None,new_accuracy=statistics.mean(x[1]for x in pairs)if pairs else None))
 by={(x['case_id'],x['arm']):x for x in rows};summary=[];paired=[]
 for cohort in ['all64','question28','caption36']:
  ids=[r['probe_alias']for r in records if cohort=='all64'or (r['query_type']=='question')==(cohort=='question28')]
  for metric in ['m_vIoU','m_tIoU','m_sIoU_dense_GT']:
   for arm in ['F','T',phase]:
    vals=[by[cid,arm]['metrics'][metric]for cid in ids if by[cid,arm]['metrics'][metric]is not None];summary.append(dict(cohort=cohort,arm=arm,metric=metric,count=len(vals),unknown_count=len(ids)-len(vals),mean=statistics.mean(vals)if vals else None))
   delta={cid:100*(by[cid,phase]['metrics'][metric]-by[cid,'T']['metrics'][metric])for cid in ids if by[cid,phase]['metrics'][metric]is not None and by[cid,'T']['metrics'][metric]is not None};paired.append(dict(cohort=cohort,metric=metric,count=len(delta),unknown_count=len(ids)-len(delta),mean_delta_pp=statistics.mean(delta.values())if delta else None,per_source_delta_pp=delta,positive=sum(x>1e-10 for x in delta.values()),negative=sum(x< -1e-10 for x in delta.values())))
 question={x['metric']:x['mean_delta_pp']for x in paired if x['cohort']=='question28'};complete_question_support=all(x['count']==28 and x['unknown_count']==0 for x in paired if x['cohort']=='question28'and x['metric']in ['m_vIoU','m_tIoU']);eligible=complete_question_support and question['m_vIoU']is not None and question['m_tIoU']is not None and question['m_vIoU']>0 and question['m_tIoU']>=0
 common=dict(status='complete',count=64,question_count=28,GT_read=True,bootstrap_executed=False,prediction_seal_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'))
 write(b/'POSTSEAL_SCORE_ROWS.json',dict(common,rows=rows));write(b/'POSTSEAL_SCORE_SUMMARY.json',dict(common,summary=summary,paired=paired));write(b/'POSTSEAL_RANKING_DIAGNOSTICS.json',dict(common,records=ranks,definition='query-macro common valid candidate-pair order agreement with actual tIoU; reward ties half credit; GT ties/unknown excluded and counted',P0_macro_accuracy=statistics.mean(x['P0_accuracy']for x in ranks if x['P0_accuracy']is not None)if any(x['P0_accuracy']is not None for x in ranks)else None,new_macro_accuracy=statistics.mean(x['new_accuracy']for x in ranks if x['new_accuracy']is not None)if any(x['new_accuracy']is not None for x in ranks)else None))
 write(b/'CPU_COMPLETION.json',dict(common,p2_eligible=bool(eligible),scope='development comparison and fixed P1 gate only; not independent confirmation or overall FINAL',question_delta_pp=question,CPU_only=True,new_GT_oracle_fits=0));print(phase,'complete64',question,'P2 eligible',eligible)
if __name__=='__main__':main(sys.argv[1])

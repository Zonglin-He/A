"""Score only after all256 T2 predictions and exact replay are sealed."""
from pathlib import Path
import copy,gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_t2_v1 import BASE,CORE,read,sha,put,verify
def subset_aggregate(rows):
 """Pre-adaptation eligibility may have unequal domain sizes; keep both Ns."""
 import math,numpy as np
 from scipy.stats import t
 summary=[];paired=[];arms=list(dict.fromkeys(r['arm'] for r in rows))
 for ds in ('hc2','vidstg'):
  rr=[r for r in rows if r['dataset']==ds];ids=sorted({r['case_id'] for r in rr});idx={(r['case_id'],r['arm']):r['metrics'] for r in rr}
  if not ids:continue
  assert len(idx)==len(rr)==len(ids)*len(arms)
  for a in arms:
   for key in idx[ids[0],a]:
    values=[idx[i,a][key] for i in ids if idx[i,a][key] is not None];summary.append(dict(dataset=ds,arm=a,metric=key,valid=len(values),unknown=len(ids)-len(values),mean=math.fsum(values)/len(values) if values else None))
  for j,a in enumerate(arms):
   for b in arms[:j]:
    for key in idx[ids[0],a]:
     values={i:100*(idx[i,a][key]-idx[i,b][key]) for i in ids if idx[i,a][key] is not None and idx[i,b][key] is not None};n=len(values);mean=math.fsum(values.values())/n if n else None;err=float(t.ppf(.975,n-1)*np.std(list(values.values()),ddof=1)/math.sqrt(n)) if n>1 else None
     paired.append(dict(dataset=ds,comparison=a+'_minus_'+b,metric=key,valid=n,unknown=len(ids)-n,mean_delta_pp=mean,paired_source_t95_interval_pp=[mean-err,mean+err] if err is not None else None,positive=sum(x>1e-6 for x in values.values()),negative=sum(x< -1e-6 for x in values.values()),neutral=sum(abs(x)<=1e-6 for x in values.values()),per_source_delta_pp=values))
 return summary,paired
def main():
 import torch,numpy as np
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts.stvg_overnight_score_v1 import aggregate
 from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
 verify();root=read(BASE/'ROOT_READBACK.json');bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert root['status']=='pass' and root['count']==bar['count']==256 and bar['status']=='complete'
 files={f['case_id']:f for f in bar['files']}
 for f in files.values():assert sha(ROOT/f['path'])==f['sha256']
 controls=validate_barrier('A');rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];usage=[];invariants=[];eligible={r['case_id']:r for r in read(BASE/'CANDIDATE_LOCK.json')['records']}
 for r in rr:
  cid=r['probe_alias'];case=cases[cid];old=torch.load(ROOT/controls[cid]['path'],map_location='cpu',weights_only=False);new=torch.load(ROOT/files[cid]['path'],map_location='cpu',weights_only=False);assert new['source_case_sha256']==case['source_case_sha256'];pred=copy.deepcopy(old['arms']['F']);pred['interval']=list(new['selected_interval']);common=[w['source_grid_position'] for w in read(CORE/'spatial/E1/proposals'/(cid+'.json'))['windows']]
  for name,p in [('Frozen',old['arms']['F']),('T4',old['arms']['T']),('T2',pred)]:
   assert np.array_equal(p['boxes'],old['arms']['F']['boxes']);v,_=metric._score_student_arm(p,case,targets[cid],common,official,independent);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=r['dataset'],arm=name,metrics=v))
  f=new['fit'];usage.append(dict(case_id=cid,dataset=r['dataset'],T2_eligible=eligible[cid]['eligible'],T4_eligible=eligible[cid]['T4_eligible'],actual_fit=f is not None,optimizer_steps=f['optimizer_steps'] if f else 0,no_fit_reason=new['no_fit_reason'],termination=f['termination'] if f else None,GT_read=False));invariants.append(dict(case_id=cid,Frozen_spatial_outputs_exact_for_both_T_only_arms=True,GT_read=False));del old,new;gc.collect()
 summary,paired=subset_aggregate(rows);check_summary,check_paired=aggregate(rows,256);assert summary==check_summary
 for actual,check in zip(paired,check_paired):assert {k:v for k,v in actual.items() if k!='paired_source_t95_interval_pp'}==check
 common_ids={r['case_id'] for r in usage if r['T2_eligible'] and r['T4_eligible']};common_rows=[r for r in rows if r['case_id'] in common_ids];cs,cp=subset_aggregate(common_rows)
 put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=256,per_domain=128,rows=rows,summary=summary,paired=paired,common_eligible=dict(count=len(common_ids),summary=cs,paired=cp),usage=usage,invariants=invariants,GT_bindings=bindings,GT_read=True,bootstrap=False,cohort='exposed development; not independent confirmation',CURRENT_changed=False))
 put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('T2_T4_256_POSTSEAL_COMPLETE',flush=True)
if __name__=='__main__':main()

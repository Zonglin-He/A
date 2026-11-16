"""Postseal full-cohort eligibility, ranking headroom and cohort composition.
Reuses existing scalar outcomes; no GT/media/model/feature/prediction recomputation.
"""
from pathlib import Path
import json,hashlib,numpy as np,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round119';O=H/'analysis_attempt001';P=R/'research/round100/scores_attempt001'
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
ARMS=['original_source','prepared_frozen','binding_direct','direct_FC','direct_four','lm_FC','joint3_FC','joint3_shuffle_FC'];COHORTS=['FULL','DEV','EXP','REMAINDER'];THRESHOLDS=[.25,.5]
def metric(rr,k):
 y=np.array([x['values'][k] for x in rr]);o=np.array([x['oracle_IoU'] for x in rr]);out={'n':len(rr),'mIoU_percent':float(y.mean()*100),'oracle_mIoU_percent':float(o.mean()*100),'mean_oracle_headroom_pp':float((o-y).mean()*100),'oracle_p10':float(np.quantile(o,.1)),'oracle_median':float(np.median(o)),'oracle_p90':float(np.quantile(o,.9))}
 for tau in THRESHOLDS:
  tag=str(int(tau*100));covered=o>tau;hit=y>tau;assert not (hit&~covered).any();failure=~hit;cov=int(covered.sum());rank=int((failure&covered).sum());no=int((~covered).sum());out.update({f'Acc{tag}_percent':float(hit.mean()*100),f'oracle_Acc{tag}_percent':float(covered.mean()*100),f'no_good_candidate_at{tag}':no,f'covered_but_not_selected_at{tag}':rank,f'selection_recall_given_covered_at{tag}':float(hit.sum()/cov) if cov else None,f'fraction_of_failures_recoverable_at{tag}':float(rank/failure.sum()) if failure.any() else None});assert no+rank==int(failure.sum())
 return out
def paired(rr,k,c):
 dd=np.array([x['values'][k]-x['values'][c] for x in rr]);out={'n':len(rr),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_pp_within_direction':float(max(0,dd.max())*100/len(rr)),'largest_negative_pp_within_direction':float(min(0,dd.min())*100/len(rr))}
 for t in THRESHOLDS:
  tag=str(int(t*100));a=np.array([x['values'][k]>t for x in rr]);b=np.array([x['values'][c]>t for x in rr]);s=np.array([x['values']['original_source']>t for x in rr]);out.update({f'rescue_at{tag}':int((a&~b).sum()),f'damage_at{tag}':int((b&~a).sum()),f'rescue_of_original_source_failure_at{tag}':int((a&~b&~s).sum()),f'restore_original_source_hit_at{tag}':int((a&~b&s).sum())})
 return out
def main():
 plan=rd(H/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R/p)==h
 for p,h in plan['inputs'].items():assert sha(R/p)==h
 assert rd(R/'research/round100/SCALAR_AUDIT.json')['n']==18620;O.mkdir(exist_ok=False);rows=[];met=[];pairs=[];scenes=[];composition=[];cases=[];statuses={};repro=0;comparisons=[('prepared_frozen','original_source'),('lm_FC','prepared_frozen'),('lm_FC','direct_FC'),('joint3_FC','prepared_frozen'),('joint3_FC','direct_four'),('joint3_FC','joint3_shuffle_FC')]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;old=rd(P/(name+'_ALL_ROWS_PRIVATE.json'));dev={r['id'] for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')};exp={r['id'] for r in rd(R/f'research/round109/input/TARGET_{tgt.upper()}256_PRIVATE.json')};assert not dev&exp;known={r['id_PRIVATE'] for r in old};assert dev|exp<=known;rr=[]
   cross={}
   for co in ['DEV','EXP']:
    for r in rd(R/f'research/round118/scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json'):cross[r['id_PRIVATE']]=r['arms']['original_source']['IoU']
   for r in old:
    sid=r['id_PRIVATE'];vs={k:r['arms'][k]['IoU'] for k in ARMS};oracles=[a['oracle_IoU_diagnostic'] for a in r['arms'].values()];assert all(abs(x-oracles[0])<1e-12 for x in oracles);assert all(0<=x<=oracles[0]+1e-12 for x in vs.values());status=r['status'];signature=','.join(sorted(status.keys()));statuses[signature]=statuses.get(signature,0)+1
    if sid in cross:assert abs(vs['original_source']-cross[sid])<1e-12,(name,sid,vs['original_source'],cross[sid]);repro+=1
    rr.append({'source':src,'target':tgt,'id_PRIVATE':sid,'scene_PRIVATE':r['scene_PRIVATE'],'cohort':'DEV' if sid in dev else 'EXP' if sid in exp else 'REMAINDER','oracle_IoU':oracles[0],'values':vs,'unknown_roles':status.get('unknown_roles',None),'visible_seeds':status.get('visible_seeds',None),'status_schema':signature})
   dump(O/(name+'_ALL_ROWS_PRIVATE.json'),rr);rows+=rr
   for co in COHORTS:
    cr=rr if co=='FULL' else [r for r in rr if r['cohort']==co];assert len(cr)>0
    for k in ARMS:met.append({'cohort':co,'source':src,'target':tgt,'arm':k,**metric(cr,k)})
    for k,c in comparisons:pairs.append({'cohort':co,'source':src,'target':tgt,'method':k,'control':c,**paired(cr,k,c)})
    for sc in sorted({x['scene_PRIVATE'] for x in cr}):
     sr=[r for r in cr if r['scene_PRIVATE']==sc]
     for k in ARMS:scenes.append({'cohort':co,'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,**metric(sr,k)})
   for co in ['DEV','EXP']:
    cr=[r for r in rr if r['cohort']==co];scd=sorted({x['scene_PRIVATE'] for x in cr})
    for k in ARMS:
     muF=metric(rr,k)['mIoU_percent'];muD=metric(cr,k)['mIoU_percent'];standard=sum(sum(x['scene_PRIVATE']==sc for x in cr)/len(cr)*metric([x for x in rr if x['scene_PRIVATE']==sc],k)['mIoU_percent'] for sc in scd);mix=standard-muF;within=muD-standard;assert abs(mix+within-(muD-muF))<1e-10;composition.append({'cohort':co,'source':src,'target':tgt,'arm':k,'full_mean':muF,'cohort_mean':muD,'difference_pp':muD-muF,'full_scene_means_reweighted_by_cohort_mix':standard,'scene_mixture_component_pp':mix,'within_scene_component_pp':within,'full_scene_count':len({r['scene_PRIVATE'] for r in rr}),'cohort_scene_count':len(scd),'causal_attribution':False})
   for k,c in comparisons:
    for kind,sgn in [('good',1),('failure',-1)]:
     tail=[r for r in rr if sgn*(r['values'][k]-r['values'][c])>1e-12]
     if tail:
      r=max(tail,key=lambda r:abs(r['values'][k]-r['values'][c]));cases.append({'source':src,'target':tgt,'id_PRIVATE':r['id_PRIVATE'],'method':k,'control':c,'kind':kind,'delta_IoU':r['values'][k]-r['values'][c],'six_direction_macro_contribution_pp':(r['values'][k]-r['values'][c])*100/(6*len(rr)),'cohort':r['cohort'],'identity':'UNKNOWN_no_media_reinspection'})
 assert len(rows)==18620 and repro==2304;agg=[]
 numeric=[k for k,v in met[0].items() if isinstance(v,(int,float)) and k!='n']
 for co in COHORTS:
  for g in ['all_six_equal','waymo','drone','quad']:
   for k in ARMS:
    mm=[m for m in met if m['cohort']==co and m['arm']==k and (g=='all_six_equal' or m['target']==g)];agg.append({'cohort':co,'group':g,'arm':k,**{n:float(np.mean([x[n] for x in mm])) for n in numeric}})
 for n,x in [('METHOD_ROWS',met),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('COMPOSITION_ROWS',composition),('AGGREGATE_ROWS',agg),('GOOD_FAILURE_CASES_PRIVATE',cases)]:dump(O/(n+'.json'),x)
 dump(H/'DIAGNOSTIC_RECEIPT.json',{'state':'COMPLETE_POSTSEAL_EXISTING_SCALAR_ONLY_ELIGIBILITY_AND_COHORT_DECOMPOSITION','n':18620,'same_source_across_R118_inputs_exact':repro,'method_arms':ARMS,'cohort_partition':{c:sum(r['cohort']==c for r in rows) for c in ['DEV','EXP','REMAINDER']},'status_schema_counts':statuses,'new_GT_scoring':0,'new_model_or_updates':0,'new_media_or_features':0,'FINAL_access':False,'method_selection_from_diagnostic':False});print('DIAGNOSE119_COMPLETE',len(rows),repro,flush=True)
if __name__=='__main__':main()

"""Independent scalar DINO geometry/metrics; actual-node teacher-support audit."""
import os,sys,time,collections,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_residual_advantage64_v1 import D,P,ROOT,ARMS,STRATA,verify,read,put,load,sha

def scalar(b,t,row,ds):
 cx,cy,w,h=map(float,b);W,H=row['input']['width'],row['input']['height'];a=[(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]
 if ds=='hc2':a=[max(v,0.)for v in a]
 ix=max(0.,min(a[2],t[2])-max(a[0],t[0]));iy=max(0.,min(a[3],t[3])-max(a[1],t[1]));it=ix*iy;u=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,t[2]-t[0])*max(0.,t[3]-t[1])-it;return it/u if u>0 else 0.

def independent_geo(E,tr,row,ds):
 W,H=row['input']['width'],row['input']['height'];cx,cy,w,h=map(float,E);tw=tr[2]-tr[0];th=tr[3]-tr[1];c=max(abs(cx*W-(tr[0]+tr[2])/2)/tw,abs(cy*H-(tr[1]+tr[3])/2)/th);s=max(w*W/tw,tw/(w*W),h*H/th,th/(h*H));iou=scalar(E,tr,row,ds);bad=iou<=.3;return iou,c,s,dict(all=True,bad=bad,far=bad and c>=1,scale=bad and s>=2,scale_near_center=bad and s>=2 and c<=.25)

def main():
 import numpy as np,torch,ijson
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 torch.set_num_threads(4);p=verify(True);bar=read(D/'CACHE_BARRIER.json');assert bar['count']==64;z=read(P/'RESULTS_PUBLIC.json');checks=collections.Counter();err=collections.Counter();start=time.perf_counter()
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};raw=load_vidor_annotations(ROOT/'downloads/vidor/validation-annotation.zip',None);keys={plans['vidstg'][r['query_ordinal']]['key']for r in p['records']if r['dataset']=='vidstg'}
 with (ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json').open('rb')as f:tids={k:int(v['official_annotation']['target_id'])for k,v in ijson.kvitems(f,'',use_float=True)if k in keys}
 actual=[];cache={};frame_index={(f['alias'],f['position']):f for f in z['frames']};state_index={(s['alias'],s['position'],s['sample_index']):s for s in z['conditional_states']}
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];ds=r['dataset'];x=load(D/'extracted002'/(a+'.pt'));rr=plans[ds][r['query_ordinal']];truth=gt[ds][0][r['query_ordinal']];span=gt[ds][1][r['query_ordinal']];E=x['E'];saved=next(q for q in z['rows']if q['alias']==a)
  # Whole existing decoder-grid residuals, never fabricate DINO on these nodes.
  for j,fid in enumerate(x['frame_ids']):
   rec=dict(alias=a,dataset=ds,panel=r['panel'],position=j,observed=j in x['planned_positions'],GT_known=int(fid)in truth,teacher_observation_at_node=next((f['probe_available']for f in x['frames']if f['position']==j),False),identity_available=False)
   if rec['GT_known']:
    tr=truth[int(fid)];ei,c,s,st=independent_geo(E[j],tr,rr,ds);rec.update(E_IoU=ei,center_error=c,scale_factor=s,strata={k:bool(v)for k,v in st.items()})
    if ds=='vidstg':
     others=[];tid=tids[rr['key']]
     for obj in raw[rr['source']]['trajectories'][int(fid)]:
      bb=obj.get('bbox',{});b=[float(bb.get(k,0))for k in ['xmin','ymin','xmax','ymax']]
      if int(obj['tid'])!=tid and b[2]>b[0]and b[3]>b[1]:others.append(scalar(E[j],b,rr,ds))
     oi=max(others,default=0.);rec.update(identity_available=True,max_other_IoU=oi);rec['strata']['instance_mismatch']=ei<=.1 and oi>=.5
   actual.append(rec)
  for f in x['frames']:
   pos=f['position'];v=frame_index[a,pos];assert v['GT_known']==(f['frame_id']in truth);assert v['probe_available']==f['probe_available'];checks['planned_coverage_records']+=1
   if not v['GT_known']:continue
   tr=truth[f['frame_id']];ei,c,s,st=independent_geo(E[pos],tr,rr,ds);assert abs(ei-v['E_IoU'])<1e-12 and abs(c-v['normalized_center_error'])<1e-10 and abs(s-v['scale_factor'])<1e-10
   for k,b in st.items():assert bool(b)==v['strata'][k]
   ac=next(t for t in actual if t['alias']==a and t['position']==pos)
   if ds=='vidstg':assert ac['strata']['instance_mismatch']==v['strata']['instance_mismatch'];checks['instance_flags_independent']+=1
   for key,boxes in [('retained',f['retained_boxes']),('native',f['native']['boxes']if f['native']else[])]:
    vals=[scalar(b,tr,rr,ds)for b in boxes];checks['scalar_candidate_IoUs']+=len(vals)
    if vals:
     field='retained_best_gain_pp'if key=='retained'else'native_all_best_gain_pp';diff=abs(100*(max(vals)-ei)-v[field]);assert diff<1e-10;err['candidate_gain']=max(err['candidate_gain'],diff)
   for ss in f['samples']:
    t=state_index[a,pos,ss['index']];vi=scalar(ss['input'],tr,rr,ds);vo=scalar(ss['box'],tr,rr,ds);assert abs(vi-t['input_IoU'])<1e-12 and abs(vo-t['teacher_IoU'])<1e-12;assert t['sample_recovery']==bool(ss['accepted']and vo>=.5 and vo-ei>=.1);checks['scalar_conditional_state_IoUs']+=2
  ids=np.asarray(x['frame_ids']);known=np.array([int(f)in truth for f in ids]);obs=np.isin(np.arange(len(ids)),x['planned_positions'])
  for arm,b in x['outputs'].items():
   assert np.array_equal(b[~obs],E[~obs]);v=official(b,rr,truth,span,x['interval'],ds);tube=DenseTube(b,rr,truth,span,clip=ds=='hc2');si=np.array([iou_gt(b[i],truth[int(fid)],rr,ds)if known[i]else np.nan for i,fid in enumerate(ids)]);mask=~np.isin(tube.fids,ids);v.update(unobserved_actual_s=float(si[known&~obs].mean())if(known&~obs).any()else None,observed_actual_s=float(si[known&obs].mean())if(known&obs).any()else None,benchmark_interpolated_s=float(tube.iou[mask].mean())if mask.any()else None)
   for k,u in v.items():
    vv=saved['values'][arm][k];assert u is None and vv is None or abs(u-vv)<1e-12;checks['independent_metrics']+=1
   assert v['unobserved_actual_s']==saved['values']['E_MBR']['unobserved_actual_s']
  checks['parents']+=1
 # Independently recompute selected stratified per-parent bootstrap summaries.
 for ds,panels in z['groups'].items():
  for pan,sts in panels.items():
   for st,g in sts.items():
    ff=[f for f in z['frames']if f['dataset']==ds and(pan=='combined'or f['panel']==pan)and f['GT_known']and f['strata'].get(st,False)];aliases=sorted({f['alias']for f in ff});assert len(ff)==g['known_frames']and len(aliases)==g['eligible_parent_count']
    for field,b in g['fields'].items():
     vals=[]
     for a in aliases:
      va=[f[field]for f in ff if f['alias']==a and f.get(field)is not None]
      if va:vals.append(float(np.mean(va)))
     assert len(vals)==b['parents']
     if vals:
      ci=np.quantile(np.random.default_rng(20261018).choice(np.array(vals),size=(10000,len(vals)),replace=True).mean(1),[.025,.975]);assert abs(np.mean(vals)-b['mean'])<1e-12 and np.allclose(ci,b['CI95'],atol=1e-12,rtol=0)
     checks['independent_stratum_bootstraps']+=1
 groups={}
 for ds in ['hc2','vidstg']:
  groups[ds]={}
  for scope in ['all_actual','unobserved_actual','observed_actual']:
   sub=[r for r in actual if r['dataset']==ds and(scope=='all_actual'or r['observed']==(scope=='observed_actual'))];known=[r for r in sub if r['GT_known']];ss={}
   for st in STRATA:
    rr=[r for r in known if r['strata'].get(st,False)];ss[st]=dict(nodes=len(rr),parents=len({r['alias']for r in rr}),nodes_with_actual_DINO_observation=sum(r['teacher_observation_at_node']for r in rr))
   groups[ds][scope]=dict(total_nodes=len(sub),GT_known_nodes=len(known),unknown_GT_nodes=len(sub)-len(known),identity_available_nodes=sum(r['identity_available']for r in known),strata=ss)
 recovery=read(D/'RECOVERY_LOCK.json')
 for f,h in recovery['preserved_partial_files'].items():assert sha(ROOT/f)==h
 verify(True);put(P/'ACTUAL_NODE_DIAGNOSIS_PUBLIC.json',dict(status='complete',groups=groups,rows=actual,new_teacher_at_unobserved_nodes=False,GT_offline_only=True,HC_identity_unknown=True));put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),max_errors=dict(err),CPU_seconds=time.perf_counter()-start,source_cache_partial_and_production_formal_hashes_preserved=True,no_model_or_teacher_or_adaptation=True,all_readouts_before_GT=True));print('root audit pass',dict(checks),dict(err),flush=True)
if __name__=='__main__':main()

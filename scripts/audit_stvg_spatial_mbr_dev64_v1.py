"""Root independent scalar MBR, official metric and parent bootstrap audit."""
import os,sys,time,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_spatial_mbr_dev64_v1 import D,P,ARMS,ENDPOINTS,CONTRASTS,verify,load,read,put,sha

def scalar(a,b):
 ax,ay,aw,ah=map(float,a);bx,by,bw,bh=map(float,b);al,at,ar,ab=ax-aw/2,ay-ah/2,ax+aw/2,ay+ah/2;bl,bt,br,bb=bx-bw/2,by-bh/2,bx+bw/2,by+bh/2;i=max(0.,min(ar,br)-max(al,bl))*max(0.,min(ab,bb)-max(at,bt));return i/(aw*ah+bw*bh-i)

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 torch.set_num_threads(4);start=time.perf_counter();p=verify(full=True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_GT'];res=read(P/'RESULTS_PUBLIC.json');assert len(res['rows'])==64;checks=collections.Counter();errors=collections.Counter()
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  x=load(D/'predictions'/(r['alias']+'.pt'));src=load(ROOT/r['path']);fit=src['fits'][r['arm']];t=fit['rounds'][-1];bs=t['candidates'].numpy();v=t['feedback']['valid'].numpy();center=fit['path'][-2]['boxes'].numpy();allowed=src['allowed'].numpy();active=allowed&t['feedback']['informative'].numpy();assert np.array_equal(x['outputs']['Last_center'],center)and np.array_equal(x['outputs']['Gaussian'],fit['final'].numpy())and np.array_equal(x['outputs']['E_MBR'],src['outputs']['E_MBR']);assert np.array_equal(bs[:,0],center)
  rec=next(z for z in res['rows']if z['alias']==r['alias']);assert rec['active']==bool(t['feedback']['per_anchor_rewards'].shape[0]);rr={a:center.copy()for a in ['P_MBR','Q_MBR']}
  for j in range(len(bs)):
   d=x['decisions'][j];assert d['active']==bool(active[j])
   if not active[j]:assert np.array_equal(center[j],x['outputs']['Gaussian'][j]);continue
   m=np.array([[scalar(a,b)for b in bs[j]]for a in bs[j]]);checks['scalar_pairwise_IoUs']+=17*17
   for arm,key in [('P_MBR','student_prob'),('Q_MBR',None)]:
    w=(t[key][j]if key else t['feedback']['prob'][j]).numpy().copy();assert abs(w[~v[j]]).sum()<1e-12;w[~v[j]]=0;w/=w.sum();utility=np.array([sum(float(w[k])*scalar(bs[j,i],bs[j,k])for k in range(17))for i in range(17)]);errors['utility']=max(errors['utility'],float(abs(utility-d[arm]['all_utilities']).max()));assert np.allclose(utility,d[arm]['all_utilities'],rtol=0,atol=2e-13)
    best=max(utility[v[j]]);ties=[i for i in range(17)if v[j,i]and utility[i]>=best-1e-12];ix=0 if 0 in ties else min(ties);assert ix==d[arm]['index'];assert v[j,ix];rr[arm][j]=bs[j,ix];checks['independent_MBR_choices']+=1
  for a in rr:assert np.array_equal(rr[a],x['outputs'][a]);checks['readout_outputs']+=1
  meta=plans[r['dataset']][r['query_ordinal']];truth,span=gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']];ids=np.asarray(x['frame_ids']);known=np.array([int(f)in truth for f in ids]);obs=np.isin(np.arange(len(ids)),x['planned_positions']);assert int((known&~obs).sum())==rec['known_unobserved_actual'];checks['all_parent_denominators']+=1
  for arm,z in x['outputs'].items():
   assert np.array_equal(z[~allowed],x['outputs']['E_MBR'][~allowed])and np.array_equal(z[:,2:],x['outputs']['E_MBR'][:,2:]);vals=official(z,meta,truth,span,x['interval'],r['dataset']);tube=DenseTube(z,meta,truth,span,clip=r['dataset']=='hc2');si=np.array([iou_gt(z[i],truth[int(f)],meta,r['dataset'])if known[i]else np.nan for i,f in enumerate(ids)]);interp=~np.isin(tube.fids,ids)
   vals.update(unobserved_actual_s=float(si[known&~obs].mean())if(known&~obs).any()else None,observed_actual_s=float(si[known&obs].mean())if(known&obs).any()else None,benchmark_interpolated_s=float(tube.iou[interp].mean())if interp.any()else None)
   for e,a in vals.items():
    b=rec['values'][arm][e];assert a is None and b is None or a is not None and b is not None and abs(a-b)<1e-12
    if a is not None:errors['metrics']=max(errors['metrics'],abs(a-b))
    checks['metric_values']+=1
  checks['parents']+=1
 for pan,g in res['groups'].items():
  for ds,s in g.items():
   rr=[r for r in res['rows']if r['dataset']==ds and(pan=='combined'or r['panel']==pan)];assert len(rr)==s['parents']
   for a,b in CONTRASTS:
    for ep in ENDPOINTS:
     v=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in rr if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);saved=s['paired'][a+'_minus_'+b][ep];assert len(v)==saved['parents']
     if len(v):
      draws=np.random.default_rng(20261018).choice(v,size=(10000,len(v)),replace=True);ci=np.quantile(draws.mean(1),[.025,.975]);assert abs(float(v.mean())-saved['mean'])<1e-12 and np.allclose(ci,saved['CI95'],rtol=0,atol=1e-12)
     else:assert saved['mean']is None
     checks['parent_bootstrap_contrasts']+=1
 fix=read(D/'RECOVERY_LOCK.json')
 for f,h in fix['preserved_predictions'].items():assert sha(ROOT/f)==h
 verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),max_absolute_errors=dict(errors),original32_partial_outputs_bitexact_preserved=True,all64_barrier_before_GT=True,formal64_and_configuration_and_production_unchanged=True,GT_unknown_and_original_denominators_preserved=True,no_new_model_expert_training_OPD=True,CPU_seconds=time.perf_counter()-start));print('root independent audit passed',dict(checks),dict(errors),flush=True)
if __name__=='__main__':main()

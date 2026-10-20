"""Seal and score initialization/optimizer coupling, all parents and controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time,math,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_initialization_coupled_common_v1 import ROOT,D,P,MAIN_D,MAIN_P,BASE,read,put,sha,load,initial,verify

def writeback(E,b,a):
 if a==0:return E.copy()
 if a==1:return b.copy()
 return np.concatenate((E[:,:2]+a*(b[:,:2]-E[:,:2]),E[:,2:]*np.exp(a*np.log(b[:,2:]/E[:,2:]))),-1)
def definitions(ds,phase):
 p=verify();spec=p['datasets'][ds]
 if phase=='coarse':configs=spec['coarse_configs'];steps=spec['coarse_steps'];alphas=spec['coarse_alphas']
 else:
  fine=read(D/ds/'FINE_PLAN_LOCK.json');configs={**fine['configs'],fine['center_id']:fine['center_config']};steps=fine['steps'];alphas=fine['alphas']
 return {cid+f'_S{k:02}_A{int(round(a*100)):03}':dict(config_id=cid,lr=cfg['lr'],tau=cfg['tau'],steps=k,alpha=a,initialization_gamma=cfg['initialization_gamma'],fit_phase='coarse'if cid.startswith('G')else'fine')for cid,cfg in configs.items()for k,a in itertools.product(steps,alphas)}
def getfit(ds,phase,cid,alias):
 bar=read(D/ds/f'{phase.upper()}_FIT_BARRIER.json');e=bar['files'][cid+'/'+alias];assert sha(ROOT/e['path'])==e['sha256'];return load(ROOT/e['path'])['fit']
def seal(ds,phase):
 p=verify();spec=p['datasets'][ds];defs=definitions(ds,phase);files={}
 for rec in [r for r in p['records']if r['dataset']==ds]:
  a=rec['alias'];old=load(BASE/f'predictions/{ds}/{a}.pt');S=np.asarray(old['Source'],float);E1=np.asarray(old['E_Ref']);out=dict(frame_ids=np.array(old['frame_ids']),positions8=np.array(old['planned_positions']),Strong_interval=np.array(old['intervals']['Strong']),Original_Base=old['Live'],E_G100=E1)
  for g in spec['gammas']:out['E_G'+str(int(g*100)).zfill(3)]=initial(S,E1,g)
  with np.load(MAIN_D/ds/spec['reference_phase']/'predictions'/f'{a}.npz')as z:out['Original_Tuned']=z[spec['reference_arm']].copy()
  cache={}
  for arm,c in defs.items():
   key=c['fit_phase'],c['config_id']
   if key not in cache:cache[key]=getfit(ds,*key,a)
   f=cache[key];E=initial(S,E1,c['initialization_gamma']);assert np.array_equal(f['path'][0]['boxes'].numpy(),E);out[arm]=writeback(E,f['path'][c['steps']]['boxes'].numpy(),c['alpha']);assert np.isfinite(out[arm]).all()and(out[arm][:,2:]>0).all()
  dest=D/ds/phase/'predictions'/f'{a}.npz';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez(dest,**out);files[str(dest.relative_to(ROOT))]=sha(dest)
 put(D/ds/phase/'PREDICTION_BARRIER.json',dict(status='sealed_all128_before_gamma_OPD_GT',definitions=defs,files=files,parents=128,GT_read=False,epoch=time.time()));print('GAMMA_READOUT_SEALED',ds,phase,len(defs),flush=True)

def score(ds,phase):
 import torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 p=verify();assert read(P/ds/phase/'PRE_GT_AUDIT_PUBLIC.json')['status']=='pass';assert read(P/ds/phase/'READOUT_AUDIT_PUBLIC.json')['status']=='pass';bar=read(D/ds/phase/'PREDICTION_BARRIER.json');defs=bar['definitions'];spec=p['datasets'][ds];controls=['E_G100']+['E_G'+str(int(g*100)).zfill(3)for g in spec['gammas']]+['Original_Base','Original_Tuned'];arms=controls+list(defs)
 for f,h in bar['files'].items():assert sha(ROOT/f)==h
 put(D/ds/phase/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_phase_predictions_sealed=True,barrier_sha256=sha(D/ds/phase/'PREDICTION_BARRIER.json')))
 records=[r for r in p['records']if r['dataset']==ds];raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];gt,spans,prov=truths(ds,dict(parents=[r['query_ordinal']for r in records]));screen={r['alias']:r for r in read(MAIN_P/'INITIALIZATION_STRENGTH_RESULTS_PUBLIC.json')['datasets'][ds]['rows']};base={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-20/single_referdino_bidirectional256/RESULTS_PUBLIC.json')['directions'][ds]['rows']};tuned={r['alias']:r for r in read(MAIN_P/ds/spec['reference_phase']/'RESULTS_PUBLIC.json')['rows']};rows=[];checks=ident=0;errmax=0.;beg=time.perf_counter()
 for rec in records:
  a,q=rec['alias'],rec['query_ordinal'];row=raw[q];values={}
  with np.load(D/ds/phase/'predictions'/f'{a}.npz')as z:
   ids=z['frame_ids'].tolist();iv=z['Strong_interval'].tolist();pos=z['positions8'].tolist();fixed=[f for f in ids if f in gt[q]];obs=[f for f in fixed if f in {ids[i]for i in pos}];unobs=[f for f in fixed if f not in obs];lookup={f:i for i,f in enumerate(ids)};node=[];dense=[];denseids=None
   for arm in arms:
    b=z[arm];tube=DenseTube(b,row,gt[q],spans[q],clip=ds=='hc2');v=tube.score(iv);off=official(b,row,gt[q],spans[q],iv,ds);e=max(abs(v[k]-off[k])for k in ['v','t','s']);assert e<2e-10;checks+=1;errmax=max(errmax,e);pix=xyxy(b,row['input']['width'],row['input']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix;ni=np.array([float(box_iou(pix[lookup[f]],gt[q][f]))for f in fixed]);li={f:i for i,f in enumerate(fixed)}
    for key,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[key]=float(np.mean([ni[li[f]]for f in group]))if group else None
    values[arm]=v;node.append(ni);dense.append(tube.iou);denseids=tube.fids
    ref=screen[a]['values'][arm[2:]]if arm.startswith('E_G')else base[a]['values']['Full']if arm=='Original_Base'else tuned[a]['values'][spec['reference_arm']]if arm=='Original_Tuned'else None
    if ref is not None:
     for k in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']:assert abs(v[k]-ref[k])<2e-10;ident+=1
   dest=D/ds/phase/'private_metric_receipts'/f'{a}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez(dest,arms=np.array(arms),actual_ids=np.array(fixed),observed_ids=np.array(obs),actual_IoU=np.array(node),dense_GT_ids=denseids,dense_GT_IoU=np.array(dense),interval=np.array(iv),GT_span=np.array(spans[q]));rows.append(dict(alias=a,values=values,GT_supported_actual_nodes=len(fixed),observed_actual_nodes=len(obs),unobserved_actual_nodes=len(unobs)))
  if len(rows)%32==0:print('GAMMA_SCORE',ds,phase,len(rows),len(arms),flush=True)
 metrics=list(rows[0]['values'][controls[0]]);means={a:{k:float(np.mean([r['values'][a][k]for r in rows if r['values'][a][k]is not None]))for k in metrics}for a in arms};pairs=[]
 for arm,c in defs.items():
  own='E_G'+str(int(c['initialization_gamma']*100)).zfill(3)
  pairs.extend((arm,ref)for ref in [own,'Original_Base','Original_Tuned'])
 rng=np.random.default_rng(20261020);ix=rng.integers(0,128,(10000,128));weights=np.stack([np.bincount(j,minlength=128)for j in ix])/128.;paired={}
 for key in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']:
  x=np.array([[100*(r['values'][a][key]-r['values'][b][key])for a,b in pairs]for r in rows]);ci=np.quantile(weights@x,[.025,.975],axis=0)
  for j,(a,b)in enumerate(pairs):paired.setdefault(a+'_minus_'+b,{})[key]=dict(mean=float(x[:,j].mean()),CI95=ci[:,j].tolist(),n=128)
 for a,b in pairs:
  x=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);paired[a+'_minus_'+b]['tail']=dict(positive=int((x>1e-10).sum()),negative=int((x< -1e-10).sum()),unchanged=int((abs(x)<=1e-10).sum()),below_minus5=int((x< -5).sum()),positive_contribution_pp=float(np.maximum(x,0).mean()),negative_contribution_pp=float(np.minimum(x,0).mean()),worst=[dict(alias=rows[i]['alias'],delta_pp=float(x[i]))for i in np.argsort(x)[:8]])
 ranked=sorted(defs,key=lambda a:(-means[a]['v'],abs(defs[a]['alpha']-1),defs[a]['steps'],defs[a]['lr'],-defs[a]['tau'],abs(defs[a]['initialization_gamma']-1)))
 put(P/ds/phase/'RESULTS_PUBLIC.json',dict(status='completed_initialization_optimizer_coupled_development_phase',dataset=ds,phase=phase,parents=128,definitions=defs,means=means,paired=paired,rows=rows,development_ranking=ranked,independent_confirmation=False,GT_provenance=prov,CPU_scoring_seconds=time.perf_counter()-beg));put(P/ds/phase/'AUDIT_PUBLIC.json',dict(status='pass',official_dense_checks=checks,max_abs=errmax,original_and_initialization_control_identity_checks=ident,all128_retained=True));print('GAMMA_PHASE_SCORED',ds,phase,ranked[0],means[ranked[0]],flush=True)

def fine_plan(ds):
 p=verify();r=read(P/ds/'coarse/RESULTS_PUBLIC.json');assert read(P/ds/'coarse/ROOT_AUDIT_PUBLIC.json')['status']=='pass';a=r['development_ranking'][0];c=r['definitions'][a];cfg=p['datasets'][ds]['coarse_configs'][c['config_id']];configs={f'F{i:02}':dict(cfg,lr=lr,tau=tau)for i,(lr,tau)in enumerate(itertools.product([cfg['lr']/math.sqrt(2),cfg['lr']*math.sqrt(2)],[cfg['tau']/math.sqrt(2),cfg['tau']*math.sqrt(2)]))};f=dict(coarse_winner=a,coarse_results_sha256=sha(P/ds/'coarse/RESULTS_PUBLIC.json'),center_id=c['config_id'],center_config=cfg,configs=configs,steps=sorted(set([max(1,c['steps']-5),c['steps'],min(40,c['steps']+5)])),alphas=sorted(set([max(.05,c['alpha']-.25),c['alpha'],c['alpha']+.25])),epoch=time.time());assert not(D/ds/'FINE_PLAN_LOCK.json').exists();put(D/ds/'FINE_PLAN_LOCK.json',f);put(P/ds/'FINE_PLAN_PUBLIC.json',f);print('GAMMA_FINE_LOCKED',ds,a,flush=True)
if __name__=='__main__':
 action,ds=sys.argv[1:3]
 if action=='seal':seal(ds,sys.argv[3])
 elif action=='score':score(ds,sys.argv[3])
 elif action=='fine-plan':fine_plan(ds)

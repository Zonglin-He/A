"""Seal and score each complete coarse/fine phase; choose one global Vid config."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time,math,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE as OLD,read,put,sha,load,verify
RUN=D;PUBLIC=P;DS=None
def bind(ds):
 global D,P,DS
 DS=ds;D=RUN/ds;P=PUBLIC/ds
def baseline_arm(p):return p['baseline']+f"_S{p['baseline_steps'][DS]:02}_A100"
def array(x):return x.detach().cpu().numpy().copy()if hasattr(x,'detach')else np.array(x,copy=True)
def writeback(E,b,a):
 if a==0:return E.copy()
 if a==1:return b.copy()
 return np.concatenate((E[:,:2]+a*(b[:,:2]-E[:,:2]),E[:,2:]*np.exp(a*np.log(b[:,2:]/E[:,2:]))),-1)
def getfit(phase,key,alias):
 bar=read(D/f'{phase.upper()}_FIT_BARRIER.json');entry=bar['files'][key+'/'+alias];f=ROOT/entry['path'];assert sha(f)==entry['sha256'];z=load(f)
 return z['fit']
def definitions(phase):
 p=verify();out={}
 if phase=='coarse':configs=p['configs'][DS];steps=p['coarse_steps'];alphas=p['coarse_alphas']
 else:
  f=read(D/'FINE_PLAN_LOCK.json');configs={**f['configs'],f['center_id']:f['center_config']};steps=f['steps'];alphas=f['alphas']
 for key,cfg in configs.items():
  for k,a in itertools.product(steps,alphas):
   arm=f'{key}_S{k:02}_A{int(round(a*100)):03}';out[arm]=dict(config_id=key,lr=cfg['lr'],tau=cfg['tau'],steps=k,alpha=a,fit_phase='coarse'if key.startswith('C')else'fine')
 base=p['baseline'];out.setdefault(baseline_arm(p),dict(config_id=base,lr=.01,tau=.2,steps=p['baseline_steps'][DS],alpha=1.,fit_phase='coarse'))
 return out
def seal(phase):
 p=verify();defs=definitions(phase);files={};start=time.perf_counter()
 for rec in [r for r in p['records']if r['dataset']==DS]:
  a=rec['alias'];old=load(OLD/f'predictions/{DS}/{a}.pt');E=array(old['E_Ref']);out=dict(E_Strong=E,frame_ids=np.array(old['frame_ids']),positions8=np.array(old['planned_positions']),Strong_interval=np.array(old['intervals']['Strong']));fitcache={}
  for arm,c in defs.items():
   key=(c['fit_phase'],c['config_id'])
   if key not in fitcache:fitcache[key]=getfit(*key,a)
   fit=fitcache[key];assert np.array_equal(array(fit['path'][0]['boxes']),E);b=array(fit['path'][c['steps']]['boxes']);assert np.array_equal(writeback(E,b,0),E);out[arm]=writeback(E,b,c['alpha']);assert np.isfinite(out[arm]).all()and(out[arm][:,2:]>0).all()
  assert np.array_equal(out[baseline_arm(p)],array(old['Live']))
  f=D/phase/'predictions'/f'{a}.npz';f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();np.savez(f,**out);files[str(f.relative_to(ROOT))]=sha(f)
 put(D/phase/'PREDICTION_BARRIER.json',dict(status='sealed',parents=128,files=files,definitions=defs,GT_read=False,epoch=time.time(),phase=phase,plan_sha256=sha(RUN/'PLAN_LOCK.json'),seconds=time.perf_counter()-start))
 print('READOUT_SEAL',phase,len(defs),flush=True)
def score(phase):
 import torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 p=verify();assert read(P/phase/'PRE_GT_AUDIT_PUBLIC.json')['status']=='pass'
 for f,h in read(RUN/'SCORING_IMPLEMENTATION_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
 bar=read(D/phase/'PREDICTION_BARRIER.json');assert bar['plan_sha256']==sha(RUN/'PLAN_LOCK.json');defs=bar['definitions'];arms=['E_Strong']+list(defs);base=baseline_arm(p)
 for f,h in bar['files'].items():assert sha(ROOT/f)==h
 put(D/phase/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_readouts_sealed=True,barrier_sha256=sha(D/phase/'PREDICTION_BARRIER.json'),stage=phase,development_selection=True))
 start=time.perf_counter();gt,spans,prov=truths(DS,dict(parents=[r['query_ordinal']for r in [r for r in p['records']if r['dataset']==DS]]));raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{DS}/PLAN.json')['rows'];original={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-20/single_referdino_bidirectional256/RESULTS_PUBLIC.json')['directions'][DS]['rows']};rows=[];checks=0;errmax=0.;oldchecks=0
 for rec in [r for r in p['records']if r['dataset']==DS]:
  a,q=rec['alias'],rec['query_ordinal'];r=raw[q];values={}
  with np.load(D/phase/'predictions'/f'{a}.npz')as z:
   ids=z['frame_ids'].tolist();iv=z['Strong_interval'].tolist();pos=z['positions8'].tolist();assert ids==r['frame_ids'];fixed=[f for f in ids if f in gt[q]];obs=[f for f in fixed if f in {ids[i]for i in pos}];unobs=[f for f in fixed if f not in set(obs)];lookup={f:i for i,f in enumerate(ids)};node=[];dense=[];dense_ids=None
   for arm in arms:
    b=z[arm];tube=DenseTube(b,r,gt[q],spans[q],clip=DS=='hc2');v=tube.score(iv);off=official(b,r,gt[q],spans[q],iv,DS);err=max(abs(v[k]-off[k])for k in ['v','t','s']);assert err<2e-10;checks+=1;errmax=max(errmax,err)
    pix=xyxy(b,r['input']['width'],r['input']['height']);pix=np.maximum(pix,0)if DS=='hc2'else pix;node_iou=np.array([float(box_iou(pix[lookup[f]],gt[q][f]))for f in fixed]);fi={f:j for j,f in enumerate(fixed)}
    for key,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[key]=float(np.mean([node_iou[fi[f]]for f in group]))if group else None
    values[arm]=v;assert v['t']==values['E_Strong']['t'];node.append(node_iou);dense.append(tube.iou);dense_ids=tube.fids
    if arm in ['E_Strong',base]:
     old=original[a]['values']['Without_spatial_adaptation'if arm=='E_Strong'else'Full']
     for key in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']:assert abs(v[key]-old[key])<2e-10;oldchecks+=1
   private=D/phase/'private_metric_receipts'/f'{a}.npz';private.parent.mkdir(parents=True,exist_ok=True);np.savez(private,arms=np.array(arms),actual_ids=np.array(fixed),observed_ids=np.array(obs),actual_IoU=np.array(node),dense_GT_ids=dense_ids,dense_GT_IoU=np.array(dense),interval=np.array(iv),GT_span=np.array(spans[q]))
   rows.append(dict(alias=a,values=values,actual_nodes=len(ids),GT_supported_actual_nodes=len(fixed),observed_actual_nodes=len(obs),unobserved_actual_nodes=len(unobs)))
  if len(rows)%16==0:print('SCORE',phase,len(rows),len(arms),flush=True)
 metrics=list(rows[0]['values']['E_Strong']);means={arm:{k:float(np.mean([r['values'][arm][k]for r in rows if r['values'][arm][k]is not None]))for k in metrics}for arm in arms}
 pairs=[(arm,b)for arm in defs for b in ['E_Strong',base]if arm!=b];rng=np.random.default_rng(20261020);indices=rng.integers(0,128,(10000,128));weights=np.stack([np.bincount(i,minlength=128)for i in indices])/128.;comparisons={}
 for key in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']:
  x=np.array([[100*(r['values'][a][key]-r['values'][b][key])for a,b in pairs]for r in rows]);CI=np.quantile(weights@x,[.025,.975],axis=0)
  for j,(a,b)in enumerate(pairs):comparisons.setdefault(a+'_minus_'+b,{})[key]=dict(mean=float(x[:,j].mean()),CI95=CI[:,j].tolist(),n=128)
 for a,b in pairs:
  x=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);comparisons[a+'_minus_'+b]['tail']=dict(positive=int((x>1e-10).sum()),negative=int((x< -1e-10).sum()),unchanged=int((abs(x)<=1e-10).sum()),below_minus5=int((x< -5).sum()),positive_contribution_pp=float(np.maximum(x,0).mean()),negative_contribution_pp=float(np.minimum(x,0).mean()),worst=[dict(alias=rows[j]['alias'],delta_pp=float(x[j]))for j in np.argsort(x)[:8]])
 ranked=sorted(defs,key=lambda a:(-means[a]['v'],abs(defs[a]['alpha']-1),defs[a]['steps'],defs[a]['lr'],-defs[a]['tau']))
 result=dict(status='completed_full128_development_parameter_phase',phase=phase,parents=128,queries=128,seed=41,definitions=defs,means=means,paired=comparisons,rows=rows,development_ranking=ranked,GT_provenance=prov,independent_confirmation=False,per_video_choice=False,readoutalpha_is_not_optimizer_lr=True,costs=dict(CPU_scoring_seconds=time.perf_counter()-start,GPU_ledger_path=str((D/'GPU_LEDGER.json').relative_to(ROOT))))
 put(P/phase/'RESULTS_PUBLIC.json',result);put(P/phase/'AUDIT_PUBLIC.json',dict(status='pass',official_dense_checks=checks,maximum_error=errmax,old_E40_identity_checks=oldchecks,fixed_time_and_all128=True))
 print('PHASE_SCORED',phase,ranked[0],means[ranked[0]],flush=True)
def fine_plan():
 p=verify();r=read(P/'coarse/RESULTS_PUBLIC.json');winner=r['development_ranking'][0];c=r['definitions'][winner];cfg=p['configs'][DS][c['config_id']];configs={}
 for i,(lr,tau)in enumerate(itertools.product([cfg['lr']/math.sqrt(2),cfg['lr']*math.sqrt(2)],[cfg['tau']/math.sqrt(2),cfg['tau']*math.sqrt(2)])):configs[f'F{i:02}']=dict(cfg,lr=lr,tau=tau)
 f=dict(status='locked_fine_from_global_coarse_winner',coarse_results_sha256=sha(P/'coarse/RESULTS_PUBLIC.json'),coarse_winner=winner,center_id=c['config_id'],center_config=cfg,configs=configs,steps=sorted(set([max(1,c['steps']-5),c['steps'],min(40,c['steps']+5)])),alphas=sorted(set([max(.05,c['alpha']-.25),c['alpha'],c['alpha']+.25])),epoch=time.time())
 assert not(D/'FINE_PLAN_LOCK.json').exists();put(D/'FINE_PLAN_LOCK.json',f);put(P/'FINE_PLAN_PUBLIC.json',f);print('FINE_LOCKED',f,flush=True)
if __name__=='__main__':
 action=sys.argv[1];bind(sys.argv[2])
 if action=='seal':seal(sys.argv[3])
 elif action=='score':score(sys.argv[3])
 elif action=='fine-plan':fine_plan()

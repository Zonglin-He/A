"""Finite CPU-only saved-step/output-writeback study. No inference or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
D=ROOT/'artifacts/stvg_vid_saved_parameters128_v1'
P=ROOT/'results/stvg_error_state/2026-10-20/vid_saved_parameters128'
OLD=ROOT/'artifacts/stvg_spatial_strong_bidirectional256_v1'
RESULT=ROOT/'results/stvg_error_state/2026-10-20/spatial_strong_bidirectional256/RESULTS_PUBLIC.json'
ARMS=['E_Strong']+[f'Step{k}_A{int(a*100):03}'for k in [20,30,40]for a in [.25,.5,1.]]
def read(f):return json.loads(Path(f).read_text())
def put(f,x):Path(f).parent.mkdir(parents=True,exist_ok=True);Path(f).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f):return torch.load(f,map_location='cpu',weights_only=False)
def array(x):return x.detach().cpu().numpy().copy()if torch.is_tensor(x)else np.array(x,copy=True)
def writeback(E,b,a):
 if a==0:return E.copy()
 if a==1:return b.copy()
 return np.concatenate((E[:,:2]+a*(b[:,:2]-E[:,:2]),E[:,2:]*np.exp(a*np.log(b[:,2:]/E[:,2:]))),-1)
def pins_ok(p):
 for f,h in {**p['pins'],**p['source_files']}.items():assert sha(ROOT/f)==h,f
def prepare():
 assert not(D/'PLAN_LOCK.json').exists()
 torch.set_num_threads(2);old=read(OLD/'PLAN_LOCK.json');records=[r for r in old['records']if r['dataset']=='vidstg'];assert len(records)==128 and len({r['parent']for r in records})==128
 inherited=read(ROOT/'artifacts/stvg_vid_spatial_failure_cached_v1/SCOPE.json')['pins']
 own=['protocols/stvg_vid_saved_parameters128_v1.md','scripts/evaluate_stvg_vid_saved_parameters128_v1.py',
 'scripts/score_stvg_opd_p1_v1.py','scripts/diagnose_tastvg_pipeline_cpu_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
 'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py',
 'artifacts/decota_paper_experiments_v1/vidstg/PLAN.json']
 source_files={};new_files={};cache_exists=0;beg=time.perf_counter()
 for rec in records:
  a=rec['alias'];pf=OLD/f'predictions/vidstg/{a}.pt';tf=OLD/f'traces/vidstg/{a}.pt'
  source_files.update({str(f.relative_to(ROOT)):sha(f)for f in [pf,tf]})
  z,tr=load(pf),load(tf);fit=tr['fits']['Live_41'];E=array(z['E'])
  assert np.array_equal(array(fit['path'][0]['boxes']),E)
  assert np.array_equal(array(fit['path'][40]['boxes']),array(z['Live'][41]))
  assert fit['actual_updates']==40 and fit['configuration']['tau']==.2 and fit['configuration']['lr']==.01
  assert not z['GT_read']and not tr['GT_read'];assert list(z['planned_positions'])==fit['positions']
  out={'frame_ids':np.array(z['frame_ids']),'Strong_interval':np.array(z['intervals']['Strong']),
       'positions8':np.array(z['planned_positions']),'E_Strong':E}
  for k in [20,30,40]:
   b=array(fit['path'][k]['boxes']);assert fit['path'][k]['step']==k and np.array_equal(writeback(E,b,0),E)
   for alpha in [.25,.5,1.]:
    name=f'Step{k}_A{int(alpha*100):03}';v=writeback(E,b,alpha);assert np.isfinite(v).all()and(v[:,2:]>0).all();out[name]=v
    if alpha==1:assert np.array_equal(v,b)
  dest=D/'predictions'/f'{a}.npz';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez(dest,**out);new_files[str(dest.relative_to(ROOT))]=sha(dest)
  # Inventory only; no substitute RGB forward is authorized.
  cache_exists+=int('spatial_inputs'in tr or 'spatial_inputs'in z)
 p=dict(status='locked_all_predictions_sealed',dataset='vidstg',direction='HC2_to_VidSTG',parents=128,queries=128,seed=41,
   steps=[20,30,40],alphas=[.25,.5,1.],arms=ARMS,records=records,configuration=old['configs']['vidstg'],
   Strong=old['Strong'],writeback='center residual linear; size log-ratio residual linear; alpha1 bitexact original',
   selection='global complete-vIoU, ties alpha1 then fewer steps; development only; no per-video choices',
   existing_decoder_input_caches_in_current_panel=cache_exists,new_decoder_inputs_generated=False,
   different_datasets_may_use_different_fixed_parameters=True,HC_unchanged=True,
   pins={**inherited,**{f:sha(ROOT/f)for f in own}},source_files=source_files,created_epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(D/'PREDICTION_BARRIER.json',dict(status='sealed',files=new_files,parents=128,configurations=len(ARMS),GT_read=False,epoch=time.time(),plan_sha256=sha(D/'PLAN_LOCK.json')))
 pub={k:v for k,v in p.items()if k not in ['records','created_epoch']};put(P/'PROTOCOL_PUBLIC.json',pub)
 put(D/'STATE.json',dict(status='predictions_sealed_pending_scoring',new_model_forwards=0,new_updates=0,new_RGB_or_expert_forwards=0,extract_seconds=time.perf_counter()-beg));print('SEALED',128,len(ARMS),flush=True)
def paired(x):
 x=np.asarray(x,float);g=np.random.default_rng(20261020);v=np.concatenate([x[g.integers(0,len(x),(100,len(x)))].mean(1)for _ in range(100)])
 return dict(mean=float(x.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),n=len(x))
def score():
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 torch.set_num_threads(2);p=read(D/'PLAN_LOCK.json');bar=read(D/'PREDICTION_BARRIER.json');assert sha(D/'PLAN_LOCK.json')==bar['plan_sha256'];pins_ok(p)
 for f,h in bar['files'].items():assert sha(ROOT/f)==h
 start=time.perf_counter();put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),prediction_barrier_sha256=sha(D/'PREDICTION_BARRIER.json'),scope='new offline evaluation explicitly authorized; prediction/parameter generation label-free'))
 gt,spans,prov=truths('vidstg',dict(parents=[r['query_ordinal']for r in p['records']]))
 raw=read(ROOT/'artifacts/decota_paper_experiments_v1/vidstg/PLAN.json')['rows'];old={r['alias']:r for r in read(RESULT)['directions']['vidstg']['rows']}
 rows=[];checks=0;oldchecks=0;maxerror=0.
 for rec in p['records']:
  a,q=rec['alias'],rec['query_ordinal'];r=raw[q]
  with np.load(D/'predictions'/f'{a}.npz')as z:
   ids=z['frame_ids'].tolist();iv=z['Strong_interval'].tolist();pos=z['positions8'].tolist();assert r['frame_ids']==ids
   fixed=[f for f in ids if f in gt[q]];observed=[f for f in fixed if f in {ids[i]for i in pos}];unobserved=[f for f in fixed if f not in set(observed)];lookup={f:i for i,f in enumerate(ids)}
   values={};private=dict(frame_ids=ids,GT_supported_actual_ids=fixed,interval=iv,per_arm={})
   for arm in ARMS:
    box=z[arm];tube=DenseTube(box,r,gt[q],spans[q],clip=False);v=tube.score(iv);off=official(box,r,gt[q],spans[q],iv,'vidstg');err=max(abs(v[k]-off[k])for k in ['v','t','s']);assert err<2e-10;maxerror=max(maxerror,err);checks+=1
    pixel=xyxy(box,r['input']['width'],r['input']['height']);node_iou={f:float(box_iou(pixel[lookup[f]],gt[q][f]))for f in fixed}
    for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',observed),('unobserved_actual_sIoU',unobserved)]:v[name]=float(np.mean([node_iou[f]for f in group]))if group else None
    v['boundary_MAE']=(abs(iv[0]-spans[q][0])+abs(iv[1]-spans[q][1]))/2;values[arm]=v
    private['per_arm'][arm]=dict(actual_node_IoU=node_iou,dense_GT_ids=tube.fids.tolist(),dense_GT_IoU=tube.iou.tolist())
    if arm in ['E_Strong','Step40_A100']:
     prev=old[a]['values']['Without_spatial_adaptation'if arm=='E_Strong'else'Full']
     for key in prev:
      assert abs(prev[key]-v[key])<2e-10,(a,arm,key,prev[key],v[key]);oldchecks+=1
    assert v['t']==values['E_Strong']['t']
   put(D/'private_node_metrics'/f'{a}.json',private)
   rows.append(dict(alias=a,values=values,GT_supported_actual_nodes=len(fixed),observed_actual_nodes=len(observed),unobserved_actual_nodes=len(unobserved)))
  if len(rows)%32==0:print('SCORED',len(rows),flush=True)
 metric_names=list(rows[0]['values']['E_Strong']);means={arm:{key:float(np.mean([r['values'][arm][key]for r in rows if r['values'][arm][key]is not None]))for key in metric_names}for arm in ARMS}
 pairs=[(arm,b)for arm in ARMS[1:]for b in ['E_Strong','Step40_A100']if arm!=b]+[('Step20_A100','Step30_A100')]
 comparisons={}
 for arm,base in pairs:
  dd={key:paired([100*(r['values'][arm][key]-r['values'][base][key])for r in rows if r['values'][arm][key]is not None and r['values'][base][key]is not None])for key in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']}
  v=np.array([100*(r['values'][arm]['v']-r['values'][base]['v'])for r in rows]);dd['tail']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),unchanged=int((np.abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=[dict(alias=rows[i]['alias'],delta_pp=float(v[i]))for i in np.argsort(v)[:8]],best=[dict(alias=rows[i]['alias'],delta_pp=float(v[i]))for i in np.argsort(v)[-8:][::-1]])
  comparisons[arm+'_minus_'+base]=dd
 ranked=sorted(ARMS[1:],key=lambda a:(-means[a]['v'],a[-3:]!='100',int(a[4:6])))
 result=dict(status='completed_offline_saved_predictions_development_comparison',parents=128,queries=128,seed=41,means=means,paired=comparisons,rows=rows,
  development_ranking=ranked,independent_confirmation=False,per_video_parameter_selection=False,
  GT_provenance=prov,all_predictions_sealed_before_new_scoring=True,
  costs=dict(scoring_seconds=time.perf_counter()-start,new_model_forwards=0,new_expert_forwards=0,new_RGB_reads=0,new_student_updates=0,upstream_costs='inherited from original run; no new deployment latency measurement'),
  qualification=dict(decoder_input_cache_found_in_current_Vid128=p['existing_decoder_input_caches_in_current_panel'],new_lr_or_tau_experiment=False,alpha_is_output_postprocessing_not_optimizer_or_new_OPD_rollout=True))
 put(P/'RESULTS_PUBLIC.json',result);put(P/'AUDIT_PUBLIC.json',dict(status='pass',official_vs_Dense_checks=checks,max_abs= maxerror,old_E_and40_metric_identity_checks=oldchecks,identical_time_for_all_arms=True,all128_parents_retained=True))
 pins_ok(p);put(D/'STATE.json',dict(status='scored_pending_independent_audit_publication',**result['costs']));print(json.dumps(dict(means=means,ranking=ranked,audit=read(P/'AUDIT_PUBLIC.json'))),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='score':score()

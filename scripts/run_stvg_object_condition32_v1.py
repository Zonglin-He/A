"""Bounded frozen-query readouts32, global seal, separate CPU scoring and audit."""
from pathlib import Path
import os,sys,time,gc,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_object_condition32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/object_condition32'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';PREFIX=ROOT/'artifacts/stvg_shared_text_query32_v1/prefix'
CODE=['vg_tta/stvg_object_condition32_v1.py','scripts/run_stvg_object_condition32_v1.py','scripts/continue_stvg_object_condition32_v1.py','protocols/stvg_object_condition32_v1.md']
VERIFIER=None

def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(l['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)

def prepare():
 from vg_tta.stvg_object_condition32_v1 import ARMS,properties
 old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
 files=list(CODE)+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','methods/decota_final_simplified_v1/replay.py','vg_tta/stvg_soft_correspondence_v1.py','vg_tta/stvg_dino_query_closedloop32_v1.py','scripts/run_stvg_shared_text_query32_v1.py']
 for r in old['records']:
  a=r['alias'];files +=[str((PRE/k/(a+'.pt')).relative_to(ROOT))for k in ['source','E','predictions']]+[str((PREFIX/(a+'.pt')).relative_to(ROOT))]
 put(D/'CPU_PROPERTIES.json',properties())
 put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,
  reference_interface='second-pass native initial isq/itq',native_first_pass_routing_fixed=True,coefficient_mass_and_query_norm_preserved=True,
  spatial_support='same E-selected positions; E versus actual DINO boxes',motion_support='all actual frames; E or DINO-conditioned E-retaining trajectory',
  region_support='fractional valid-grid cell overlap; no ROI feature averaging before decoder',empty_region='native per-frame weight fallback, recorded',
  no_new_training=True,no_new_DINO=True,no_new_Source_capture=True,no_weight_update=True,no_OPD=True,
  source_commit='33cba24bd3378955026043163eaef79081827ed3',GPU_seconds_cap=900,created_epoch=time.time(),deadline_epoch=time.time()+3600,
  GT_after_full32_seal=True,development_exposed=True,independent_confirmation=False,near_threshold_fraction=.125,no_parameter_search=True))
 pins={**read(ROOT/'artifacts/stvg_shared_text_query32_v1/RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in set(files)}}
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));verify(full=True);status('prepared',0,GT_read=False)
 print('locked32 cached fused prefixes; no new Source/DINO',flush=True)

def distribution(records,logits):
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 g=probabilities(records,logits);ids=sorted(set(sum([x['ids']for x in g],[])));M=independent_cdf(g,ids)
 assert (M>=0).all()and abs(M.sum()-1)<1e-10
 return ids,M,final_mbr(ids,M)['interval']


def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.replay import SpatialReplay
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_shared_text_query32_v1 import input_pool
 from vg_tta.stvg_object_condition32_v1 import ARMS,readout
 p=verify(True);lease=None
 while lease is None:
  verify(True)
  try:lease=gpu()
  except BlockingIOError:status('waiting_shared_GPU_lease',0,GT_read=False);time.sleep(15)
 start=time.perf_counter();files=[];domain=None;model=None;cost=[]
 def check():
  verify(True);assert time.perf_counter()-start+3<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  put(D/'GPU_RUNTIME.json',dict(status='active',pid=os.getpid(),start_epoch=time.time(),shared_lease=True))
  for ix,r in enumerate(p['records']):
   check();tick=time.perf_counter();a=r['alias'];ds=r['dataset'];status('frozen_query_condition_readouts',ix,alias=a,GT_read=False)
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict())
   s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));old=load(PRE/'predictions'/(a+'.pt'));c=load(PREFIX/(a+'.pt'))
   assert torch.equal(c['source'],s['Source'])and c['records']==s['records']
   base=SpatialReplay(model,transfer(c['views'],'cuda'),len(s['frame_ids']));base.restore(base.initial)
   for _,param in base.named:param.requires_grad_(False)
   assert torch.equal(base.zero['boxes'].cpu(),s['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],s['logits']))
   pool=input_pool(s['expert']);positions=[];targets=[]
   for f,k in zip(pool,e['estimate']['selected']):
    if k>=0:positions.append(f['position']);targets.append(f['boxes'][k])
   assert len(positions)==e['estimate']['support_frames'];E=torch.as_tensor(e['E']).double().cuda();source=s['Source'].cuda();dino=torch.as_tensor(np.asarray(targets).reshape(-1,4),device='cuda',dtype=torch.float32)
   vals={};zero=readout(base,E,source,[],dino)
   assert torch.equal(zero['boxes'],E.cpu())and all(torch.equal(x,y)for x,y in zip(zero['logits'],s['logits']))
   ids,M,I=distribution(c['records'],zero['logits']);assert ids==old['frame_ids']and I==old['interval']
   vals['E_MBR']=zero
   rt=time.perf_counter()
   vals['E_region_spatial']=readout(base,E,source,positions,dino,spatial='E')
   vals['DINO_region_spatial']=readout(base,E,source,positions,dino,spatial='DINO')
   vals['Object_temporal']=readout(base,E,source,positions,dino,motion=E)
   vals['Spatial_then_temporal']=readout(base,E,source,positions,dino,spatial='DINO',motion=vals['DINO_region_spatial']['boxes'].cuda())
   torch.cuda.synchronize();readout_seconds=time.perf_counter()-rt
   outputs={k:v['boxes'].numpy()for k,v in vals.items()};outputs['Object_temporal']=E.cpu().numpy()
   assert torch.equal(vals['DINO_region_spatial']['boxes'],vals['Spatial_then_temporal']['boxes'])
   intervals={k:I for k in ARMS};distributions={'E_MBR':M}
   for k in ['E_region_spatial','DINO_region_spatial']:assert all(torch.equal(x,y)for x,y in zip(vals[k]['logits'],zero['logits']))
   for k in ['Object_temporal','Spatial_then_temporal']:
    ii,M1,I1=distribution(c['records'],vals[k]['logits']);assert ii==ids;intervals[k]=I1;distributions[k]=M1
   assert np.array_equal(outputs['Object_temporal'],outputs['E_MBR'])
   if not positions:
    assert all(np.array_equal(outputs[k],outputs['E_MBR'])and intervals[k]==I for k in ARMS)
   assert state_hash(model.state_dict())==mh and not any(p.requires_grad or p.grad is not None for p in model.parameters())
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=ids,planned_positions=old['planned_positions'],positions=positions,targets=dino.cpu(),selected=e['estimate']['selected'],active=bool(positions),
    outputs=outputs,intervals=intervals,values=vals,distributions=distributions,model_hash=mh,zero_exact=True,native_routing_fixed=True,
    compute=dict(new_Source_captures=0,new_DINO_calls=0,readout_seconds=readout_seconds,case_seconds=time.perf_counter()-tick),GT_read=False,pretrained_weights_frozen=True)))
   cost.append(dict(alias=a,seconds=time.perf_counter()-tick,readout_seconds=readout_seconds));print(a,ix+1,round(time.perf_counter()-start,3),flush=True)
   if ix==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-start,rough32_seconds=16*(time.perf_counter()-start),not_a_guarantee=True))
   del base,s,e,old,c,E,source,dino,vals,zero;gc.collect();torch.cuda.empty_cache()
  check();verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,
   GPU_allocation_seconds=time.perf_counter()-start+3,engineering_initialization_seconds_upper_bound=3,final_worker_GPU_seconds=time.perf_counter()-start,
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),costs=cost,new_Source_captures=0,new_DINO_calls=0));status('all32_sealed_before_GT',32,GT_read=False)
 finally:lease.close()


def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(4)
 from vg_tta.stvg_object_condition32_v1 import ARMS
 from scripts.run_stvg_shared_text_query32_v1 import metric
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint_confirm32/RESULTS_PUBLIC.json')['rows']}
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));row=plans[ds][q];gg=gt[ds][0][q];span=gt[ds][1][q];ids=np.array(o['frame_ids']);planned=o['planned_positions']
  values={arm:metric(b,o['intervals'][arm],row,gg,span,ds,o['frame_ids'],planned)for arm,b in o['outputs'].items()}
  known=np.array([int(f)in gg for f in ids]);distance=abs(ids[:,None]-ids[planned][None]).min(1)/max(1,ids[-1]-ids[0]);unobs=~np.isin(np.arange(len(ids)),planned)
  for arm,b in o['outputs'].items():
   si=np.array([iou_gt(b[i],gg[int(f)],row,ds)if known[i]else np.nan for i,f in enumerate(ids)])
   for name,mask in [('near_actual_s',distance<=p['near_threshold_fraction']),('far_actual_s',distance>p['near_threshold_fraction'])]:values[arm][name]=mean(si[known&unobs&mask])
  for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
   x,y=values['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
  diagnostics={}
  for arm in ARMS[1:]:
   v=o['values'][arm];tr=v['traces'];M=o['distributions'].get(arm,o['distributions']['E_MBR']);M0=o['distributions']['E_MBR'];b=o['outputs'][arm];E=o['outputs']['E_MBR']
   diagnostics[arm]=dict(changed_interval=o['intervals'][arm]!=o['intervals']['E_MBR'],total_variation=float(abs(M-M0).sum()/2),
    start_change_frames=o['intervals'][arm][0]-o['intervals']['E_MBR'][0],end_change_frames=o['intervals'][arm][1]-o['intervals']['E_MBR'][1],
    duration_change_frames=values[arm]['duration_frames']-values['E_MBR']['duration_frames'],mean_center_shift=float(np.linalg.norm(b[:,:2]-E[:,:2],axis=-1).mean()),
    max_logit_change=max(float(abs(x-y).max())for x,y in zip(v['logits'],o['values']['E_MBR']['logits'])),
    spatial_query_delta=[float(t['spatial']['delta_norm'])if t['spatial']and not t['spatial']['noop']else 0 for t in tr],
    temporal_query_delta=[float(t['temporal']['delta_norm'])if t['temporal']and not t['temporal']['noop']else 0 for t in tr],
    region_fallback_count=sum(sum(t[k].get('fallback',[]))for t in tr for k in ['spatial','temporal']if t[k]))
  rows.append(dict(alias=a,dataset=ds,active=o['active'],reference_count=len(o['positions']),values=values,diagnostics=diagnostics,
   support=dict(actual_nodes=len(ids),known_nodes=int(known.sum()),unknown_nodes=int((~known).sum()),unobserved_known=int((known&unobs).sum()),near_known=int((known&unobs&(distance<=.125)).sum()),far_known=int((known&unobs&(distance>.125)).sum())),compute=o['compute']))
 endpoints=list(rows[0]['values']['E_MBR']);groups={}
 contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('DINO_region_spatial','E_region_spatial'),('Spatial_then_temporal','DINO_region_spatial')]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds]
  groups[ds]=dict(count=len(rr),active=sum(r['active']for r in rr),references=sum(r['reference_count']for r in rr),
   metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in endpoints}for a in ARMS},
   paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s','near_actual_s','far_actual_s']}for a,b in contrasts},
   boundary_changes={a:dict(changed=sum(r['diagnostics'][a]['changed_interval']for r in rr),expanded=sum(r['diagnostics'][a]['duration_change_frames']>0 for r in rr),contracted=sum(r['diagnostics'][a]['duration_change_frames']<0 for r in rr),mean_total_variation=mean([r['diagnostics'][a]['total_variation']for r in rr]))for a in ARMS[1:]})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,compute={k:v for k,v in bar.items()if k in ['GPU_allocation_seconds','engineering_initialization_seconds_upper_bound','final_worker_GPU_seconds','peak_allocated_bytes','new_Source_captures','new_DINO_calls']},GT_offline_only=True,development_exposed=True,query_equals_parent=True))
 put(D/'CPU_COMPLETION.json',dict(status='complete',count=32));print({ds:{a:{k:round(100*g['metrics'][a][k],6)for k in ['v','t','unobserved_actual_s']}for a in ARMS}for ds,g in groups.items()},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise

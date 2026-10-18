"""Finite true-layer3 feature qualification; scores sealed before offline GT."""
from pathlib import Path
import os,sys,time,gc,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_fine_feature32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/fine_feature32'
PRE=ROOT/'artifacts/stvg_discriminative_dense32_v1';OLD=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';NIGHT=ROOT/'artifacts/stvg_opd_overnight_20261018_v1'
CODE=['vg_tta/stvg_fine_feature32_v1.py','scripts/run_stvg_fine_feature32_v1.py','scripts/continue_stvg_fine_feature32_v1.py','protocols/stvg_fine_feature32_v1.md']
VERIFIER=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(r['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_fine_feature32_v1 import properties,LAYERS
 p=read(OLD/'PLAN_LOCK.json');night=read(NIGHT/'STATE.json');files=CODE+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 for r in p['records']:
  a=r['alias'];files +=[str((PRE/k/(a+'.pt')).relative_to(ROOT))for k in ['features','feedback']]+[str((OLD/'predictions'/(a+'.pt')).relative_to(ROOT))]
 put(D/'CPU_PROPERTIES.json',properties())
 put(D/'PLAN_LOCK.json',dict(status='locked',records=p['records'],count=32,per_domain=16,configuration=p['configuration'],layers=LAYERS,chosen_layer='vis_encoder.0.body.layer3',
  preselected_reason='immediately preceding ResNet stage, stride16 vs old stride32, retains deeper semantics without new projection',
  scoring='original L2 cosine affinity tau.07; fractional-region softIoU; originalGaussian prior and reward tau.1; identical34 states and argmax readout',
  downsampling='adaptive average pool raw layer3 channels to exact original grid, then normalize/match; not interpolate old maps',
  gate='both-domain Fine direct full-v and unobserved actual-s minus E >0 AND Fine minus Coarse full-v >0; confidence intervals descriptive, uncertainty retained; no threshold search',
  no_new_expert=True,no_new_DINO=True,no_training=True,no_new_head=True,new_Source_captures_max=32,GPU_cap_seconds=900,
  all_outputs_before_NEW_GT=True,development_exposed=True,independent_confirmation=False,source_baseline_commit='d0bc23955d0b69dd20ad8416a9a919063b8d4c6b',created_epoch=time.time(),deadline_epoch=min(time.time()+3600,night['deadline_epoch'])))
 pins={**read(PRE/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in files}};put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));status('prepared',0,GT_read=False)


def measure():
 import torch,numpy as np
 import torch.nn.functional as F
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_fine_feature32_v1 import propagate_chunked,downsample,LAYERS
 from vg_tta.stvg_cosine_dense_opd32_v1 import direct
 p=verify(True);lease=None
 while lease is None:
  verify(True)
  try:lease=gpu()
  except BlockingIOError:status('waiting_shared_GPU_lease',0,GT_read=False);time.sleep(15)
 tick=time.perf_counter();model=None;domain=None;cache=collections.OrderedDict();files=[];features=[];cost=[]
 try:
  put(D/'GPU_RUNTIME.json',dict(pid=os.getpid(),start_epoch=time.time(),cap_seconds=900))
  for ix,r in enumerate(p['records']):
   verify(True);assert time.perf_counter()-tick+3<900;assert torch.cuda.mem_get_info()[0]>4*2**30;at=time.perf_counter();a=r['alias'];ds=r['dataset'];status('capture_true_layer3_and_seal_feedback',ix,alias=a,GT_read=False)
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict());cache.clear()
   fine=[];coarse=[];pixel_masks=[]
   h1=model.vis_encoder[0].body.layer3.register_forward_hook(lambda m,args,out:fine.append(out.detach().float().cpu()))
   h2=model.input_proj.register_forward_hook(lambda m,args,out:coarse.append(out.detach().float().cpu()))
   h3=model.vis_encoder.register_forward_pre_hook(lambda m,args:pixel_masks.append(args[0].mask.detach().cpu()))
   try:row,recs,base,native,expert,inp,cap_seconds=capture(model,ds,r['query_ordinal'],cache)
   finally:h1.remove();h2.remove();h3.remove()
   oldsource=load(ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1/source'/(a+'.pt'));assert torch.equal(native['boxes'].cpu(),oldsource['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],oldsource['logits']))
   x=load(PRE/'features'/(a+'.pt'));fb=load(PRE/'feedback'/(a+'.pt'))['feedback'];old=load(OLD/'predictions'/(a+'.pt'));n=len(x['frame_ids']);assert len(fine)==len(coarse)==2
   fm=torch.empty(n,*fine[0].shape[1:]);cm=torch.empty_like(x['maps'])
   for off in [0,1]:fm[off::2]=fine[off];cm[off::2]=coarse[off]
   assert torch.equal(cm,x['maps']);assert fm.shape[-2]>cm.shape[-2]and fm.shape[1]==1024;assert torch.isfinite(fm).all()
   assert len(pixel_masks)==2
   fine_valid=torch.empty(n,*fm.shape[-2:],dtype=torch.bool)
   for off in [0,1]:fine_valid[off::2]=~F.interpolate(pixel_masks[off][:,None].float(),size=fm.shape[-2:],mode='nearest')[:,0].bool()
   # All current inputs have a rectangular resized image; verify mask edge
   # validity against the actual backbone mask, not a candidate/GT visibility rule.
   assert fine_valid.shape==(n,*fm.shape[-2:])
   positions=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];boxes=[torch.tensor(f['boxes'][k])for f,k in zip(x['frames'],x['selected'])if k>=0]
   features.append(save(D/'features'/(a+'.pt'),dict(alias=a,dataset=ds,maps=fm,valid=fine_valid,frame_ids=x['frame_ids'],E=x['E'],positions=positions,boxes=boxes,planned_positions=x['planned_positions'],interval=x['interval'],model_hash=mh,Source_exact=True,coarse_capture_exact=True,new_DINO_calls=0,GT_read=False)))
   del base,native,expert,fine,coarse,cm;gc.collect();torch.cuda.empty_cache()
   gpu_start=time.perf_counter();maps=fm.cuda();v=fine_valid.cuda();dsmap,dsv=downsample(maps,v,x['maps'].shape[-2:]);softs={'Coarse':fb['cosine_all_soft_maps']}
   softs['Fine_layer3']=propagate_chunked(maps,v,positions,boxes);softs['Fine_downsampled']=propagate_chunked(dsmap,dsv,positions,boxes);torch.cuda.synchronize();match_seconds=time.perf_counter()-gpu_start
   valid={'Coarse':x['valid'],'Fine_layer3':fine_valid,'Fine_downsampled':dsv.cpu()};E=x['E'].double();eps=fb['noise'];outputs={'E_MBR':E.numpy()};reads={}
   for layer in LAYERS:
    rd=direct(E,softs[layer],valid[layer],eps,p['configuration'],x['planned_positions']);assert torch.equal(rd['candidates'],old['direct']['candidates']);reads[layer]=rd;outputs[layer]=rd['boxes'].numpy()
   assert np.array_equal(outputs['Coarse'],old['outputs']['Cosine_readout']);assert torch.equal(reads['Coarse']['feedback']['rewards'],old['direct']['feedback']['rewards'])
   assert state_hash(model.state_dict())==mh
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],active=bool(positions),outputs=outputs,readouts=reads,soft_maps=softs,valid=valid,permutation=fb['permutation'],
    shapes={k:list(t.shape)for k,t in [('Coarse',x['maps']),('Fine_layer3',fm),('Fine_downsampled',dsmap)]},
    compute=dict(new_Source_captures=1,new_DINO_calls=0,capture_seconds=cap_seconds,matching_seconds=match_seconds,case_seconds=time.perf_counter()-at),GT_read=False,pretrained_weights_frozen=True)))
   cost.append(dict(alias=a,seconds=time.perf_counter()-at,matching_seconds=match_seconds));print(a,ix+1,round(time.perf_counter()-tick,3),flush=True)
   if ix==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-tick,rough32_seconds=16*(time.perf_counter()-tick),not_a_guarantee=True))
   del fm,maps,v,dsmap,dsv,softs,x,fb,old,reads,oldsource;gc.collect();torch.cuda.empty_cache()
  verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,feature_files=features,all32_before_NEW_GT=True,GPU_allocation_seconds=time.perf_counter()-tick+3,initialization_allowance_seconds=3,actual_worker_GPU_seconds=time.perf_counter()-tick,costs=cost,new_Source_captures=32,new_DINO_calls=0,peak_allocated_bytes=torch.cuda.max_memory_allocated()));status('all32_four_outputs_and_feedback_sealed',32,GT_read=False)
 finally:
  lease.close();put(D/'GPU_STOP.json',dict(worker_elapsed_seconds=time.perf_counter()-tick,all_partials_preserved=True))


def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_fine_feature32_v1 import LAYERS
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import xy,iou,summarize_frame
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];diag=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];truth,span=gt[ds][0][q],gt[ds][1][q];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,row,truth,span);ids=np.array(o['frame_ids']);known=np.array([int(f)in truth for f in ids]);unobs=~np.isin(np.arange(len(ids)),o['planned_positions']);mask=known&unobs;rr={};private={}
  for layer in LAYERS:
   rd=o['readouts'][layer];bb=rd['candidates'].numpy();Q=np.full(bb.shape[:2],np.nan);xx=xy(bb,row,ds);frame=[]
   for t in np.flatnonzero(known):Q[t]=iou(xx[t],np.asarray(truth[int(ids[t])]))
   raw=rd['feedback']['rewards'].numpy();prob=rd['feedback']['prob'].numpy();valid=rd['feedback']['valid'].numpy().copy();valid[:,0]=True
   for t in np.flatnonzero(mask):
    z=summarize_frame(Q[t],raw[t],prob[t],valid[t],Q[t,0],o['active'],Q[t,0],Q[t,0])
    for key in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']:z.pop(key)
    z.update(position=int(t));frame.append(z)
   rr[layer]=dict(frame_means={k:mean([f[k]for f in frame])for k in frame[0]if k!='position'}if frame else {},frames=frame,
    weighted_quality_minus_E_pp=mean([100*float((prob[t]*Q[t]).sum()-Q[t,0])for t in np.flatnonzero(mask)]),
    selected_better_than_E_fraction=mean([float(Q[t,int(rd['best'][t])]>Q[t,0]+1e-10)for t in np.flatnonzero(mask)])if o['active']else None)
   private[layer]=dict(quality=Q,known_unobserved_mask=mask)
  diag.append(save(D/'diagnostics'/(a+'.pt'),dict(alias=a,dataset=ds,layers=private,GT_offline_only=True)))
  rows.append(dict(alias=a,dataset=ds,active=o['active'],known_unobserved=int(mask.sum()),unknown_unobserved=int((~known&unobs).sum()),shapes=o['shapes'],ranking=rr,compute=o['compute'],**m))
 arms=['E_MBR']+LAYERS;groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];metrics_names=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']
  groups[ds]=dict(count=16,active=sum(r['active']for r in rr),metrics={arm:{k:mean([r['values'][arm][k]for r in rr])for k in metrics_names}for arm in arms},
   paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ['v','unobserved_actual_s']}for a,b in [(x,'E_MBR')for x in LAYERS]+[('Fine_layer3','Coarse'),('Fine_layer3','Fine_downsampled')]},
   ranking={layer:{key:paired([r['ranking'][layer]['frame_means'].get(key)for r in rr])for key in rr[0]['ranking'][layer]['frame_means']}for layer in LAYERS})
 passed=all(groups[ds]['paired']['Fine_layer3_minus_E_MBR']['v']['mean']>0 and groups[ds]['paired']['Fine_layer3_minus_E_MBR']['unobserved_actual_s']['mean']>0 and groups[ds]['paired']['Fine_layer3_minus_Coarse']['v']['mean']>0 for ds in groups)
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_after_full_seal=True,development_exposed=True,independent_confirmation=False,no_training=True));put(P/'GATE.json',dict(status='evaluated',passed=bool(passed),rule=p['gate'],OPD_not_yet_run=True,next='matched actual OPD comparison eligible after review'if passed else'stop this true-fine-layer cosine interface; do not sweep layers/temperature'))
 put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32,gate=bool(passed),diagnostic_files=diag));print('GATE',passed,{ds:{a:g['metrics'][a]['v']for a in arms}for ds,g in groups.items()},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise

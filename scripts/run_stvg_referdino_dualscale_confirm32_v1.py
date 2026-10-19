"""Only the frozen development-selected policy, on the previously reserved32.

No confirmation GT may select settings. Source/native teacher acquisition is
counted once per parent, shared across all controls, with the same eight frames.
"""
from pathlib import Path
import os,sys,time,gc,collections,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_referdino_dualscale_hc128_v1 import D as DEV,P as DP,verify as devverify
D=DEV/'confirmation';P=DP/'confirmation';VER=None

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256'];pins=dict(l['pins'])
 for f in sorted((D/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 p0=devverify(True);assert read(DP/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';sel=read(DEV/'SELECTION_LOCK.json');assert not D.exists();D.mkdir();P.mkdir()
 frozen_results=DEV/'DEVELOPMENT_SELECTION_INPUT.json';assert not frozen_results.exists();shutil.copyfile(DP/'RESULTS_PUBLIC.json',frozen_results);assert sha(frozen_results)==sel['all_results_SHA256']
 rs=p0['confirmation_reserved'];assert len(rs)==32 and not{r['parent']for r in rs}&{r['parent']for r in p0['records']}
 cfg=dict(lr=.01,steps=40,samples=32,tau=.05,sigma=.025);plan=dict(status='frozen_before_confirmation_predictions_GT',records=rs,count=32,seeds=[41,42,43],selected_policy=sel['selected_policy'],selected_scales=sel['selected_scales'],configuration=cfg,selection_sha256=sha(DEV/'SELECTION_LOCK.json'),development_results_sha256=sha(DP/'RESULTS_PUBLIC.json'),historical_exposure='reserved before128/never used these interface-policy rounds; historical project Source/full method exposure remains',GPU_seconds_cap=1200,stage_GPU_cap=900,deadline_epoch=time.time()+7200,created_epoch=time.time(),no_confirmation_based_selection=True,same8_positions=True,new_Source_queries_cap=32,new_ReferDINO_queries_cap=32,new_experts_or_model_training=0)
 pins=dict(read(DEV/'RUNTIME_LOCK.json')['pins'])
 for f in sorted((DEV/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 files=[str(frozen_results.relative_to(ROOT)),'scripts/evaluate_stvg_referdino_dualscale_source_baseline_v1.py','protocols/stvg_referdino_dualscale_source_baseline_v1.md',str((DEV/'SELECTION_LOCK.json').relative_to(ROOT))]+[str(f.relative_to(ROOT))for f in ROOT.joinpath('scripts').glob('*stvg_referdino_dualscale_confirm32_v1.py')]+[r['prior_input']for r in rs]
 pins.update({f:sha(ROOT/f)for f in files});put(D/'PLAN_LOCK.json',plan);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(P/'CONFIG_PUBLIC.json',{k:v for k,v in plan.items()if k!='records'});verify(True);print('CONFIRM LOCKED',plan['selected_policy'],flush=True)

def begin(stage):
 import torch
 p=verify(True);used=sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json'));start=time.perf_counter()
 def check():
  verify();assert time.time()<p['deadline_epoch'];assert used+time.perf_counter()-start+3<p['GPU_seconds_cap'];assert time.perf_counter()-start+3<p['stage_GPU_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 return p,start,check

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture as fresh
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 p,start,check=begin('capture');lease=gpu();files=[]
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict());cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']['vidstg']
  for r in p['records']:
   check();f=D/'inputs'/(r['alias']+'.pt');assert not f.exists();row,recs,base,native,ex,inp,seconds=fresh(model,'hc2',r['query_ordinal'],collections.OrderedDict());src=native['boxes'].cpu();pool=[]
   for a in ex['anchors']['single4']:
    pos=int(a['position']);probe=ex['observations'][('original',pos)]['probe'];b=np.asarray(probe['boxes'],dtype=np.float64).reshape(-1,4);scores=np.asarray(probe['target_scores'],dtype=np.float64);assert probe['accepted']and 0<len(b)<=3;pool.append(dict(position=pos,boxes=b,scores=scores))
   pools=[encode(v['boxes'])-encode(src.double().numpy())[v['position']]for v in pool];est=estimate(pools,[v['scores']for v in pool],cal['scale'],cal['truncation'],1/16);E=apply(src.double().numpy(),est['delta']);U,q=mass_table(probabilities(recs,base.zero['logits']));assert U==row['frame_ids'];iv=final_mbr(U,q)['interval']
   files.append(save(f,dict(alias=r['alias'],dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=U,Source=src,E=E,interval=iv,planned_positions=r['positions8'],original_positions4=r['positions4'],spatial_inputs=detached(base.spatial_inputs,'cpu'),records=detached(recs,'cpu'),source_seconds=seconds,original_input=inp,GT_read=False)));base.restore(base.initial);del base,row,recs,native,ex;gc.collect();torch.cuda.empty_cache();print('capture',r['alias'],flush=True)
  assert state_hash(model.state_dict())==mh;put(D/'CAPTURE_BARRIER.json',dict(count=32,files=files,Source_queries=32,Source_offset_forwards=64,GT_read=False))
 finally:put(D/'cost'/f'capture_{time.time_ns()}.json',dict(stage='capture',GPU_seconds=time.perf_counter()-start+3,files=len(files)));lease.close()

def teacher():
 import torch,numpy as np
 from scripts.run_final_simplification_v1 import lease as acquire
 from scripts.decota_paper_common_v1 import guard
 from scripts.run_decota_paper_main_v1 import read_row,frames_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_referdino_native32_v1 import NativeRefer
 p,start,check=begin('teacher');assert read(D/'CAPTURE_BARRIER.json')['count']==32;sys.addaudithook(guard);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';lease=acquire();files=[]
 try:
  model=NativeRefer();mh=state_hash(model.model.state_dict())
  for r in p['records']:
   check();f=D/'teacher0'/(r['alias']+'.pt');assert not f.exists();row=read_row('hc2',r['query_ordinal']);frames,ids,_=frames_for('hc2',row,collections.OrderedDict());assert list(ids)==load(D/'inputs'/(r['alias']+'.pt'))['frame_ids'];rgb=frames[r['positions8']];out=model(rgb,row['input']['caption']);files.append(save(f,dict(alias=r['alias'],native=out,rgb_sha256=[hashlib.sha256(x.tobytes()).hexdigest()for x in rgb],GT_read=False)));del frames,rgb,out;gc.collect();torch.cuda.empty_cache();print('teacher',r['alias'],flush=True)
  assert state_hash(model.model.state_dict())==mh;put(D/'TEACHER_BARRIER.json',dict(count=32,files=files,ReferDINO_query_forwards=32,image_frames=256,GT_read=False))
 finally:put(D/'cost'/f'teacher_{time.time_ns()}.json',dict(stage='teacher',GPU_seconds=time.perf_counter()-start+3,files=len(files)));lease.close()

def measure(chunk):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_referdino_native_opd_hc128_v1 import fit as oldfit
 from vg_tta.stvg_referdino_dualscale_hc128_v1 import fit as newfit,Replay
 p,start,check=begin('measure');assert 0<=chunk<2 and read(D/'TEACHER_BARRIER.json')['count']==32;lease=gpu();files=[]
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict())
  for r in p['records'][chunk*16:(chunk+1)*16]:
   check();a=r['alias'];dest=D/'predictions'/(a+'.pt');assert not dest.exists();tick=time.perf_counter();inp=load(D/'inputs'/(a+'.pt'));target=load(D/'teacher0'/(a+'.pt'))['native']['boxes'];rp=Replay(model,inp,inp['Source'],inp['E'],r['positions8'],check);fits={};outputs=dict(E_MBR=inp['E'],Source_MBR=inp['Source'].double().numpy());cfg=p['configuration']
   direct=oldfit(rp,target,a,'Direct',None,cfg);fits['Direct']=direct;outputs['Direct']=direct['final'].numpy()
   for seed in p['seeds']:
    if p['selected_policy']=='Old':
     f=oldfit(rp,target,a,'Fixed',seed,cfg);l=oldfit(rp,target,a,'Live',seed,cfg)
    else:
     old=oldfit(rp,target,a,'Live',seed,cfg);fits[f'Old_Live_{seed}']=old;outputs[f'Old_Live_{seed}']=old['final'].numpy();cc=dict(cfg,scales=p['selected_scales']);f=newfit(rp,target,a,'Fixed',seed,cc);l=newfit(rp,target,a,'Live',seed,cc)
    assert torch.equal(f['path'][1]['state'],l['path'][1]['state'])
    for key in ['actions','weights','cost','eps']:assert torch.equal(f['rounds'][0]['rollout'][key],l['rounds'][0]['rollout'][key])
    for name,out in [('Selected_Fixed',f),('Selected_Live',l)]:fits[f'{name}_{seed}']=out;outputs[f'{name}_{seed}']=out['final'].numpy()
   assert all(not x.requires_grad and x.grad is None for x in model.parameters());assert state_hash(rp.decoder.state_dict())==state_hash(model.ground_decoder.decoder.state_dict());cost=dict(case_seconds=time.perf_counter()-tick,decoder_seconds=rp.seconds,trajectory_decoder_calls=rp.calls,offset_decoder_calls=2*rp.calls,actual_updates=rp.backwards,new_Source=0,new_teacher=0)
   files.append(save(dest,dict(alias=a,dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=inp['frame_ids'],interval=inp['interval'],planned_positions=r['positions8'],outputs=outputs,fits=fits,GT_read=False,compute=cost)));del rp,inp,target,direct,f,l,fits,outputs;gc.collect();torch.cuda.empty_cache();print('confirm sealed',a,flush=True)
  assert state_hash(model.state_dict())==mh;verify(True);put(D/f'CHUNK_{chunk}_BARRIER.json',dict(count=16,files=files,GT_read=False))
  if all((D/f'CHUNK_{i}_BARRIER.json').exists()for i in range(2)):
   fs=[x for i in range(2)for x in read(D/f'CHUNK_{i}_BARRIER.json')['files']];put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=32,files=fs,all_outputs_before_confirmation_GT=True,selection_lock_unchanged=sha(DEV/'SELECTION_LOCK.json')==p['selection_sha256']))
 finally:put(D/'cost'/f'measure{chunk}_{time.time_ns()}.json',dict(stage='measure',chunk=chunk,GPU_seconds=time.perf_counter()-start+3,files=len(files)));lease.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','teacher','measure']);ap.add_argument('--chunk',type=int);a=ap.parse_args()
 try:measure(a.chunk)if a.stage=='measure'else globals()[a.stage]()
 except BaseException as e:
  D.mkdir(parents=True,exist_ok=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise

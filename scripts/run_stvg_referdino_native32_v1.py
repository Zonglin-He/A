"""One native ReferDINO8 cost-gated study. No GT until gate passes and32 seal."""
import os,sys,time,json,gc,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_referdino_native32_v1 import positions8,interpolate,RT,EXT
D=ROOT/'artifacts/stvg_referdino_native32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/referdino_native32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1'
VER=None

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 if VER is None:VER=PinVerifier(l['pins'])
 VER.verify(full=full);resource_guard();return read(D/'PLAN_LOCK.json')

def prepare():
 import numpy as np
 assert not D.exists();D.mkdir();P.mkdir(parents=True);rs=read(PRE/'PLAN_LOCK.json')['records'];pilots=['NH001','NH002','NV001','NV002'];out=[];own=['scripts/run_stvg_referdino_native32_v1.py','scripts/setup_stvg_referdino_runtime32_v1.py','vg_tta/stvg_referdino_native32_v1.py','protocols/stvg_referdino_native32_v1.md','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','checkpoints/referdino/ryt_mevis_swinb.pth','checkpoints/groundingdino-b/groundingdino_swinb_cogcoor.pth','artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json','vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py','methods/decota_final_simplified_v1/observations.py','scripts/run_decota_paper_main_v1.py']
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
 own += [str((ROOT/EXPERT_SNAPSHOT/'model.safetensors').relative_to(ROOT))]
 for r in rs:
  e=load(PRE/'E'/(r['alias']+'.pt'));s=load(PRE/'source'/(r['alias']+'.pt'));sel=positions8(e['frame_ids'],e['planned_positions']);assert len(sel)==8 and set(e['planned_positions'])<=set(sel)
  if r['alias']in pilots:assert len(s['expert']['observations'])==4
  out.append(dict(**r,positions8=sel,physical_frame_ids8=[e['frame_ids'][i]for i in sel],positions4=e['planned_positions']))
  own += [str((PRE/sub/(r['alias']+'.pt')).relative_to(ROOT))for sub in ['source','E']]+[f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/subjects/{r["query_ordinal"]:05}.json']
 own += [str(f.relative_to(ROOT))for f in EXT.rglob('*.py')]+[str((EXT/'configs/mevis_swinb.yaml').relative_to(ROOT))]+[str(f.relative_to(ROOT))for f in (RT/'bert-from-frozen-teacher').iterdir()if f.is_file()]+[str((RT/'SETUP_RECEIPT.json').relative_to(ROOT))]+[str(Path(read(RT/'SETUP_RECEIPT.json')['extension']).relative_to(ROOT))]
 cfg=dict(teachers=['original cached DINO-T baseline','single frozen ReferDINO ryt_mevis_swinb'],count=32,per_domain=16,pilots=pilots,frames=8,keep_original4=True,selection='native sigmoid(logits).mean(time).max(class).argmax(query), common track',full_original_caption=True,what_skip=False,AMP=False,short_side=360,crop_len=36,tracking_alpha=.1,cost_ratio_limit=1.5,cost_gate='sum4 steady expert times ratio, one same-input warmup each model separately counted',GPU_seconds_cap=900,new_training_OPD=0,time='original E finalMBR unchanged',arms=['E_MBR','Ref_E_MBR','Ref_direct8_MBR'],GT_only_after_gate_and_all32_seal=True,source_calibration_unchanged=True,no_new_confidence_threshold=True,E_references='same native singleton selected track at original4 only',direct='linear physical time xyxy, endpoint hold',development_exposed=True,formerly_skipped=['NV003','NV006','NV009','NV013','NV015'],continued_quality_target='both domain full-v means positive and at least one near5pp, uncertainty/concentration checked; no automatic OPD',no_automatic_followup=True)
 put(D/'PLAN_LOCK.json',dict(status='locked_before_new_inference_GT',records=out,configuration=cfg,created_epoch=time.time(),deadline_epoch=time.time()+7200));put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(own))}));put(P/'CONFIG_PUBLIC.json',dict(**cfg,positions=[{k:r[k]for k in ['alias','dataset','positions4','positions8','physical_frame_ids8']}for r in out]));put(P/'MODEL_PROVENANCE.json',dict(weight=read(ROOT/'checkpoints/referdino/DOWNLOAD_RECEIPT.json'),code=read(ROOT/'checkpoints/referdino/CODE_DOWNLOAD_RECEIPT.json'),runtime=read(RT/'SETUP_RECEIPT.json'),T_sha256=EXPERT_SHA256,native_network_and_selection_unchanged=True));put(D/'STATUS.json',dict(status='prepared',done=0,total=32,GT_read=False),True);print('LOCKED32/pilot4 noGT',flush=True)

def private_output(r,native):
 import numpy as np
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 a=r['alias'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));src=s['Source'].double().numpy();boxes=native['boxes'].numpy().astype(float);idx=[r['positions8'].index(i)for i in e['planned_positions']];setting=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2'];res=encode(boxes[idx])-encode(src)[e['planned_positions']];es=estimate([v[None]for v in res],[np.ones(1)for _ in res],setting['scale'],setting['truncation'],1/16);ref=apply(src,es['delta']);direct=interpolate(e['frame_ids'],r['positions8'],boxes)
 return dict(alias=a,dataset=r['dataset'],query_ordinal=r['query_ordinal'],frame_ids=e['frame_ids'],planned_positions=e['planned_positions'],positions8=r['positions8'],interval=e['interval'],native=native,estimate=es,outputs=dict(E_MBR=e['E'],Ref_E_MBR=ref,Ref_direct8_MBR=direct),GT_read=False)

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import read_row,frames_for
 from scripts.run_final_simplification_v1 import lease as acquire
 from scripts.decota_paper_common_v1 import guard
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_referdino_native32_v1 import NativeRefer
 p=verify(True);sys.addaudithook(guard);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';lock=acquire();begin=time.perf_counter();cache=collections.OrderedDict();pilot=[next(r for r in p['records']if r['alias']==a)for a in p['configuration']['pilots']];timing={};allocations={};warm={};pilot_rc=[]
 def check():
  verify();assert time.perf_counter()-begin+3<900 and time.time()<p['deadline_epoch'];assert torch.cuda.mem_get_info()[0]>4*2**30
 def inputs(r):
  row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);e=load(PRE/'E'/(r['alias']+'.pt'));assert ids==e['frame_ids'];return row,frames,e
 try:
  check();lt=time.perf_counter();dino=SpatialExpert(ROOT/EXPERT_SNAPSHOT);allocations['T_load_seconds']=time.perf_counter()-lt;mh=state_hash(dino.model.state_dict())
  for j,r in enumerate(pilot):
   check();row,frames,e=inputs(r);s=load(PRE/'source'/(r['alias']+'.pt'));prior=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False)
   if j==0:
    torch.cuda.synchronize();t=time.perf_counter();ex=observations(dino,row['parses'],frames,e['frame_ids'],prior['indices'],audit=True);torch.cuda.synchronize();warm['T']=dict(expert_seconds=time.perf_counter()-t,calls=ex['new_DINO'])
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();ex=observations(dino,row['parses'],frames,e['frame_ids'],prior['indices'],audit=True);torch.cuda.synchronize();elapsed=time.perf_counter()-t;assert ex['new_DINO']==4 and ex['positions4']==e['planned_positions']
   for key,v in ex['observations'].items():assert v['receipt']['inputs']==s['expert']['observations'][key]['receipt']['inputs']
   timing[r['alias']]=dict(alias=r['alias'],dataset=r['dataset'],T_expert_seconds=elapsed,T_forward_seconds=sum(v['receipt']['forward_seconds']for v in ex['observations'].values()),T_peak_allocated_bytes=torch.cuda.max_memory_allocated(),T_peak_reserved_bytes=torch.cuda.max_memory_reserved(),T_calls=4);print('T timed',r['alias'],elapsed,flush=True);cache.clear();del frames,ex
  assert state_hash(dino.model.state_dict())==mh;del dino;gc.collect();torch.cuda.empty_cache();check();lt=time.perf_counter();native=NativeRefer();allocations['Refer_load_seconds']=time.perf_counter()-lt;mh=state_hash(native.model.state_dict());put(D/'MODEL_LOAD_AUDIT.json',native.load_audit)
  # CUDA kernel equivalence uses synthetic tensors, no model feedback/GT.
  import MultiScaleDeformableAttention as C
  from models.GroundingDINO.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
  v=torch.randn(1,20,2,4,device='cuda');ss=torch.tensor([[4,4],[2,2]],device='cuda');st=torch.tensor([0,16],device='cuda');loc=torch.rand(1,5,2,2,4,2,device='cuda');w=torch.rand(1,5,2,2,4,device='cuda');w=w/w.sum((-1,-2),keepdim=True);err=(C.ms_deform_attn_forward(v,ss,st,loc,w,64)-ms_deform_attn_core_pytorch(v,ss,loc,w)).abs().max().item();assert err<2e-6;put(D/'CUDA_KERNEL_CHECK.json',dict(status='pass',max_abs=err,synthetic_only=True));del v,ss,st,loc,w
  for j,r in enumerate(pilot):
   check();row,frames,e=inputs(r);rgb=frames[r['positions8']];caption=row['input']['caption']
   if j==0:warm['Refer']=native(rgb,caption)['cost']
   torch.cuda.reset_peak_memory_stats();z=native(rgb,caption);z['cost'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());z['rgb_sha256']=[hashlib.sha256(f.tobytes()).hexdigest()for f in rgb];timing[r['alias']].update(Refer_expert_seconds=z['cost']['expert_seconds'],Refer_cost=z['cost'],ratio=z['cost']['expert_seconds']/timing[r['alias']]['T_expert_seconds']);pilot_rc.append(save(D/'predictions'/(r['alias']+'.pt'),private_output(r,z)));print('Refer timed',r['alias'],z['cost']['expert_seconds'],'ratio',timing[r['alias']]['ratio'],flush=True);cache.clear();del frames,z,rgb;gc.collect()
  ratio=sum(v['Refer_expert_seconds']for v in timing.values())/sum(v['T_expert_seconds']for v in timing.values());passed=ratio<=1.5;gate=dict(status='pass'if passed else'failed_cost_gate',ratio=ratio,limit=1.5,rows=list(timing.values()),warmups=warm,loading=allocations,GT_read=False,parameters=native.params,load_audit=native.load_audit,kernel_error=err,hardware=torch.cuda.get_device_name(),GPU_allocation_at_gate_seconds=time.perf_counter()-begin+3,counts=dict(T_measured_forwards=16,T_warmup_forwards=4,Refer_measured_queries=4,Refer_warmup_queries=1),full32_allowed=passed)
  put(P/'COST_GATE_PUBLIC.json',gate);put(D/'COST_GATE.json',gate)
  if not passed:
   assert state_hash(native.model.state_dict())==mh;verify(True);put(D/'COST_STOP_RECEIPT.json',dict(status='stopped_at_cost_gate',GPU_allocation_seconds=time.perf_counter()-begin+3,completed_pilot_outputs=4,remaining_not_run=28,GT_read=False,weights_unchanged=True,no_quality_claim=True));put(D/'STATUS.json',dict(status='cost_gate_failed_stopped_without_GT',done=4,total=32,remaining_not_run=28,GT_read=False),True);print('COST GATE FAILED',ratio,'STOP no GT',flush=True);return
  files=list(pilot_rc)
  for r in p['records']:
   if r['alias']in p['configuration']['pilots']:continue
   check();row,frames,e=inputs(r);torch.cuda.reset_peak_memory_stats();rgb=frames[r['positions8']];z=native(rgb,row['input']['caption']);z['cost'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());z['rgb_sha256']=[hashlib.sha256(f.tobytes()).hexdigest()for f in rgb];files.append(save(D/'predictions'/(r['alias']+'.pt'),private_output(r,z)));put(D/'STATUS.json',dict(status='running_after_gate_pass',done=len(files),total=32,GT_read=False),True);print('sealed',r['alias'],len(files),flush=True);cache.clear();del frames,rgb,z;gc.collect()
  assert state_hash(native.model.state_dict())==mh;verify(True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,GPU_allocation_seconds=time.perf_counter()-begin+3,Refer_queries=32,Refer_warmup_queries=1,T_forwards=20,weights_unchanged=True));put(D/'STATUS.json',dict(status='sealed_before_scoring',done=32,total=32,GT_read=False),True)
 finally:lock.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e)));raise

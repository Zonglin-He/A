"""Finite staged conditional ReferDINO feedback probe; global no-GT barrier."""
import os,sys,time,gc,collections,hashlib,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_referdino_condition_hc128_v1';P=ROOT/'results/stvg_error_state/2026-10-19/referdino_condition_hc128';OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2';VER=None
CODE=['vg_tta/stvg_referdino_condition_hc128_v1.py','scripts/run_stvg_referdino_condition_hc128_v1.py','protocols/stvg_referdino_condition_hc128_v1.md']
def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 pins=dict(l['pins'])
 for f in sorted((D/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return read(D/'PLAN_LOCK.json')
def prepare():
 import torch
 from vg_tta.stvg_referdino_native32_v1 import positions8
 assert not D.exists();D.mkdir();P.mkdir(parents=True)
 rows=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];nh=read(ROOT/'artifacts/stvg_referdino_native32_v1/PLAN_LOCK.json')['records'];excluded={r['parent']for r in nh if r['dataset']=='hc2'};groups=collections.defaultdict(list)
 for r in rows:
  if r['source']not in excluded:groups[r['source']].append(r)
 rng=random.Random(2026101901);parents=sorted(groups);rng.shuffle(parents);confirm=parents[:32];dev=parents[32:160];assert len(dev)==128
 def record(parent,i,prefix):
  row=rng.choice(groups[parent]);q=row['ordinal'];f=OLD/f'inputs/vidstg_to_hc2/clean/{q:05}.pt';old=torch.load(f,map_location='cpu',weights_only=False);assert not old['GT_read'];ex=old['expert'];pos=ex['positions4'];sel=positions8(old['frame_ids'],pos)
  return dict(alias=f'{prefix}{i+1:03}',dataset='hc2',parent=parent,query_ordinal=q,positions4=list(pos),positions8=sel,physical_frame_ids8=[old['frame_ids'][j]for j in sel],prior_input=str(f.relative_to(ROOT)),prior_sha256=sha(f))
 rs=[record(s,i,'RC')for i,s in enumerate(dev)];held=[record(s,i,'RH')for i,s in enumerate(confirm)]
 pilot=next(r for r in nh if r['alias']=='NH001');pilot=dict(**pilot,engineering_only=True)
 cfg=dict(lr=.01,steps=40,optimizer='Adam .9/.999 eps1e-8 wd0',loss='mean anchors 5L1+2(1-GIoU)',state='8 independent 256D content vectors before temporal self-attention, original two offsets',reference_slot='native winning track initial proposal ancestor per frame, locked before conditioning',reference_projection='cxcywh individually clamp1e-5..1-1e-5 for native sigmoid initial reference; count every changed element',teacher='single frozen ryt_mevis_swinb.pth',frames=8,student_time='original E finalMBR',GPU_seconds_cap=1800,per_stage_cap=900,full_OPD=False)
 plan=dict(records=rs,confirmation_reserved=held,engineering=pilot,configuration=cfg,count=128,created_epoch=time.time(),deadline_epoch=time.time()+10800,selection_seed=2026101901,selection='parent uniform shuffle then uniform query, excludes NH16 parents, hold32 before128',historical_exposure='official HC dev split has old Source/full-method scoring; reserved32 excluded current interface design/selection, NOT globally never exposed',GT_read=False)
 put(D/'PLAN_LOCK.json',plan)
 pins={**read(ROOT/'artifacts/stvg_referdino_native32_v1/RUNTIME_LOCK.json')['pins'],**read(ROOT/'artifacts/stvg_hc_anchor_content16_v1/RUNTIME_LOCK.json')['pins']};pins.update({f:sha(ROOT/f)for f in CODE});put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins))
 put(P/'CONFIG_PUBLIC.json',dict(count=128,reserved_parent_count=32,configuration=cfg,selection_seed=2026101901,selection=plan['selection'],historical_exposure=plan['historical_exposure'],NH16_engineering_only=True,all_outputs_sealed_before_new_GT=True,no_new_experts_or_trained_models=True,teacher_inputs='same8 full RGB/text for native/T(S0)/T(S1), no extra observation positions',student_states='S0=E; S1=40-step update to T(S0); from same S1 another40 old vs refreshed feedback; fresh identical Adam for both',no_automatic_full_OPD_or_confirmation=True))
 verify(True);print('LOCKED128 +32 held parents',flush=True)
def begin(stage):
 import torch
 p=verify(True);assert time.time()<p['deadline_epoch'];used=sum(x['GPU_seconds']for x in [read(f)for f in (D/'cost').glob('*.json')]);assert used<p['configuration']['GPU_seconds_cap'];start=time.perf_counter()
 def check():
  verify();assert time.time()<p['deadline_epoch'];assert used+time.perf_counter()-start+3<p['configuration']['GPU_seconds_cap'];assert time.perf_counter()-start+3<p['configuration']['per_stage_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 return p,start,check

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture as fresh
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 def input_pool(expert):
  frames=[]
  for a in expert["anchors"]["single4"]:
   pos=int(a["position"]);probe=expert["observations"][("original",pos)]["probe"];b=np.asarray(probe["boxes"],dtype=np.float64).reshape(-1,4);scores=np.asarray(probe["target_scores"],dtype=np.float64);assert probe["accepted"] and 0<len(b)<=3;frames.append(dict(position=pos,boxes=b,scores=scores))
  return frames
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 p,start,check=begin('capture');lease=gpu();files=[]
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict());cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']['vidstg']
  for i,r in enumerate(p['records']):
   check();f=D/'inputs'/(r['alias']+'.pt')
   if f.exists():files.append(read(f.with_suffix('.receipt.json')));continue
   row,recs,base,native,ex,inp,seconds=fresh(model,'hc2',r['query_ordinal'],collections.OrderedDict());src=native['boxes'].cpu();pool=input_pool(ex);pools=[encode(v['boxes'])-encode(src.double().numpy())[v['position']]for v in pool];est=estimate(pools,[v['scores']for v in pool],cal['scale'],cal['truncation'],1/16);E=apply(src.double().numpy(),est['delta']);U,q=mass_table(probabilities(recs,base.zero['logits']));assert U==row['frame_ids'];iv=final_mbr(U,q)['interval']
   files.append(save(f,dict(alias=r['alias'],dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=U,Source=src,E=E,interval=iv,planned_positions=r['positions8'],original_positions4=r['positions4'],spatial_inputs=detached(base.spatial_inputs,'cpu'),records=detached(recs,'cpu'),source_seconds=seconds,original_input=inp,GT_read=False)))
   base.restore(base.initial);del base,row,recs,native,ex;gc.collect();torch.cuda.empty_cache();put(D/'STATUS.json',dict(stage='capture',done=i+1,total=128,GT_read=False),True);print('capture',i+1,flush=True)
  assert state_hash(model.state_dict())==mh;put(D/'CAPTURE_BARRIER.json',dict(count=128,files=files,weights_unchanged=True,GT_read=False))
 finally:put(D/'cost'/f'capture_{time.time_ns()}.json',dict(stage='capture',GPU_seconds=time.perf_counter()-start+3,files=len(files)));lease.close()

def teacher(round):
 import torch,numpy as np
 from scripts.run_final_simplification_v1 import lease as acquire
 from scripts.decota_paper_common_v1 import guard
 from scripts.run_decota_paper_main_v1 import read_row,frames_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_referdino_condition_hc128_v1 import ConditionalRefer
 p,start,check=begin('teacher'+str(round));sys.addaudithook(guard);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';lease=acquire();files=[]
 try:
  model=ConditionalRefer();mh=state_hash(model.model.state_dict())
  if round==0 and not(D/'ENGINEERING.json').exists():
   r=p['engineering'];row=read_row('hc2',r['query_ordinal']);fr,ids,_=frames_for('hc2',row,collections.OrderedDict());rgb=fr[r['positions8']];baseline_full={};native_infer=model.model.infer
   def raw_capture(*args,**kw):
    out=native_infer(*args,**kw);baseline_full.update({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()for k,v in out.items()if torch.is_tensor(v)});return out
   model.model.infer=raw_capture;raw=model(rgb,row['input']['caption']);model.model.infer=native_infer;noop=model.run(rgb,row['input']['caption']);assert torch.equal(raw['boxes'],noop['boxes'])and torch.equal(raw['native_query_scores'],noop['native_query_scores']);assert baseline_full['pred_masks']==noop['trace']['full_masks_hash'];assert baseline_full['pred_boxes']==hashlib.sha256(noop['trace']['all_boxes'][None].numpy().tobytes()).hexdigest()and baseline_full['pred_logits']==hashlib.sha256(noop['trace']['all_logits'][None].numpy().tobytes()).hexdigest();slots=noop['trace']['selected_slots_native'];b=noop['trace']['initial'][slots,torch.arange(8)].numpy();eq=model.run(rgb,row['input']['caption'],b,slots);err=float((eq['boxes']-noop['boxes']).abs().max());assert err<2e-5
   put(D/'ENGINEERING.json',dict(status='pass',engineering_only_alias='NH001',no_intervention_boxes_scores_and_full_logits_masks_bitexact=True,initial_reference_roundtrip_max_abs=err,decoder_query_counts=[len(x)for x in noop['trace']['pruning_lineages']],matching_calls=len(noop['trace']['matching_permutations']),GT_read=False));del fr,rgb,raw,noop,eq
  for i,r in enumerate(p['records']):
   check();a=r['alias'];f=D/f'teacher{round}'/(a+'.pt')
   if f.exists():files.append(read(f.with_suffix('.receipt.json')));continue
   z=load(D/'inputs'/(a+'.pt'));row=read_row('hc2',r['query_ordinal']);fr,ids,_=frames_for('hc2',row,collections.OrderedDict());assert ids==z['frame_ids'];rgb=fr[r['positions8']]
   if round==0:
    native=model.run(rgb,row['input']['caption']);slots=native['trace']['selected_slots_native'];bs=z['E'][r['positions8']]
   else:
    native=None;old=load(D/'teacher0'/(a+'.pt'));slots=old['slots'];s=load(D/'student1'/(a+'.pt'));bs=s['fit']['final'][r['positions8']].numpy()
   out=model.run(rgb,row['input']['caption'],bs,slots)
   if native is not None:assert out['trace']['native_tgt_hash']==native['trace']['native_tgt_hash']
   files.append(save(f,dict(alias=a,round=round,native=native,conditional=out,slots=slots,student_reference=bs,rgb_sha256=[hashlib.sha256(x.tobytes()).hexdigest()for x in rgb],GT_read=False)))
   put(D/'STATUS.json',dict(stage=f'teacher{round}',done=i+1,total=128,GT_read=False),True);print('teacher',round,i+1,flush=True);del fr,rgb,z,out,native;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.model.state_dict())==mh;put(D/f'TEACHER{round}_BARRIER.json',dict(count=128,files=files,weights_unchanged=True,GT_read=False))
 finally:put(D/'cost'/f'teacher{round}_{time.time_ns()}.json',dict(stage=f'teacher{round}',GPU_seconds=time.perf_counter()-start+3,files=len(files)));lease.close()

def student(round):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_hc_anchor_content16_v1 import Replay
 from vg_tta.stvg_referdino_condition_hc128_v1 import continue_fit
 p,start,check=begin('student'+str(round));lease=gpu();files=[]
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict())
  for i,r in enumerate(p['records']):
   check();a=r['alias'];f=D/f'student{round}'/(a+'.pt')
   if f.exists():files.append(read(f.with_suffix('.receipt.json')));continue
   z=load(D/'inputs'/(a+'.pt'));rp=Replay(model,z,z['Source'],z['E'],r['positions8'],check);old=load(D/'teacher0'/(a+'.pt'));cfg=p['configuration']
   if round==1:
    fit=continue_fit(rp,old['conditional']['boxes'],cfg,torch.zeros_like(rp.state));out=dict(alias=a,fit=fit,GT_read=False)
   else:
    s=load(D/'student1'/(a+'.pt'));new=load(D/'teacher1'/(a+'.pt'));state=s['fit']['final_state'];fixed=continue_fit(rp,old['conditional']['boxes'],cfg,state);refresh=continue_fit(rp,new['conditional']['boxes'],cfg,state);assert torch.equal(fixed['path'][0]['state'],refresh['path'][0]['state'])and torch.equal(fixed['path'][0]['boxes'],s['fit']['final'])
    out=dict(alias=a,dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=r['positions8'],positions8=r['positions8'],original_positions4=r['positions4'],outputs=dict(E_MBR=z['E'],S1=s['fit']['final'].numpy(),Old_feedback=fixed['final'].numpy(),Refreshed_feedback=refresh['final'].numpy()),fixed=fixed,refresh=refresh,GT_read=False)
   out['compute']=dict(decoder_calls=rp.calls,offset_decoder_calls=2*rp.calls,backwards=rp.backwards,decoder_seconds=rp.seconds);files.append(save(f,out));del rp,z,old,out;gc.collect();torch.cuda.empty_cache();put(D/'STATUS.json',dict(stage=f'student{round}',done=i+1,total=128,GT_read=False),True);print('student',round,i+1,flush=True)
  assert state_hash(model.state_dict())==mh;put(D/f'STUDENT{round}_BARRIER.json',dict(count=128,files=files,weights_unchanged=True,GT_read=False))
  if round==2:put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,all_outputs_before_new_GT=True,GT_read=False))
 finally:put(D/'cost'/f'student{round}_{time.time_ns()}.json',dict(stage=f'student{round}',GPU_seconds=time.perf_counter()-start+3,files=len(files)));lease.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','teacher0','student1','teacher1','student2']);args=ap.parse_args()
 try:
  if args.stage.startswith('teacher'):teacher(int(args.stage[-1]))
  elif args.stage.startswith('student'):student(int(args.stage[-1]))
  else:globals()[args.stage]()
 except BaseException as exc:
  D.mkdir(exist_ok=True,parents=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=args.stage,type=type(exc).__name__,error=str(exc),partials_preserved=True));raise

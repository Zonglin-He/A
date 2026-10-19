"""Native fixed-teacher Direct/Fresh-Fixed/Live HC128, three locked seeds."""
from pathlib import Path
import os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_referdino_native_opd_hc128_v1';P=ROOT/'results/stvg_error_state/2026-10-19/referdino_native_opd_hc128';PRE=ROOT/'artifacts/stvg_referdino_condition_hc128_v1';VER=None
SEEDS=[41,42,43]
def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256'];pins=dict(l['pins'])
 for f in sorted((D/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 import torch
 from vg_tta.stvg_referdino_native_opd_hc128_v1 import properties
 assert not D.exists();D.mkdir();P.mkdir(parents=True);old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete';rs=old['records'];assert len(rs)==128 and len({r['parent']for r in rs})==128
 cfg=dict(lr=.01,steps=40,samples=32,sigma=.025,tau=.05);files=[];manifest=[]
 for r in rs:
  a=r['alias'];f=PRE/'inputs'/(a+'.pt');t=PRE/'teacher0'/(a+'.pt');z=load(f);teacher=load(t);native=teacher['native'];assert not z['GT_read']and not teacher['GT_read'];assert native['boxes'].shape==(8,4)and len(r['positions8'])==8
  assert z['planned_positions']==r['positions8'];files.extend([str(f.relative_to(ROOT)),str(t.relative_to(ROOT))]);manifest.append(dict(alias=a,input_sha256=sha(f),teacher_cache_sha256=sha(t),target_boxes_sha256=__import__('hashlib').sha256(native['boxes'].numpy().tobytes()).hexdigest(),native_selected_query=native['selected_query']))
 own=['vg_tta/stvg_referdino_native_opd_hc128_v1.py']+[str(f.relative_to(ROOT))for f in ROOT.joinpath('scripts').glob('*stvg_referdino_native_opd_hc128_v1.py')]+['protocols/stvg_referdino_native_opd_hc128_v1.md']
 l=read(PRE/'RUNTIME_LOCK.json');pins=dict(l['pins'])
 for f in sorted((PRE/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 # Protect actual cached inputs/teacher snapshots and all reused current code.
 pins.update({f:sha(ROOT/f)for f in files+own});plan=dict(status='locked_before_new_updates_GT',records=rs,count=128,configuration=cfg,seeds=SEEDS,arms=['E','Direct','Fixed','Live'],teacher='exact cached NATIVE ReferDINO selected8 trajectory; conditional T(S0)/T(S1) prohibited',Fixed='initial native-logit mean, NEW paired samples and feedback every step',Live='current actual native-logit mean, same paired random stream each step',Direct='recompute native-label direct cost from zero; no reuse of previous conditional S1',student='eight independent256D content states, frozen decoder/weights/E+finalMBR/physical8 and same E-retaining map',feedback='mean eight 5L1+2(1-GIoU); weights softmax(-cost/.05), constructed geometric targets, not native Refer probabilities',policy='Gaussian native box output logits, not decoder latent/whole trajectory candidates',selection='same exposed HC128 locked parent/query roster; no NH16/held32 run/selection',Direct_runs=128,Fixed_runs=384,Live_runs=384,total_fits=896,total_configured_updates=35840,seed_aggregation='average metrics across3 seeds within each parent then bootstrap128 parents; also report every seed, no winner selection',GPU_seconds_cap=1800,chunk_GPU_cap=900,chunks=4,chunk_size=32,deadline_epoch=time.time()+10800,created_epoch=time.time(),new_Source_or_teacher_calls=0,no_new_experts_or_model_training=True,no_parameter_scan=True,no_automatic_followup=True)
 put(D/'PLAN_LOCK.json',plan);put(D/'INPUT_MANIFEST.json',dict(rows=manifest,teacher_source='native_only',GT_read=False));put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(P/'CPU_PROPERTIES.json',properties());put(P/'CONFIG_PUBLIC.json',{k:v for k,v in plan.items()if k!='records'});put(D/'STATUS.json',dict(status='prepared',done=0,total=128,GT_read=False),True);verify(True);print('LOCKED128 native Direct+3 paired seeds Fixed/Live, noGT',flush=True)

def measure(chunk):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_referdino_native_opd_hc128_v1 import Replay,fit
 p=verify(True);assert 0<=chunk<4;used=sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json'));lease=gpu();start=time.perf_counter();files=[];done=False
 def check():
  verify();assert time.time()<p['deadline_epoch'];assert used+time.perf_counter()-start+3<p['GPU_seconds_cap'];assert time.perf_counter()-start+3<p['chunk_GPU_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict());manifest={r['alias']:r for r in read(D/'INPUT_MANIFEST.json')['rows']}
  for i,r in enumerate(p['records'][chunk*32:(chunk+1)*32]):
   check();a=r['alias'];dest=D/'predictions'/(a+'.pt')
   if dest.exists():z=load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   tick=time.perf_counter();z=load(PRE/'inputs'/(a+'.pt'));teacher=load(PRE/'teacher0'/(a+'.pt'));target=teacher['native']['boxes'];assert __import__('hashlib').sha256(target.numpy().tobytes()).hexdigest()==manifest[a]['target_boxes_sha256'];rp=Replay(model,z,z['Source'],z['E'],r['positions8'],check)
   direct=fit(rp,target,a,'Direct',None,p['configuration']);fits=dict(Direct=direct);outputs=dict(E_MBR=z['E'],Direct=direct['final'].numpy());contracts=[]
   for seed in p['seeds']:
    fixed=fit(rp,target,a,'Fixed',seed,p['configuration']);live=fit(rp,target,a,'Live',seed,p['configuration'])
    for key in ['actions','weights','cost','eps']:assert torch.equal(fixed['rounds'][0]['rollout'][key],live['rounds'][0]['rollout'][key])
    assert torch.equal(fixed['path'][1]['state'],live['path'][1]['state'])
    for k,(f,l)in enumerate(zip(fixed['rounds'],live['rounds'])):
     assert torch.equal(f['rollout']['proposal_mean'],fixed['initial_policy_mean']);assert f['rollout']['rng_seed']==l['rollout']['rng_seed'];assert torch.equal(f['rollout']['eps'],l['rollout']['eps'])
     if k:assert f['rollout']['rng_seed']!=fixed['rounds'][k-1]['rollout']['rng_seed']and not torch.equal(f['rollout']['actions'],fixed['rounds'][k-1]['rollout']['actions'])
    for arm,out in [('Fixed',fixed),('Live',live)]:name=f'{arm}_{seed}';fits[name]=out;outputs[name]=out['final'].numpy()
    contracts.append(dict(seed=seed,first_samples_weights_update_exact=True,all40_paired_noises_exact=True,Fixed_initial_center=True,Fixed_all40_samples_fresh=True,Live_current_actual_center=True))
   assert all(not q.requires_grad and q.grad is None for q in model.parameters());assert state_hash(rp.decoder.state_dict())==state_hash(model.ground_decoder.decoder.state_dict())
   cost=dict(case_seconds=time.perf_counter()-tick,decoder_seconds=rp.seconds,trajectory_decoder_calls=rp.calls,offset_decoder_calls=2*rp.calls,actual_updates=rp.backwards,new_Source=0,new_teacher=0)
   files.append(save(dest,dict(alias=a,dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=r['positions8'],outputs=outputs,fits=fits,contracts=contracts,teacher_native_selected_query=teacher['native']['selected_query'],target_sha256=manifest[a]['target_boxes_sha256'],GT_read=False,compute=cost)))
   put(D/'STATUS.json',dict(status='running',chunk=chunk,done=chunk*32+i+1,total=128,GT_read=False),True);print('sealed',a,chunk*32+i+1,'seconds',round(cost['case_seconds'],3),'updates',cost['actual_updates'],flush=True);del rp,z,teacher,target,direct,fixed,live,fits,outputs;gc.collect();torch.cuda.empty_cache()
  check();assert state_hash(model.state_dict())==mh;verify(True);put(D/f'CHUNK_{chunk}_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,weights_unchanged=True));done=True
  if all((D/f'CHUNK_{c}_BARRIER.json').exists()for c in range(4)):
   allfiles=[f for c in range(4)for f in read(D/f'CHUNK_{c}_BARRIER.json')['files']];assert len(allfiles)==128;put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=128,logical_outputs=1024,adapted_fits=896,files=allfiles,all_outputs_before_new_GT=True,GT_read=False))
 finally:put(D/'cost'/f'chunk{chunk}_{time.time_ns()}.json',dict(chunk=chunk,GPU_seconds=time.perf_counter()-start+3,files=len(files),complete=done));lease.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure']);ap.add_argument('--chunk',type=int);a=ap.parse_args()
 try:prepare()if a.stage=='prepare'else measure(a.chunk)
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,chunk=a.chunk,type=type(e).__name__,error=str(e),partials_preserved=True));raise

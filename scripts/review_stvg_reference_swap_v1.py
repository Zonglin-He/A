"""Pre-GT source, matched-support, saved-math and full-readout audit of E1."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_reference_recovery_common_v1 import load_case,PARENT,OLD
from scripts.measure_stvg_reference_recovery_v1 import fit_readback
from scripts.measure_stvg_reference_swap_v1 import _verify_runtime,validate_common_support
BASE=ROOT/'artifacts/stvg_reference_recovery_v1/Reference_swap'

def main():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from scripts import stvg_target_teacher_student_v1 as core
 from scripts.run_stvg_table1_student004 import stable_fit_key
 _verify_runtime(BASE/'RUNTIME.json','teacher');_verify_runtime(BASE/'STUDENT_RUNTIME.json','student');sys.addaudithook(guard)
 roster={r.get('case_id',r.get('probe_alias')):r for r in read(BASE/'ROSTER_LOCK.json')['records']};refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};oldrefs={r['case_id']:r for r in read(PARENT/'REFERENCE_LOCK.json')['records']}
 bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');tb=read(BASE/'TEACHER_GLOBAL_BARRIER.json');assert bar['count']==32 and bar['status']=='complete' and bar['GT_read'] is False and tb['count']==49 and tb['GT_read'] is False
 expected={str((BASE/'student'/(cid+'.pt')).relative_to(ROOT))for cid in roster};assert {x['path']for x in bar['files']}==expected
 pins=[];fits=steps=reward_checks=common=cases_with_common=0;max_r=max_q=0.
 for item in tb['files']:
  tp=BASE/item['path']if item['path'].startswith('teacher/')else ROOT/item['path'];assert sha(tp)==item['sha256'];t=read(tp);assert t.get('GT_read')is False;pins.append(dict(item,path=str(tp.relative_to(ROOT))))
 parentitems={x['path']:x for x in read(PARENT/'STUDENT_GLOBAL_BARRIER.json')['files']}
 for item in bar['files']:
  path=ROOT/item['path'];assert sha(path)==item['sha256'];d=torch.load(path,map_location='cpu',weights_only=False);cid=d['case_id'];case,sp=load_case(roster[cid]);ref=refs[cid]
  assert d['reference_records']==ref and d['source_case_sha256']==case['source_case_sha256'] and d['source_file_sha256']==sha(sp) and d['source_model_state_sha256']==case['source_model_state_sha256'] and d['pixel_sha256']==case['pixel_sha256'] and d['source_restored'] and not d['GT_read']
  support=validate_common_support(ref,case['frame_ids'],oldrefs[cid]);common+=len(support);cases_with_common+=bool(support)
  pp=PARENT/'student'/(cid+'.pt');assert sha(pp)==parentitems[str(pp.relative_to(ROOT))]['sha256']==d['parent_student_sha256'];parent=torch.load(pp,map_location='cpu',weights_only=False)
  for a in ['F','T','S_new_crop','TS_new_crop']:assert np.array_equal(d['arms'][a]['boxes'],parent['arms'][a]['boxes']) and d['arms'][a]['interval']==parent['arms'][a]['interval']
  for role,anchor_key,sa,tsa in [('WF_common','common_wf_anchors','S_WF_common','TS_WF_common'),('DINO_common','common_dino_anchors','S_DINO_common','TS_DINO_common')]:
   f=d['fits'][role];o=d['readouts'][role]
   if not support:
    assert f is None and o is None and np.array_equal(d['arms'][sa]['boxes'],d['arms']['F']['boxes']) and np.array_equal(d['arms'][tsa]['boxes'],d['arms']['T']['boxes']);continue
   assert f['config']==read(BASE/'DESIGN_LOCK.json')['configs'][d['dataset']] and f['profile']=='current5' and f['seed_key']==stable_fit_key(d['dataset'],roster[cid]['query_key_sha256']) and f['reference_sha256']==sha(BASE/'REFERENCE_LOCK.json')
   assert core._tree_equal(f['initial_state'],case['source_light_initial_state'])
   r=fit_readback(f,ref[anchor_key]);reward_checks+=r['reward_scalars'];max_r=max(max_r,r['max_reward_error']);max_q=max(max_q,r['max_q_error'])
   assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes']);b=o['boxes'].cpu().numpy().astype(float);xy=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],axis=1)
   assert np.array_equal(xy,d['arms'][sa]['boxes']) and np.array_equal(xy,d['arms'][tsa]['boxes']);fits+=1;steps+=f['optimizer_steps']
  pins.extend([item,dict(path=str(path.with_suffix('.receipt.json').relative_to(ROOT)),sha256=sha(path.with_suffix('.receipt.json'))),dict(path=str(sp.relative_to(ROOT)),sha256=sha(sp)),dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp))])
 bindings={k:sha(p)for k,p in [('student_barrier_sha256',BASE/'STUDENT_GLOBAL_BARRIER.json'),('reference_lock_sha256',BASE/'REFERENCE_LOCK.json'),('teacher_barrier_sha256',BASE/'TEACHER_GLOBAL_BARRIER.json'),('student_runtime_sha256',BASE/'STUDENT_RUNTIME.json'),('teacher_runtime_sha256',BASE/'RUNTIME.json'),('parent_student_barrier_sha256',PARENT/'STUDENT_GLOBAL_BARRIER.json')]}
 result=dict(status='pass',count=32,common_anchor_count=common,source_with_common_count=cases_with_common,actual_fit_count=fits,accepted_optimizer_steps=steps,independent_sampled_reward_scalar_count=reward_checks,max_reward_error=max_r,max_q_error=max_q,bindings=bindings,GT_read=False,CPU_only=True,scope='same source/pixel/frame/common weights/reference geometry/saved policy-reward-softmax-rollback and complete full-model readout; no independent backward replay')
 write(BASE/'ROOT_READBACK.json',result)
 for p in [BASE/n for n in ['DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json','TEACHER_JOB_LOCK.json','RUNTIME.json','STUDENT_RUNTIME.json','TEACHER_GLOBAL_BARRIER.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json']]+[OLD/'SOURCE_RUNTIME.json',OLD/'RUNTIME.json',OLD/'DEVELOPMENT_ROSTER_LOCK.json',PARENT/'STUDENT_GLOBAL_BARRIER.json']:
  pins.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=pins,scorer_sha256=sha(ROOT/'scripts/measure_stvg_reference_swap_v1.py'),GT_read=False,student_barrier_sha256=bindings['student_barrier_sha256'],teacher_barrier_sha256=bindings['teacher_barrier_sha256'],root_readback_sha256=sha(BASE/'ROOT_READBACK.json')))
 print(result)
if __name__=='__main__':main()

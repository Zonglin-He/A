"""Pre-score independent arithmetic, actual model-readout and source-state checks."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *

def check_matched_direction(nc):
 import numpy as np
 from scripts.audit_stvg_soft_ep_v1 import arr
 a=arr(nc['matched_direction']);s=arr(nc['soft_direction']);d=arr(nc['original_direction']);dtype=a.dtype;eps=float(np.finfo(dtype).eps);a=a.astype(np.float64);s=s.astype(np.float64);d=d.astype(np.float64);an=float(np.linalg.norm(a));sn=float(np.linalg.norm(s));dn=float(np.linalg.norm(d));assert dn>0 and sn>0
 expected=d*(sn/dn);norm_error=abs(an-sn);component_error=float(np.max(np.abs(a-expected)))
 assert norm_error<=max(2e-6,4*eps*sn) and component_error<=4*eps*max(float(np.max(np.abs(expected))),1e-12)+1e-12
 return norm_error/max(sn,1e-30),component_error

def main():
 import torch,numpy as np
 from scripts import stvg_target_teacher_student_v1 as core
 from scripts.audit_stvg_soft_ep_v1 import arr,err,iou,rollout_audit,_optimizer
 from scripts.run_stvg_table1_student004 import stable_fit_key
 b=BASE/'trials'/sys.argv[1];rt=read(b/'RUNTIME.json');revision=BASE/'recovery/norm_readback006/RUNTIME.json';rrt=read(revision)
 for p,h in rrt['pins'].items():assert sha(ROOT/p)==h,p
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 design=read(b/'DESIGN_LOCK.json');bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==128 and not bar['GT_read'];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};ts={r['case_id']:r for r in read(BASE/'TEMPORAL_LOCK.json')['records']};rr={r['probe_alias']:r for r in roster()};counters=dict(new_fits=0,reused_fits=0,no_reference=0,accepted=0,proposals=0,equal_norm=0,soft_lower_actual_U=0);errors=dict(reward=0.,loss=0.,adam=0.,phi=0.,response=0.);sources=[]
 assert len(bar['files'])==len({x['path']for x in bar['files']})==128
 for x in bar['files']:
  receipt=ROOT/x['receipt_path'];assert receipt== (ROOT/x['path']).with_suffix('.receipt.json')and sha(receipt)==x['receipt_sha256'];assert read(receipt)['status']=='complete'and read(receipt)['sha256']==x['sha256']and not read(receipt)['GT_read']
  p=ROOT/x['path'];assert sha(p)==x['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];rec=rr[cid];c,sp=load_case(rec);aa=refs[cid]['anchors'];pos=[a['position']for a in aa];ds=rec['dataset'];key=stable_fit_key(ds,rec['query_key_sha256'])
  assert d['reference_records']==refs[cid] and d['source_file_sha256']==sha(sp)and d['source_case_sha256']==c['source_case_sha256']and d['source_restored']and not d['GT_read']and d['runtime_sha256']==sha(b/'RUNTIME.json')
  assert np.array_equal(d['arms']['F']['boxes'],c['native_boxes_xyxy_normalized'])and np.array_equal(d['arms']['T']['boxes'],d['arms']['F']['boxes'])and d['arms']['T']['interval']==ts[cid]['selected_interval']and d['arms']['F']['interval']==c['Native_interval']
  u=[i for i in range(len(c['frame_ids']))if i not in set(pos)];ev=np.asarray([a['box_cxcywh']for a in aa]);weights=np.asarray([a['weight']for a in aa])
  for mode in ['M0','M1','M2']:
   f=d['fits'][mode];o=d['readouts'][mode]
   if not aa:
    assert f is o is None;counters['no_reference']+=1;assert np.array_equal(d['arms']['S_'+mode]['boxes'],d['arms']['F']['boxes'])and np.array_equal(d['arms']['TS_'+mode]['boxes'],d['arms']['T']['boxes']);continue
   assert f['config']==design['configs'][ds][mode] and f['seed_key']==key and core._tree_equal(f['initial_state'],c['source_light_initial_state'])and f['active_parameters']==1792 and f['GT_read']is False
   assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes']);bb=arr(o['boxes']).astype(float);xy=np.concatenate((bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2),1)
   assert np.array_equal(xy,d['arms']['S_'+mode]['boxes'])and np.array_equal(xy,d['arms']['TS_'+mode]['boxes']);assert d['arms']['S_'+mode]['interval']==c['Native_interval']and d['arms']['TS_'+mode]['interval']==ts[cid]['selected_interval'];assert len(f['path'])==f['config']['steps']
   reused='reused_fit_from_sha256'in f;counters['reused_fits'if reused else'new_fits']+=1
   if reused:
    old=BASE/'trials'/design['family_reuse_from'][ds][mode]/'student'/(cid+'.pt');assert sha(old)==f['reused_fit_from_sha256'];z=torch.load(old,map_location='cpu',weights_only=False);ff=dict(f);ff.pop('reused_fit_from_sha256');ff.pop('reused_fit_from_trial',None);assert core._tree_equal(ff,z['fits'][mode])and core._tree_equal(o,z['readouts'][mode])
   if mode=='M0':
    prev=f['initial_state'];accepted=0
    for row in f['path']:
     assert core._tree_equal(row['state'],prev)
     if row.get('skipped'):continue
     witness=dict(row,boxes_before=row['boxes'],per_anchor_loss=row['anchor_loss']);ra=rollout_audit(witness,ev,weights,f['config'],pos,key,row['step']);errors['reward']=max(errors['reward'],ra['reward']);errors['loss']=max(errors['loss'],ra['loss'])
     if row['proposed']:
      aw=dict(row,state_before=row['state'],gradient=row['update']['gradient'],original_adam_delta=row['update']['raw']);er=max(_optimizer(aw,list(row['state'])));assert er<1e-6;errors['adam']=max(errors['adam'],er)
     before=arr(row['boxes']);after=arr(row['proposed_boxes']);q0=iou(before[pos],ev);q1=iou(after[pos],ev);assert err(q0,row['reference_iou_before'])<3e-6 and err(q1,row['reference_iou_proposed'])<3e-6
     if row['accepted']:
      assert float((weights*(q1-q0)).sum()/weights.sum())>=-1e-6
      assert float((weights*(arr(row['reference_iou_proposed'])-arr(row['reference_iou_before']))).sum()/weights.sum())>=-1.5e-7;accepted+=1
     elif row['proposed']:assert core._tree_equal(row['state'],row['update_state'])and core._tree_equal(row['optimizer_before'],row['optimizer_after'])
     prev=row['update_state']
    assert core._tree_equal(prev,f['final_state'])and accepted==f['optimizer_steps'];continue
   assert f['settings']==design['response_settings'][ds][mode]
   prev=f['initial_state'];ac=pr=0
   for step,row in enumerate(f['path']):
    assert core._tree_equal(prev,row['state_before'])
    if row.get('skipped'):continue
    ra=rollout_audit(row,ev,weights,f['config'],pos,key,step);errors['reward']=max(errors['reward'],ra['reward']);errors['loss']=max(errors['loss'],ra['loss']);er=max(_optimizer(row,list(row['state_before'])));assert er<1e-6;errors['adam']=max(errors['adam'],er);pr+=1
    direction=arr(row['filtered_delta']).astype(float);adam=arr(row['original_adam_delta']).astype(float)
    if mode=='M1':assert err(direction,adam)<1e-6
    else:
     s=row['solver']
     if s:
      assert abs(s['rho']-np.linalg.norm(direction)/max(1e-30,np.linalg.norm(adam)))<1e-5;assert math.isfinite(s['solver_residual_rel'])and s['solver_residual_rel']>=0
      if s['status']!='near_zero_response_identity':assert abs(s['lambda']-s['mu']*float(adam@adam)/float(arr(s['Jd']).astype(float)@arr(s['Jd']).astype(float)))<=1e-4*max(1.,s['lambda'])
    if row['accepted']:
     ac+=1;last=row['candidates'][-1];assert last['train_pass']and last['response_pass']and last['accepted'];assert float((weights*(arr(last['anchor_IoU'])-arr(row['reference_iou_before']))).sum()/weights.sum())>=-1.5e-7;delta=np.concatenate([(arr(row['state_after'][n])-arr(v)).ravel()for n,v in row['state_before'].items()]);e=err(delta,direction*row['accepted_scale']);assert e<3e-7;errors['phi']=max(errors['phi'],e);assert np.linalg.norm(delta)>0
     after=arr(row['boxes_after']).astype(float);before=arr(row['boxes_before']).astype(float);actual=float(np.sqrt(np.square(after[u]-before[u]).mean()))if u else 0.;assert actual<=row['response_budget']+1.1e-7;assert float((weights*(iou(after[pos],ev)-iou(before[pos],ev))).sum()/weights.sum())>=-1e-6;errors['response']=max(errors['response'],abs(actual-last['U_RMS']))
    else:assert core._tree_equal(row['state_after'],row['state_before'])and core._tree_equal(row['optimizer_after'],row['optimizer_before'])
    nc=row.get('norm_control')
    if nc:
     check_matched_direction(nc);counters['equal_norm']+=1;counters['soft_lower_actual_U']+=nc['soft_U_RMS']<nc['matched_U_RMS']
    prev=row['state_after']
   assert core._tree_equal(prev,f['final_state'])and ac==f['optimizer_steps']and pr==f['proposed_optimizer_steps'];counters['accepted']+=ac;counters['proposals']+=pr
  sources.append(dict(case_id=cid,source_file_sha256=sha(sp),student_sha256=sha(p)))
 result=dict(status='pass',reader_source_sha256=sha(Path(__file__)),reader_revision_runtime_sha256=sha(revision),count=128,counters=counters,max_errors=errors,source_bindings=sources,GT_read=False,CPU_only=True,scope='independent saved reward/Adam/CG witnesses, state and rollback; actual full decoder outputs exact; no new independent backward/fullJacobian',runtime_sha256=sha(b/'RUNTIME.json'),barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'))
 write(b/'ROOT_READBACK.json',result)
 files=list(bar['files'])
 for fp in [Path(__file__),revision,BASE/'recovery/norm_readback006/ROOT_SCALE_AWARE_NORM_DIAGNOSIS.json']:
  files.append(dict(path=str(fp.relative_to(ROOT)),sha256=sha(fp)))
 files.extend(dict(path=x['receipt_path'],sha256=x['receipt_sha256'])for x in bar['files'])
 for n in ['DESIGN_LOCK.json','RUNTIME.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json']:files.append(dict(path=str((b/n).relative_to(ROOT)),sha256=sha(b/n)))
 for n in ['ROSTER_LOCK.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json','ROOT_INPUT_ASSEMBLY.json']:files.append(dict(path=str((BASE/n).relative_to(ROOT)),sha256=sha(BASE/n)))
 write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=128,files=files,runtime_sha256=sha(b/'RUNTIME.json'),scorer_sha256=sha(ROOT/'scripts/measure_stvg_soft_ep_coordinate_trial_v1.py'),GT_read=False));print('root128',counters,errors)
if __name__=='__main__':main()

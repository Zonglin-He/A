"""Saved arithmetic and postseal scoring for finite reference recovery stages."""
import os,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_evidence_common_v1 import load_case
from scripts.measure_stvg_crc_v1 import iou
from scripts.stvg_reference_recovery_common_v1 import PARENT
STUDY=ROOT/'artifacts/stvg_reference_recovery_v1'
def base_for(stage):return STUDY/('Optimizer_replay'if stage=='optimizer'else'LN_carry')
def verify(base):
 r=read(base/'RUNTIME.json');assert r['status']=='locked' and not r['GT_read']
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 return r

def fit_readback(fit,anchors):
 import torch,numpy as np
 from scripts import stvg_target_teacher_student_v1 as core
 assert fit['active_parameters']==1792 and fit['GT_read'] is False
 assert torch.count_nonzero(fit['initial_state']['spatial.query_residual'])==0
 assert fit['optimizer_state_reset_each_query'] and fit['query_residual_reset_at_arrival']
 source_boxes=fit['initial_cached_decoder_boxes'];last=source_boxes;last_state=fit['initial_state'];checks=0;max_reward=max_q=0.
 positions=[a['position']for a in anchors];refs=np.array([a['box_cxcywh']for a in anchors]);weights=np.array([a['weight']for a in anchors])
 for i,row in enumerate(fit['path']):
  assert core._tree_equal(last_state,row['state_before']) and torch.equal(last,row['boxes_before'])
  if i==0:assert row['optimizer_before']['state']=={}
  if 'rollout'in row:
   p=row['rollout'];samples=p['samples'].numpy().astype(float);calc=iou(1/(1+np.exp(-samples)),refs[:,None,:]);error=float(abs(calc-p['rewards'].numpy()).max());assert error<5e-6;max_reward=max(max_reward,error);checks+=calc.size
   logits=calc/fit['config']['tau'];logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True);qe=float(abs(q-p['weights'].numpy()).max());assert qe<5e-6;max_q=max(max_q,qe)
   if row['accepted']:
    before=iou(row['boxes_before'].numpy()[positions],refs);after=iou(row['boxes_proposed'].numpy()[positions],refs);assert float((weights*(after-before)).sum()/weights.sum())>=-2e-6
  if row['rollback']:
   assert core._tree_equal(row['optimizer_before'],row['optimizer_after']);assert core._tree_equal(row['state_before'],row['state_after'])
  last,last_state=row['boxes_after'],row['state_after']
 assert torch.equal(last,fit['final_cached_decoder_boxes']) and core._tree_equal(last_state,fit['final_state'])
 assert len(fit['path'])==fit['declared_steps']
 return dict(reward_scalars=int(checks),max_reward_error=max_reward,max_q_error=max_q)

def root_optimizer():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from scripts import stvg_target_teacher_student_v1 as core
 base=base_for('optimizer');verify(base);sys.addaudithook(guard)
 roster=read(base/'ROSTER_LOCK.json')['records'];byid={r['probe_alias']:r for r in roster};refs={r['case_id']:r for r in read(base/'REFERENCE_LOCK.json')['records']}
 bar=read(base/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==32 and not bar['GT_read']
 assert {x['path']for x in bar['files']}=={str((base/'student'/(c+'.pt')).relative_to(ROOT))for c in byid}
 checks=fits=steps=0;max_r=max_q=0.;pins=[]
 parentfiles={x['path']:x for x in read(PARENT/'STUDENT_GLOBAL_BARRIER.json')['files']}
 for item in bar['files']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];case,sp=load_case(byid[cid]);assert d['source_case_sha256']==case['source_case_sha256'] and sha(sp)==d['source_file_sha256'] and d['source_restored'] and not d['GT_read'];assert d['reference_records']==refs[cid]
  pp=PARENT/'student'/(cid+'.pt');assert sha(pp)==parentfiles[str(pp.relative_to(ROOT))]['sha256']==d['parent_student_sha256'];parent=torch.load(pp,map_location='cpu',weights_only=False)
  for a,b in [('F','F'),('T','T'),('S_current','S_new_crop'),('TS_current','TS_new_crop')]:
   assert np.array_equal(d['arms'][a]['boxes'],parent['arms'][b]['boxes']);assert d['arms'][a]['interval']==parent['arms'][b]['interval']
  aa=refs[cid]['anchors'];fit=d['fit']
  if aa:
   assert fit['profile']=='original' and fit['config']==read(base/'DESIGN_LOCK.json')['original_configs'][d['dataset']]
   assert torch.equal(d['readout']['boxes'].cpu(),fit['final_cached_decoder_boxes']);r=fit_readback(fit,aa);checks+=r['reward_scalars'];max_r=max(max_r,r['max_reward_error']);max_q=max(max_q,r['max_q_error']);fits+=1;steps+=fit['optimizer_steps']
   boxes=fit['final_cached_decoder_boxes'].numpy().astype(float);xy=np.concatenate([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],axis=1);assert np.array_equal(xy,d['arms']['S_original']['boxes'])
  else:assert fit is None and np.array_equal(d['arms']['S_original']['boxes'],d['arms']['F']['boxes'])
  pins.extend([dict(path=str(sp.relative_to(ROOT)),sha256=sha(sp)),dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp)),item])
 result=dict(status='pass',case_count=32,actual_fit_count=fits,accepted_optimizer_steps=steps,independent_sampled_reward_scalar_count=checks,max_reward_error=max_r,max_q_error=max_q,CPU_only=True,GT_read=False,scope='saved policy/reward/softmax/rollback/full-vs-cached decoder and Source reset; no independent backward replay')
 write(base/'ROOT_READBACK.json',result)
 for name in ['ROSTER_LOCK.json','DESIGN_LOCK.json','RUNTIME.json','REFERENCE_LOCK.json','ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json']:
  p=base/name;pins.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(base/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=pins,scorer_sha256=sha(__file__),GT_read=False))
 print('optimizer root',result)

def score_optimizer():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from scripts import review_stvg_sparse_r1_v2 as scorer
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 base=base_for('optimizer');verify(base);seal=read(base/'GLOBAL_PREDICTION_BARRIER.json');assert seal['status']=='complete' and seal['scorer_sha256']==sha(__file__) and not seal['GT_read']
 for p in seal['files']:assert sha(ROOT/p['path'])==p['sha256']
 cases={};outputs={}
 for rec in read(base/'ROSTER_LOCK.json')['records']:
  cid=rec['probe_alias'];cases[cid],_=load_case(rec);outputs[cid]=torch.load(base/'student'/(cid+'.pt'),map_location='cpu',weights_only=False)
 write(base/'GT_EXPOSURE.json',dict(status='after_complete_predictions_and_root',seal_sha256=sha(base/'GLOBAL_PREDICTION_BARRIER.json'),GT_read=True,time=time.time()))
 truth,provenance=scorer._truths_after_seal(cases);official,independent,_=scorer._metric_functions();rows=[]
 for cid,d in outputs.items():
  support={a['position']for a in d['reference_records']['anchors']}
  for arm,pred in d['arms'].items():
   values,audit=metric._score_student_arm(pred,cases[cid],truth[cid],support,official,independent);rows.append(dict(case_id=cid,dataset=d['dataset'],arm=arm,metrics=values,support=audit['support'],metric_audit=audit['metric_audit']))
 summary=[];paired=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];lookup={(r['case_id'],r['arm']):r for r in rr};ids=sorted({r['case_id']for r in rr});assert len(ids)==16
  for arm in sorted({r['arm']for r in rr}):
   for key in rr[0]['metrics']:
    vals=[lookup[c,arm]['metrics'][key]for c in ids if lookup[c,arm]['metrics'][key]is not None];summary.append(dict(dataset=ds,arm=arm,metric=key,count=len(vals),mean=math.fsum(vals)/len(vals)if vals else None))
  for a,b in [('S_original','F'),('TS_original','T'),('S_original','S_current'),('TS_original','TS_current')]:
   delta={c:100*(lookup[c,a]['metrics']['m_vIoU']-lookup[c,b]['metrics']['m_vIoU'])for c in ids};v=list(delta.values());paired.append(dict(dataset=ds,left=a,right=b,mean_delta_pp=math.fsum(v)/16,positive=sum(x>1e-10 for x in v),negative=sum(x< -1e-10 for x in v),neutral=sum(abs(x)<=1e-10 for x in v),worst=min(v),best=max(v),per_source_delta_pp=delta))
 write(base/'POSTSEAL_ROWS.json',dict(status='complete',rows=rows,GT_read=True));write(base/'POSTSEAL_SUMMARY.json',dict(status='complete',summary=summary,GT_read=True));write(base/'PARENT_PAIRED_DELTAS.json',dict(status='complete',records=paired,GT_read=True,bootstrap_executed=False));print('optimizer vIoU',[r for r in summary if r['metric']=='m_vIoU'])
if __name__=='__main__':
 {'root_optimizer':root_optimizer,'score_optimizer':score_optimizer}[sys.argv[1]]()

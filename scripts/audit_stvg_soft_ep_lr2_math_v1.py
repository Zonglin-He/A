"""Independent saved Gaussian reward/loss/Adam and soft-direction witnesses; no GT."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
def main():
 import torch,numpy as np
 from scripts.audit_stvg_soft_ep_v1 import _fit_audit,rollout_audit,_optimizer,arr,err,treeeq
 from scripts.run_stvg_table1_student004 import stable_fit_key
 r=read(BASE/'RUNTIME.json')
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 b=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert b['status']=='complete'and b['count']==32 and not b['GT_read']
 design=read(BASE/'DESIGN_LOCK.json');roster={x['probe_alias']:x for x in read(BASE/'ROSTER_LOCK.json')['records']};rows=[];m0rows=[]
 for x in b['files']:
  p=ROOT/x['path'];assert sha(p)==x['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];rec=roster[cid];ref=d['reference_records'];aa=ref['anchors']
  if not aa:continue
  key=stable_fit_key(rec['dataset'],rec['query_key_sha256']);cfg=design['configs'][rec['dataset']]
  for mode in ['M1','M2']:rows.append(_fit_audit(cid,mode,d['fits'][mode],ref,design['response_settings'],cfg,key))
  f=d['fits']['M0matched'];pos=[a['position']for a in aa];ev=np.array([a['box_cxcywh']for a in aa]);w=np.array([a['weight']for a in aa]);errors=dict(reward=0.,loss=0.,adam=0.);prev=f['initial_state'];accepted=0
  for row in f['path']:
   assert treeeq(row['state'],prev)
   if row.get('skipped'):continue
   a=rollout_audit(dict(row,boxes_before=row['boxes'],per_anchor_loss=row['anchor_loss']),ev,w,cfg,pos,key,row['step'])
   errors['reward']=max(errors['reward'],a['reward']);errors['loss']=max(errors['loss'],a['loss'])
   if row['proposed']:
    z=dict(row,state_before=row['state'],gradient=row['update']['gradient'],original_adam_delta=row['update']['raw']);e=max(_optimizer(z,list(row['state'])));assert e<2e-6;errors['adam']=max(errors['adam'],e)
   if row['accepted']:
    assert float((w*(arr(row['reference_iou_proposed'])-arr(row['reference_iou_before']))).sum()/w.sum())>=-1.5e-7;accepted+=1
   elif row['proposed']:assert treeeq(row['state'],row['update_state'])and treeeq(row['optimizer_before'],row['optimizer_after'])
   prev=row['update_state']
  assert treeeq(prev,f['final_state'])and accepted==f['optimizer_steps'];m0rows.append(dict(case_id=cid,accepted=accepted,max_errors=errors))
 assert len(rows)==46 and len(m0rows)==23
 write(BASE/'ROOT_ADDITIONAL_MATH_READBACK.json',dict(status='pass',count=32,soft_fit_count=46,M0matched_fit_count=23,soft_fit_audits=rows,M0matched_audits=m0rows,runtime_sha256=sha(BASE/'RUNTIME.json'),student_barrier_sha256=sha(BASE/'STUDENT_GLOBAL_BARRIER.json'),reader_sha256=sha(__file__),scope='independent saved reward/q/Gaussian loss/Adam/CG/rollback/equal-norm witnesses, no independent backward or full Jacobian',GT_read=False,CPU_only=True,model_loaded=False))
 print('independent LR2 math pass69 fits')
if __name__=='__main__':main()

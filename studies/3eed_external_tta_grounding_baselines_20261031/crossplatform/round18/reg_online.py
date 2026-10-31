"""Reg-inspired native pre-update online supplementary protocol, B32 only.
Keeps student/Adam/CBU state within platform. Metadata order is not timestamp.
"""
import sys,time,hashlib
import numpy as np
import torch
from common import ROOT,EXP,read,dump,sha,seal,check_plan
from runner import Reset
from reg_inspired import run_reg
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,mask_for,fingerprint
from runner_helpers import sample_points
def main():
 plan=check_plan();lock=read(EXP/'ONLINE_SUPPLEMENT_LOCK.json');assert lock['program_sha256']==sha(__file__)
 out=ROOT/'crossplatform/round18/reg_online32_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);reset=Reset(model);rows=read(EXP.parent/'input/B32_PRIVATE.json');tr=[];start=time.perf_counter()
 for platform in ('drone','quad'):
  reset.reset(0);teacher=student=opt=None
  for i,row in enumerate(r for r in rows if r['platform']==platform):
   pc,_,_=sample_points(row);pc=torch.tensor(pc,device='cuda')[None];text=row['caption']+' . not mentioned';m,_=mask_for(model,text,'all_content_tokens');mask=torch.as_tensor(m,device='cuda',dtype=torch.bool);seed=int(hashlib.sha256(('R18online|'+row['id']).encode()).hexdigest()[:8],16)%2**31
   pre,post,stu,direct,d,selected,teacher,student,opt=run_reg(model,pc,text,mask,seed,teacher,student,opt)
   names=['Reg_online_teacher_pre','Reg_online_teacher_post','Reg_online_student_post'];boxes=[]
   for x in (pre,post,stu):boxes.append(x['boxes'][int(x['scores'].argmax())])
   np.savez_compressed(out/'predictions_PRIVATE'/(row['id']+'.npz'),sample_id=np.array(row['id']),arm_names=np.array(names),arm_boxes_raw6=np.array(boxes,np.float32))
   tr.append({'alias':f'online_{platform}_{i:03d}','id_PRIVATE':row['id'],'platform':platform,'diagnostics':d});print('REG_ONLINE',platform,i+1,flush=True)
  np.savez_compressed(out/(platform+'_teacher_final_PRIVATE.npz'),**{n:t.detach().cpu().numpy() for n,t in teacher.items()})
 reset.reset();assert fingerprint(model)==fp;dump(out/'TRACE_PRIVATE.json',tr);dump(out/'registered_lock.json',lock);seal(out,'R18_Reg_inspired_online_preupdate_predictions_before_current_GT',n=32,source_unchanged=True,seconds=time.perf_counter()-start,program_sha256=sha(__file__));print('REG_ONLINE_SEALED',flush=True)
if __name__=='__main__':main()

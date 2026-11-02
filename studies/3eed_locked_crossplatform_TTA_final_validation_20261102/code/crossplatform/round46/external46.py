"""Existing four external grounding ports, isolated current-input episodic table."""
import sys,time,hashlib
import numpy as np
from common46 import *
def main():
 a=args(__doc__);plan,rows,out=context(a);verify(out/'source');folder=out/'external';folder.mkdir();(folder/'predictions_PRIVATE').mkdir()
 sys.path[:0]=[str(ROOT/'crossplatform/round18'),str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
 from runner import Reset
 from run_diagnostic import load_model,mask_for,fingerprint
 from runner_helpers import sample_points
 from algorithms import tent,detach
 from supplement_algorithms import sar,memo
 from reg_inspired import run_reg
 import torch
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);reset=Reset(model);start=time.perf_counter();trace=[]
 for i,r in enumerate(rows):
  seed=int(hashlib.sha256(('R46_external|'+r['id']).encode()).hexdigest()[:8],16)%2**31;tick=time.perf_counter();reset.reset(seed);points,_,_=sample_points(r);pc=torch.tensor(points,device='cuda')[None];text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');mask=torch.tensor(mask,device='cuda');torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():source=detach(model({'point_clouds':pc,'text':[text]}),mask)
  saved=npz(out/'source/samples_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(source['boxes'],saved['boxes']) and np.array_equal(source['scores'],saved['scores']),'External source bridge must match main source before method effects'
  names=[];bb=[];qq=[];ss=[];pools=[];qover=[];bover=[];diag={}
  def add(name,pred,q=None,box=None):
   overridden_q=q is not None;q=int(pred['scores'].argmax()) if q is None else int(q);names.append(name);qq.append(q);bb.append(pred['boxes'][q].copy() if box is None else box.copy());ss.append(pred['scores'].copy());pools.append(pred['boxes'].copy());qover.append(overridden_q);bover.append(box is not None)
  add('source',source)
  reset.reset(seed);pre,post,d,sel=tent(model,pc,[text],mask);d.update(reset.audit(sel));diag['Tent']=d;add('Tent_BN_statistics_first_pre',pre);add('Tent_clean_post_port',post)
  reset.reset(seed);pre,post,d,sel=sar(model,pc,[text],mask);d.update(reset.audit(sel));diag['SAR_LN_only']=d;add('SAR_LN_first_pre',pre);add('SAR_LN_clean_post_port',post)
  rng=np.random.default_rng(seed);path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);raw=np.array(raw[:,:3],np.float32,copy=True)
  if r['platform']=='drone':raw[:,2]+=np.float32(1.8)
  views=np.stack([raw[rng.choice(len(raw),16384,replace=len(raw)<16384)]+rng.normal(0,.01,(16384,3)).clip(-.03,.03).astype(np.float32) for _ in range(4)])
  reset.reset(seed);post,prior,tq,d,sel=memo(model,torch.tensor(views,device='cuda'),[text]*4,mask,dict(source,points=points),True);d.update(reset.audit(sel));diag['MEMO_Transformer_AdamW']=d;add('MEMO_BN_prior_only',prior);add('MEMO_augmentation_only',source,tq);add('MEMO_AdamW1e5_clean_post_port',post)
  reset.reset(seed);pre,post,student,direct,d,sel,_,_,_=run_reg(model,pc,text,mask,seed);d.update(reset.audit(sel));diag['Reg_inspired']=d;add('Reg_inspired_teacher_pre',pre);add('Reg_inspired_teacher_post',post);add('Reg_inspired_student_post',student);add('Reg_inspired_direct_pseudo',pre,box=direct)
  assert np.isfinite(np.asarray(bb)).all();np.savez_compressed(folder/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),queries=np.array(qq),boxes=np.asarray(bb,np.float32),scores=np.stack(ss),box_pools=np.asarray(pools,np.float32),query_override=np.asarray(qover,bool),box_override=np.asarray(bover,bool),source_pool=saved['boxes'],source_native_probability=saved['scores'])
  trace.append({'alias':r.get('alias',r['id']),'platform':r['platform'],'diagnostics':diag,'seconds':time.perf_counter()-tick,'peak_bytes':torch.cuda.max_memory_allocated(),'source_bridge_exact':True});dump(folder/'TRACE_PRIVATE.json',trace);dump(folder/'PROGRESS.json',{'completed':i+1,'n':len(rows)})
  if (i+1)%16==0:print('R46_EXTERNAL',i+1,round(time.perf_counter()-start,1),flush=True)
 reset.reset();assert fingerprint(model)==fp;seal(folder,'R46_FOUR_EXTERNAL_EPISODIC_PORTS_ALL_CONTROLS_BEFORE_TASKSCORE',n=len(rows),source_unchanged=True,protocol='reset_each_current_input_pre_and_post_separate',no_prior_A_feedback_RGB_VLM_used=True,Reg_is_inspired_not_official_reproduction=True,seconds=time.perf_counter()-start);print('R46_EXTERNAL_SEALED',len(rows),flush=True)
if __name__=='__main__':main()

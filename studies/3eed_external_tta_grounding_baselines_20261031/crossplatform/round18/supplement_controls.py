"""Capture omitted C32 no-update controls, repeating fixed pre-update operations.
Current C task metrics are already exposed. No new configuration or selection;
require bitexact source and fixed SGD within a declared absolute FP32 tolerance.
"""
from pathlib import Path
import sys,time,hashlib
import numpy as np
import torch
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
from runner import Reset
from supplement_algorithms import sar,memo,detach
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,mask_for,fingerprint
from runner_helpers import sample_points
def main():
 check_plan();lock=read(EXP/'C_CONTROL_CAPTURE_LOCK_v3.json');assert lock['program_sha256']==sha(__file__);old=ROOT/'crossplatform/round18/supplement32_attempt001';verify(old);out=ROOT/'crossplatform/round18/supplement_controls32_attempt003';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();replay=[]
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);reset=Reset(model);start=time.perf_counter()
 for i,row in enumerate(read(EXP.parent/'input/C32_PRIVATE.json')):
  seed=int(hashlib.sha256(('R18C|'+row['id']).encode()).hexdigest()[:8],16)%2**31;reset.reset(seed);p,_,_=sample_points(row);pc=torch.tensor(p,device='cuda')[None];text=row['caption']+' . not mentioned';m,_=mask_for(model,text,'all_content_tokens');mask=torch.as_tensor(m,device='cuda')
  with torch.no_grad():source=detach(model({'point_clouds':pc,'text':[text]}),mask)
  rng=np.random.default_rng(seed);path=Path(row['pointpath']);raw=np.load(path) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);raw=np.array(raw[:,:3],np.float32,copy=True)
  if row['platform']=='drone':raw[:,2]+=np.float32(1.8)
  views=np.stack([raw[rng.choice(len(raw),16384,replace=len(raw)<16384)]+rng.normal(0,.01,(16384,3)).clip(-.03,.03).astype(np.float32) for _ in range(4)])
  reset.reset(seed);sar(model,pc,[text],mask)
  reset.reset(seed);post,prior,q,d,sel=memo(model,torch.tensor(views,device='cuda'),[text]*4,mask,dict(source,points=p),False)
  with np.load(old/'predictions_PRIVATE'/(row['id']+'.npz')) as z:
   names=z['arm_names'].tolist();assert np.array_equal(source['boxes'][source['scores'].argmax()],z['arm_boxes_raw6'][0]);expected=z['arm_boxes_raw6'][names.index('MEMO_SGD_25e5')];actual=post['boxes'][post['scores'].argmax()];err=float(np.max(np.abs(actual-expected)));print('REPEAT_MAX_ABS',i,err,flush=True);assert err<=lock['max_absolute_box_replay_error_m'],'Fixed memo repeat exceeds declared FP32 bound';replay.append({'alias':f'C32_{row["platform"]}_{i:03d}','source_bitexact':True,'adapted_max_abs_m':err,'adapted_bitexact':bool(np.array_equal(actual,expected))})
  names=['MEMO_C_BN_prior_only','MEMO_C_augmentation_only'];boxes=np.array([prior['boxes'][prior['scores'].argmax()],source['boxes'][q]],np.float32);np.savez_compressed(out/'predictions_PRIVATE'/(row['id']+'.npz'),sample_id=np.array(row['id']),arm_names=np.array(names),arm_boxes_raw6=boxes)
  reset.reset(seed);memo(model,torch.tensor(views,device='cuda'),[text]*4,mask,dict(source,points=p),True)
  if (i+1)%8==0:print('C_CONTROL',i+1,flush=True)
 reset.reset();assert fingerprint(model)==fp;dump(out/'registered_lock.json',lock);dump(out/'REPLAY.json',replay);seal(out,'R18_C32_controls_fixed_SGD_repeat_FP32_bounded',n=32,repeat_bitexact=all(x['adapted_bitexact'] for x in replay),max_adapted_replay_error_m=max(x['adapted_max_abs_m'] for x in replay),source_bitexact=True,source_unchanged=True,seconds=time.perf_counter()-start);print('C_CONTROLS_SEALED')
if __name__=='__main__':main()

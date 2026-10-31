"""Official optimizer / norm-scope diagnostic on a new fixed C32 cohort.
Declared after primary results, no new task-label tuning or primary replacement.
"""
import sys,time,hashlib
import numpy as np
import torch
from common import ROOT,EXP,read,dump,sha,seal,check_plan
from runner import Reset
from supplement_algorithms import sar,memo,detach
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,mask_for,fingerprint
from runner_helpers import sample_points
def main():
 check_plan();lock=read(EXP/'SUPPLEMENT_LOCK.json')
 for rel,h in lock['programs'].items():assert sha(ROOT/rel)==h
 rows=read(EXP.parent/'input/C32_PRIVATE.json');assert sha(EXP.parent/'input/C32_PRIVATE.json')==lock['C32_sha256'];out=ROOT/'crossplatform/round18/supplement32_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);reset=Reset(model);trace=[];start=time.perf_counter()
 for i,row in enumerate(rows):
  seed=int(hashlib.sha256(('R18C|'+row['id']).encode()).hexdigest()[:8],16)%2**31;reset.reset(seed);p,_,_=sample_points(row);pc=torch.tensor(p,device='cuda')[None];text=row['caption']+' . not mentioned';m,_=mask_for(model,text,'all_content_tokens');mask=torch.as_tensor(m,device='cuda');torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():source=detach(model({'point_clouds':pc,'text':[text]}),mask)
  rng=np.random.default_rng(seed);rawpath=__import__('pathlib').Path(row['pointpath']);raw=np.load(rawpath) if rawpath.suffix=='.npy' else np.fromfile(rawpath,np.float32).reshape(-1,4);raw=np.array(raw[:,:3],np.float32,copy=True)
  if row['platform']=='drone':raw[:,2]+=np.float32(1.8)
  views=np.stack([raw[rng.choice(len(raw),16384,replace=len(raw)<16384)]+rng.normal(0,.01,(16384,3)).clip(-.03,.03).astype(np.float32) for _ in range(4)]);names=['source'];boxes=[source['boxes'][source['scores'].argmax()]];diagnostics={}
  reset.reset(seed);_,post,d,selected=sar(model,pc,[text],mask);d.update(reset.audit(selected));diagnostics['SAR_LN_only']=d;names.append('SAR_LN_only');boxes.append(post['boxes'][post['scores'].argmax()])
  for config in (False,True):
   reset.reset(seed);post,prior,q,d,selected=memo(model,torch.tensor(views,device='cuda'),[text]*4,mask,dict(source,points=p),config);d.update(reset.audit(selected));name='MEMO_AdamW_1e5' if config else 'MEMO_SGD_25e5';names.append(name);boxes.append(post['boxes'][post['scores'].argmax()]);diagnostics[name]=d
  np.savez_compressed(out/'predictions_PRIVATE'/(row['id']+'.npz'),sample_id=np.array(row['id']),arm_names=np.array(names),arm_boxes_raw6=np.array(boxes,np.float32));trace.append({'id_PRIVATE':row['id'],'alias':f'C_{row["platform"]}_{i:03d}','platform':row['platform'],'diagnostics':diagnostics,'peak_bytes':torch.cuda.max_memory_allocated()});dump(out/'TRACE_PRIVATE.json',trace);print('SUPPLEMENT_C',i+1,flush=True)
 reset.reset();assert fingerprint(model)==fp;dump(out/'registered_lock.json',lock);seal(out,'R18_official_config_C32_sealed_before_current_C_labels',n=32,source_unchanged=True,seconds=time.perf_counter()-start);print('SUPPLEMENT_SEALED')
if __name__=='__main__':main()

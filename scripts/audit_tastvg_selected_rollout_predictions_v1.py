"""Independent no-GT receipt and original A stream parity readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_selected_rollout_common_v1 import *
def run(ds):
 import torch
 torch.set_num_threads(2)
 from methods.decota_final_simplified_v1.tensors import state_hash
 p=verify(ds);counts={};oldpath=ROOT/p['baseline_result_dir'];oldbar=read(oldpath/'PREDICTION_BARRIER.json')
 for arm in ['A','G']:
  out=BASE/ds/arm;b=read(out/'PREDICTION_BARRIER.json');assert b['cells']==384 and b['GT_read'] is False
  checks=0
  for rel,h in b['files'].items():
   rec=out/rel;assert sha(rec)==h;rf=read(rec);raw=rec.with_suffix('.pt');assert sha(raw)==rf['sha256'];x=load(raw)
   assert x['GT_read'] is False and state_hash(x['pre_state'])==x['pre_sha'] and state_hash(x['post_state'])==x['post_sha']
   assert rf['pre_sha']==x['pre_sha'] and rf['post_sha']==x['post_sha'] and rf['parent']==x['parent'];checks+=1
   if arm=='A':
    oldrec=oldpath/rel;assert sha(oldrec)==oldbar['files'][rel] and sha(oldrec.with_suffix('.pt'))==read(oldrec)['sha256'];old=load(oldrec.with_suffix('.pt'))
    assert old['pre_sha']==x['pre_sha'] and old['post_sha']==x['post_sha'] and old['final_indices']==x['final_indices'] and torch.equal(old['slow']['boxes'],x['slow']['boxes'])
    assert len(old['update_steps'])==len(x['update_steps'])
    for a,z in zip(old['update_steps'],x['update_steps']):
     assert a['pre_state_sha256']==z['pre_state_sha256'] and a['post_state_sha256']==z['post_state_sha256']
     if a['update'] is not None:
      assert a['update']['loss_before']==z['update']['loss_before']
      for n,g in a['update']['gradients'].items():assert torch.equal(g,z['update']['gradients'][n])
  assert checks==384;counts[arm]=checks
 assert not torch.cuda.is_initialized();write(BASE/ds/'ROOT_PREDICTION_READBACK.json',dict(status='pass',counts=counts,A_bitwise_original=True,GT_read=False,model_execution=False,GPU_initialized=False,time=time.time()));print(ds,'384A+384G receipts/hash and A full stream bitwise PASS')
if __name__=='__main__':run(sys.argv[1])

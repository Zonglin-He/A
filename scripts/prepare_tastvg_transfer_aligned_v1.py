"""Freeze dependencies before new inference; no label reads."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_transfer_aligned_common_v1 import *
def run():
 assert not BASE.exists(),'new namespace only';assert read(OLD/'FINAL_COMPLETION.json')['status']=='completed'
 inputs={};bind=lambda f:inputs.update({str(f.relative_to(ROOT)):sha(f)})
 for f in [OLD/'FINAL_COMPLETION.json',OLD/'hc2/PLAN.json',POOL/'hc2/CAPTURE_BARRIER.json',ROOT/'methods/CURRENT_METHOD.json']:bind(f)
 for arm in ['A','R']:
  for f in (OLD/'hc2'/arm/'online').rglob('*.pt'):
   if int(f.stem)%4==0:bind(f);bind(f.with_suffix('.json'))
 names=['scripts/tastvg_transfer_aligned_common_v1.py','scripts/prepare_tastvg_transfer_aligned_v1.py','scripts/run_tastvg_single_write_v1.py','protocols/tastvg_transfer_aligned_v1.md','vg_tta/tastvg_native_spatial_rollout_s05_v1.py','vg_tta/tastvg_evidence_capture_v1.py','scripts/run_tastvg_evidence_vulnerability_v2.py','methods/decota_final_simplified_v1/backbone.py','methods/decota_final_simplified_v1/objectives.py','scripts/run_spatial_regression_alignment_v1.py']
 write(BASE/'TRANSFER_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in names},inputs=inputs,logical_pairs=768,expected_write_payloads=192,GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='preparing_transfer_and_token',GT_read=False,time=time.time()));status(BASE/'transfer/STATUS.json',dict(status='ready',done=0,total=192))
 status(BASE/'token/STATUS.json',dict(status='preparing_frozen_CLIP_and_parser',done=0,total=60))
 archive('两项新机制诊断已登记锁定；即将启动最高优先级HC saved-state迁移')
if __name__=='__main__':run()

"""Read-only legacy cache qualification. No labels, media, model or task scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import torch,numpy as np
 from torch._subclasses.fake_tensor import FakeTensorMode
 from vg_tta.decota_tastvg_episode_v1 import native_view_indices
 D=R/'artifacts/decota_corrective_identifiability_v1/spatial_gpu_v1';out=R/'artifacts/stvg_temporal_moving_rgb128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_moving_rgb128';lock=read(D/'LOCK.json');rows=[];private=[];checks=0
 forbidden=str(Path(lock['labels']).resolve())
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
   path=str(Path(args[0]).resolve())
   if path==forbidden or any(v in path for v in ['/annotations/','/annos/','GT_LABEL','labels_diagnostic','val_v2.json','valv2_proc.json']):raise PermissionError('legacy qualification forbids labels')
 sys.addaudithook(guard)
 for n,r in enumerate(lock['rows']):
  alias=('LH'if r['cohort']=='hcstvg1_test'else'LV')+f'{n%8+1:02}'
  f=D/'teachers'/f"{r['key'].replace(':','_')}.pt";rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256']and rc['GT_access']is False
  ef=D/'episodes'/f"{r['key'].replace(':','_')}.pt";erc=read(ef.with_suffix('.json'));assert sha(ef)==erc['sha256']
  # Only the metadata below is consumed. No real storage from the post-GT episode is loaded.
  with FakeTensorMode():e=torch.load(ef,map_location='cpu',weights_only=False)
  ids=list(map(int,e['frame_ids']));inp={k:e['input'][k]for k in ['width','height','fps','frame_count','video_sha256']};old_iv=list(e['native']['physical_interval']);old_raw=[list(x)for x in e['native']['raw_physical_intervals']];assert len(ids)>=4 and ids==sorted(set(ids));assert e['teacher_receipt']['sha256']==rc['sha256'];del e
  z=torch.load(f,map_location='cpu',weights_only=False);assert z['key']==r['key']and z['GT_access']is False;native=z['native'];assert native['physical_interval']==old_iv
  assert all(torch.equal(a,b)for a,b in zip(native['logits'],z['views']['roundtrip_noop']['logits']));checks+=2
  assert torch.equal(native['boxes'],z['views']['roundtrip_noop']['mapped_boxes']);checks+=1
  reconstructed=[]
  for off,v in enumerate(native['logits']):
   assert v.dtype==torch.float32 and tuple(v.shape)==(1,len(ids[off::2]),2)and bool(torch.isfinite(v).all());a,b=native_view_indices(v);reconstructed.append([ids[off::2][a],ids[off::2][b]+1])
  assert reconstructed==old_raw==native['raw_physical_intervals'];assert [min(v[0]for v in reconstructed),max(v[1]for v in reconstructed)]==old_iv;checks+=3
  cycle=z['views']['cycle'];assert cycle['actual_two_offset_forward']and cycle['shape']==[len(ids),224,224,3]and len(cycle['pixel_crop'])==len(ids)
  assert all(v.dtype==torch.float32 and tuple(v.shape)==tuple(n.shape)and bool(torch.isfinite(v).all())for v,n in zip(cycle['logits'],native['logits']))
  cfg=lock['configs'][r['cohort']];rows.append(dict(alias=alias,cohort=r['cohort'],frames=len(ids),offset_lengths=[len(ids[off::2])for off in [0,1]],frame_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),source_checkpoint_sha256=cfg['checkpoint_sha256'],source_dataset=cfg['source_dataset'],dtype='float32',native_roundtrip_logits_and_boxes_exact=True,original_MAP_physical_clock_exact=True,cycle_zero_state_same_offset_shapes=True,old_panel_present=Path(r['path']).exists()))
  private.append(dict(alias=alias,teacher=str(f.relative_to(R)),teacher_sha256=rc['sha256'],teacher_receipt_sha256=sha(f.with_suffix('.json')),episode=str(ef.relative_to(R)),episode_sha256=erc['sha256'],metadata=inp,frame_ids=ids));del z,native,cycle
 assert len(rows)==16 and checks==96 and not any(v['old_panel_present']for v in rows)
 pub=dict(status='pass_scoped_saved_outputs',parents=16,cohorts={'HCSTVG1':8,'VidSTG':8},new_GT=False,new_task_scores=False,new_models=0,new_GPU=0,source_weights_not_loaded=True,real_episode_tensor_storage_not_loaded=True,checks=checks,rows=rows,limitations=['Original panel files absent; no original media/input-cache or full-inference revalidation','HC1 grid100 and Vid varied23..200 are not current HC2 RC128 grid65','Different Vid Source checkpoint is not current Vid Source study','Historical per-episode teacher barrier is not a global pre-GT barrier; all sources previously exposed','No current moving-Live41 temporal advantage established'])
 put(out/'LEGACY_PRIVATE_QUALIFICATION.json',dict(status='pass',old_lock_sha256=sha(D/'LOCK.json'),records=private));put(P/'LEGACY_QUALIFICATION_PUBLIC.json',pub);print(json.dumps({k:v for k,v in pub.items()if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()

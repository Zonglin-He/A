import sys,time,hashlib,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_extended_common_v3 import *
def run(dataset):
 p=verify(dataset);out=BASE/dataset;t=time.time();done=0;lease=None
 try:
  import torch,numpy as np
  from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from scripts.run_tastvg_paper48_p5_online_v1 import source_capture
  from scripts.run_tastvg_full_b1_experts_v1 import observation
  from methods.decota_final_simplified_v1.tensors import state_hash
  sys.addaudithook(guard);install_clean_loader();decode=bind_decode(dataset);lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  from scripts.run_spatial_regression_alignment_v1 import model_load
  model=model_load('hcstvg1_test' if dataset=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert model.cfg.DATASET.NAME==('VidSTG' if dataset=='vidstg' else 'HC-STVG')
  old=ROOT/'artifacts/tastvg_paper48_v1'/p['paper48_panel'];oldp=read(old/'PLAN.json');oldpos={parent:at for at,parent in enumerate(oldp['orders']['order1'])};subjects=read(out/'SUBJECT_BARRIER.json');expertparents=read(out/'EXPERT_PLAN.json')['expert_needed'];pixels=0;parity=0
  for parent,row in enumerate(p['rows']):
   frames,ids=decode(row['input']);assert ids==row['frame_ids'];sf=out/'subjects'/f'{parent:05}.json';assert sha(sf)==subjects['files'][sf.name];subject=read(sf)['parses']['subject']
   for cond in p['conditions']:
    budget();rf=out/'capture'/cond/f'{parent:05}.json';shifted,pixel,spec=observation(row,cond,frames)
    if parent in expertparents:
     for stage in ['spatial','temporal']:assert read(out/'experts'/stage/cond/f'{parent:05}.json')['pixel_sha256']==pixel;pixels+=1
    key=hashlib.sha256((row['key']+'|'+pixel+'|'+mh).encode()).hexdigest();cf=out/'H_cache'/f'{key}.pt'
    if not cf.exists():
     data=source_capture(model,shifted,row,subject);data.update(pixel_sha256=pixel,checkpoint_state_sha256=mh);save(cf,data)
    else:data=load(cf)
    assert data['pixel_sha256']==pixel and data['checkpoint_state_sha256']==mh
    oldf=old/'online'/cond/'order1'/f"{oldpos[row['original_parent']]:05}.pt";assert sha(oldf)==read(oldf.with_suffix('.json'))['sha256'];z=load(oldf);assert z['pixel_sha256']==pixel
    assert torch.equal(z['source_native']['boxes'],data['prediction']['boxes']) and z['source_native']['indices']==data['prediction']['indices'];parity+=1
    payload=dict(parent=parent,condition=cond,cache=str(cf.relative_to(out)),sha256=sha(cf),pixel_sha256=pixel,GT_read=False,source_native_parity=True,checkpoint_state_sha256=mh)
    if rf.exists():assert read(rf)==payload
    else:write(rf,payload)
    done+=1
    if done%12==0:status(out/'CAPTURE_STATUS.json',dict(status='running',done=done,total=288,seconds=time.time()-t));print('CAPTURE',dataset,done,288,flush=True)
    del data,z,shifted;gc.collect();torch.cuda.empty_cache()
   assert sum(f.stat().st_size for f in (out/'H_cache').glob('*.pt'))<20*2**30,'H cache exceeded20GiB'
  assert done==parity==288 and state_hash(model.state_dict())==mh
  write(out/'CAPTURE_BARRIER.json',dict(status='completed',cells=done,native_parity=parity,expert_pixel_checks=pixels,checkpoint_state_sha256=mh,files={str(f.relative_to(out)):sha(f) for f in (out/'capture').rglob('*.json')},cache_bytes=sum(f.stat().st_size for f in (out/'H_cache').glob('*.pt')),time=time.time()))
  status(out/'CAPTURE_STATUS.json',dict(status='completed',done=done,total=288,seconds=time.time()-t))
 except BaseException as e:status(out/'CAPTURE_STATUS.json',dict(status='failed',done=done,error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
 finally:
  if lease:lease.close()
if __name__=='__main__':run(sys.argv[1])

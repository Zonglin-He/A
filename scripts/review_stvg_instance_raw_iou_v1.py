"""CPU arithmetic readback of the two newly admitted ordinary IoU fits."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,read,write,sha,guard

def main():
 import torch
 from vg_tta.decota_spatial_opd_tunable_v1 import rollout,mean_coordinates,likelihood_loss
 sys.addaudithook(guard);checks=0;maxerr=0.
 for cid in ['vidstg_000','vidstg_001']:
  p=BASE/'student'/(cid+'.pt');need=read(p.with_suffix('.receipt.json'));assert sha(p)==need['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);f=d['fits']['raw_iou'];ref=d['reference_record']['raw_decoupled']['anchors'];pos=[a['position']for a in ref];ev=torch.tensor([a['box_cxcywh']for a in ref],dtype=torch.float32)
  for e in f['path']:
   saved=e['rollout'];mean=mean_coordinates(e['boxes'][pos]);fresh=rollout(mean,ev,f['seed_key'],e['step'],config=f['config'])
   for k in ['mean','samples','rewards','weights']:
    err=float((fresh[k]-saved[k]).abs().max());maxerr=max(maxerr,err);assert err<2e-5;checks+=fresh[k].numel()
   loss=likelihood_loss(saved['mean'],saved['mean'],saved['samples'],saved['weights'],f['config']['sigma']);assert abs(float(loss)-e['loss'])<2e-5;checks+=1
  assert torch.equal(d['readouts']['raw_iou']['boxes'].cpu(),f['final_cached_decoder_boxes'])
 assert not torch.cuda.is_initialized()
 write(BASE/'ROOT_RAW_IOU_READBACK.json',dict(status='pass',GT_read=False,GPU_used=False,count=2,steps=10,scalar_checks=checks,max_abs_cpu_recompute_error=maxerr,full_final_redecodes_match=2,scope='newly admitted fixed ordinary IoU fits; CPU rewards weights samples and likelihood'))
 print('pass',checks,maxerr)
if __name__=='__main__':main()

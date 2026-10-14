"""Root CPU readback of physical frame, pixel, checkpoint, dtype and feature layout custody."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
B=R/'artifacts/stvg_short_event_tracklets_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def main():
 assert not torch.cuda.is_initialized();bar=read(B/'FEATURE_CAPTURE_BARRIER.json');assert bar['status']=='sealed'and bar['count']==8 and not bar['GT_read'];rt=read(B/'FEATURE_RUNTIME.json');assert bar['feature_runtime_sha256']==sha(B/'FEATURE_RUNTIME.json')
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(R/p)==h,p
 m=read(B/'PREPARED_MANIFEST.json');count=0;rows=[]
 for c in m['cases']:
  cid=c['case_id'];f=bar['files'][cid];p=R/f['path'];assert sha(p)==f['sha256']and p.stat().st_size==f['bytes'];z=torch.load(p,map_location='cpu',weights_only=True);mta=z['metadata'];a=z['appearance_maps'];assert not a.requires_grad and a.device.type=='cpu'and torch.isfinite(a).all();assert a.ndim==4 and a.shape[:2]==(20,256);assert mta['source_model_state_sha256']==c['source_model_state_sha256'];assert mta['source_case_sha256']==c['source_case_sha256'];assert mta['feature_maps_sha256']==hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  expected=sorted({i for w in c['centers']for i in w['physical_frame_ids']});assert z['frame_ids']==expected
  t=read(B/'tracklets'/f'{cid}.json');px=t['frame_readback']['selected_frames'];assert {str(x['physical_frame_id']):x['source_frame_rgb_sha256']for x in px}==mta['pixel_sha256_by_physical_frame'];count+=len(expected)
  rows.append(dict(case_id=cid,frame_count=len(expected),shape=list(a.shape),dtype=str(a.dtype),feature_sha256=f['sha256'],source_sha256=c['source_case_sha256'],checkpoint_sha256=c['source_model_state_sha256'],selected_RGB_exact_with_CPU_tracker=True))
 out=dict(status='pass',count=8,frame_count=count,feature_barrier_sha256=sha(B/'FEATURE_CAPTURE_BARRIER.json'),records=rows,GT_read=False,GPU_used=False,independent_new_feature_forward=False,time=time.time());p=B/'ROOT_FEATURE_READBACK.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('feature root custody pass',count)
if __name__=='__main__':main()

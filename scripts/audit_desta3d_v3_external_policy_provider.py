"""CPU-only full seals, two-loader payload equivalence and physical evidence support."""
import sys,time,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_external_policy_provider_v2 import D,OUT,read,write,sha
from vg_tta.external_qualification_io import verify_seal,tensor_sha

def main(stage):
 import torch,numpy as np
 from vg_tta.external_privileged_views import parse_teacher_text,dense_spatial_support
 from vg_tta.exact_frame_decode_audit_v2 import decode
 start=time.monotonic();run=OUT/('extgate_'+stage);done,seal=verify_seal(run);rows=read(run/'INPUTS.json')
 if stage=='smoke002':
  ph=[read(run/(l+'_LOADING.json')) for l in ['efficient','official']];assert ph[0]['parameter_hashes']==ph[1]['parameter_hashes'];assert all(p['max_frame']==100 for p in ph)
  same=[];payload_checks=0
  for i,row in enumerate(rows):
   raw={}
   for recipe in ['greedy','official']:
    vals=[]
    for loader in ['efficient','official']:
     ep=run/'episodes'/loader/f'{i:02}';r=read(ep/(recipe+'_RAW.json'));f=torch.load(ep/(recipe+'_FEATURES.pt'),map_location='cpu',weights_only=False)
     assert r['decode']==recipe and r['seed']==20260928 and r['max_frame']==100 and r['max_new_tokens']==1024
     assert torch.isfinite(f['first_vision_features']).all();vals.append((r,f))
    assert vals[0][0]==vals[1][0]
    for k in vals[0][1]:assert torch.equal(vals[0][1][k],vals[1][1][k]);payload_checks+=1
    raw[recipe]=vals[0][0]['output_token_ids']
   same.append(raw['greedy']==raw['official'])
  choice='greedy' if all(same) else 'official';assert choice==done['qualification_decode']
  report=dict(status='passed',files=len(seal['files']),all_parameter_hashes_exact=True,payload_checks=payload_checks,qualification_decode=choice,decode_same=same,CPU_seconds=time.monotonic()-start)
  write(run/'ROOT_SMOKE_READBACK.json',report)
 else:
  smoke=OUT/'extgate_smoke002';sm=read(smoke/'ROOT_SMOKE_READBACK.json');assert sm['status']=='passed'
  assert len(rows)==16 and rows==read(D/'INPUTS.json');coverage=[]
  for i,row in enumerate(rows):
   ep=run/'episodes'/f'{i:02}';ident=read(ep/'INPUT.json');r=read(ep/'RAW_PREDICTION.json');ev=read(ep/'EVIDENCE.json')
   frames,ids=decode(row['input']);assert ident['pixel_sha256']==hashlib.sha256(frames.tobytes()).hexdigest() and ids==ident['frame_ids'] and row['key']==ident['key']
   lo,hi=ident['clip_bounds'];slots=np.linspace(lo,hi,100);dist=np.abs(np.asarray(ids)[:,None]-slots);tol=8*np.finfo(float).eps*np.maximum(1,np.maximum(np.abs(ids).max(),np.abs(slots)))
   ix=(dist<=dist.min(0)+tol).argmax(0);assert ix.tolist()==ident['repeated_observation_indices'];assert np.asarray(ids)[ix].tolist()==ident['teacher_pixel_frame_ids']
   assert r['decode']==sm['qualification_decode'];assert parse_teacher_text(r['raw_text'],ids,clip_bounds=[lo,hi])==ev
   _,diag=dense_spatial_support(ids,ev);assert diag==read(ep/'SUPPORT.json')
   if i<2:assert sha(ep/'RAW_PREDICTION.json')==sha(smoke/'episodes/efficient'/f'{i:02}'/(r['decode']+'_RAW.json'))
   coverage.append(dict(query=i+1,format_valid=ev['format_valid'],temporal_usable=ev['temporal_usable'],spatial_usable=ev['spatial_usable'],errors=ev['errors'],possibly_truncated=r['possibly_truncated'],**diag))
  report=dict(status='passed',files=len(seal['files']),queries=16,parents=16,coverage=coverage,GT_read=False,target_read=False,CPU_seconds=time.monotonic()-start)
  write(run/'ROOT_EVIDENCE_READBACK.json',report)
 print('ROOT_PROVIDER_PASSED',stage,len(seal['files']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke002','evidence002']);main(p.parse_args().stage)

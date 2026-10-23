"""Exact prefix vs actual Source observer; same outputs, explicit real paired timing."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
from full import FUSED,read,sha,atomic,identity
from stvg.runtime import setup,lease,load_source
from stvg.visual_prefix import native_appearance_grid
from stvg.representation import extract
from stvg.visual_cache import VisualCache
from stvg.decode import decode
from stvg.native import frozen_native_forward
from stvg.representation import observe_native_grid
from stvg.query import QuerySubjectParser
from stvg.spatial import SpatialGeometry,evidence_dp
import numpy as np,torch
D=R/'outputs/maintenance/native2d_visual_shortcut_20261023'

def main():
 setup();torch.set_num_threads(1);lock=lease();p=read(FUSED/'outputs/native2d_risk_v2/plan.json');parser=QuerySubjectParser(R/'checkpoints/stanza');rows=[];atomic(D/'probe_worker.json',identity());start=time.perf_counter()
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);vc=VisualCache(4*2**30)
   for i in [x for x in p['dev_items']if x['dataset']==ds][:2]:
    frames,ids=decode(i['row']['metadata'],ds);cap=torch.load(i['capture_path'],map_location='cpu',weights_only=False);tr=torch.load(i['trace_path'],map_location='cpu',weights_only=False);expected=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();row=dict(input=i['row']['metadata'],frame_ids=ids,parses={'subject':parser(i['row']['metadata']['caption'])['subject']});model.cuda();torch.cuda.synchronize();tick=time.perf_counter()
    with torch.no_grad(),observe_native_grid(model)as observed:batch,recs,base=frozen_native_forward(model,frames,row)
    torch.cuda.synchronize();full_seconds=time.perf_counter()-tick;assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);del batch,recs,base;dur=[]
    for repeat in range(2):
     torch.cuda.synchronize();tick=time.perf_counter()
     with vc.context(model,frames,ids):grid=native_appearance_grid(model,frames,ids,i['row']['metadata'])
     torch.cuda.synchronize();dur.append(time.perf_counter()-tick)
     for (a,v),(b,w)in zip(grid,observed):assert torch.equal(a,b)and a.dtype==b.dtype and torch.equal(v,w)
     z,_=extract(dict(frame_ids=ids,native_local_grid=grid),aux);assert np.array_equal(z,expected)
    r=dict(dataset=ds,alias=i['alias'],Source_resident_full_seconds=full_seconds,direct_appearance_cold_seconds=dur[0],direct_visual_cache_hit_seconds=dur[1],raw_grids_masks_dtype_and_pooled_features_bitwise=True,source_calls=1,prefix_calls=2,cache=vc.stats());rows.append(r);print(json.dumps(r),flush=True)
   del model;torch.cuda.empty_cache()
  receipt=dict(status='passed',queries=4,rows=rows,code_sha256=sha(Path(__file__)),prefix_code_sha256=sha(R/'stvg/visual_prefix.py'),new_expert=0,new_training=0,new_GT=0,no_parameters_changed=True,timing_boundary='resident Source/newprefix preprocessing+transfers+forward; same decodedRGB, decoder diagnostics kept in Source. Decode/modelload excluded and reported separately in finalfull. Cachehit is cache cost, not coldforward speed.',wall_seconds=time.perf_counter()-start);atomic(D/'prefix_probe.json',receipt)
 finally:lock.close()
if __name__=='__main__':main()

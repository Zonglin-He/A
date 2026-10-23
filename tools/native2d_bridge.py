"""Actual Source/cached-visual observer regression on4fixedqueries, no teacher or labels."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import numpy as np,torch
from full import FUSED,RUN,read,sha,atomic,identity
from stvg.runtime import setup,lease,load_source
from stvg.decode import decode
from stvg.query import QuerySubjectParser
from stvg.native import frozen_native_forward
from stvg.global6 import read_logits,posterior
from stvg.representation import observe_native_grid,extract
from stvg.visual_cache import VisualCache
from stvg.spatial import SpatialGeometry,evidence_dp

def main():
 setup();torch.set_num_threads(1);lock=lease();p=read(FUSED/'outputs/native2d_risk_v1/plan.json');parser=QuerySubjectParser(R/'checkpoints/stanza');started=time.perf_counter();checks=0;Source_calls=0;atomic(RUN/'bridge_worker.json',identity());vc=VisualCache(4*2**30)
 try:
  for ds in ['hc2','vidstg']:
   m=load_source(read(R/'configs/default.json')['datasets'][ds]);items=[i for i in p['dev_items']if i['dataset']==ds][:2]
   for i in items:
    cap=torch.load(i['capture_path'],map_location='cpu',weights_only=False);tr=torch.load(i['trace_path'],map_location='cpu',weights_only=False);expected=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];frames,physical=decode(i['row']['metadata'],ds);assert ids==physical;row=dict(input=i['row']['metadata'],frame_ids=ids,parses={'subject':parser(i['row']['metadata']['caption'])['subject']});aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    for repeat in range(2):
     torch.manual_seed(20260920);np.random.seed(20260920)
     with torch.no_grad(),observe_native_grid(m)as grid,vc.context(m,frames,ids):batch,recs,base=frozen_native_forward(m,frames,row)
     assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);logs,_=read_logits(m.temp_embed,base.temporal_inputs,[x.detach().cpu()for x in base.zero['logits']]);assert all(torch.equal(a,b)for a,b in zip(logs,cap['logits']));q0=posterior(recs,logs,ids);assert np.array_equal(q0,tr['q']['NativeG6']);features,_=extract(dict(frame_ids=ids,native_local_grid=grid),aux);assert np.array_equal(features,expected);checks+=1;Source_calls+=1
    print(ds,i['alias'],'cold/cachehit pooledfeature/Source/alllayers/q0bitwise',flush=True)
   del m;torch.cuda.empty_cache()
  atomic(RUN/'bridge.json',dict(status='passed',queries=4,Source_calls=Source_calls,checks=checks,cold_and_visualcache_hit_bitwise=True,visual=vc.stats(),new_expert=0,new_labels=0,seconds=time.perf_counter()-started,code_sha256=sha(Path(__file__))))
 finally:lock.close()
if __name__=='__main__':main()

"""Actual unmodified Source224 + native-prior Native2D cold/cachehit end-to-end bridge."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'tools')]
import numpy as np,torch
from full import RUN,read,sha,atomic,identity
from stvg.runtime import setup,lease,load_source
from stvg.decode import decode
from stvg.query import QuerySubjectParser
from stvg.native import frozen_native_forward
from stvg.prior import native_prior
from stvg.predictor import Predictor
from stvg.representation import observe_native_grid,extract
from stvg.visual_cache import VisualCache
from stvg.spatial import SpatialGeometry,evidence_dp

def main():
 setup();torch.set_num_threads(1);lock=lease();items=read(RUN/'promotion_inputs.json');parser=QuerySubjectParser(R/'checkpoints/stanza');started=time.perf_counter();checks=0;calls=0;atomic(RUN/'bridge_worker.json',identity());vc=VisualCache(4*2**30)
 try:
  for ds in ['hc2','vidstg']:
   m=load_source(read(R/'configs/default.json')['datasets'][ds])
   for i in [v for v in items if v['dataset']==ds][:2]:
    cap=torch.load(i['path'],map_location='cpu',weights_only=False);tr=torch.load(i['feature_path'],map_location='cpu',weights_only=False);ref=torch.load(i['reference_trace'],map_location='cpu',weights_only=False);expected=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];frames,physical=decode(i['row']['metadata'],ds);assert ids==physical;row=dict(input=i['row']['metadata'],frame_ids=ids,parses={'subject':parser(i['row']['metadata']['caption'])['subject']});aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    for repeat in range(2):
     torch.manual_seed(20260920);np.random.seed(20260920)
     with torch.no_grad(),observe_native_grid(m)as grid,vc.context(m,frames,ids):batch,recs,base=frozen_native_forward(m,frames,row)
     assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);logs=[v.detach().cpu().clone()for v in base.zero['logits']];q0=native_prior(recs,logs,ids);assert np.array_equal(q0,ref['native_prior']);features,_=extract(dict(frame_ids=ids,native_local_grid=grid),aux);assert np.array_equal(features,expected)
     out=Predictor(ds).finish(dict(Source=cap['Source'],E=cap['E'],geometry=g,frame_ids=ids,native_local_grid=grid,native_logits=logs,records=recs,posterior={'Native':q0},intervals={'Native':cap.get('native_interval',[])},planned_positions=cap.get('positions8',cap.get('planned_positions')),cost={},initialization={}))
     assert np.array_equal(out['posterior'],ref['NativeFeedback']);assert not out['cost']['Global6_aggregation']and out['cost']['additional_time_head_forwards']==0
     with np.load(i['reference_prediction'],allow_pickle=False)as z:assert np.array_equal(out['boxes'],z['NativeFeedback'])and np.array_equal(out['interval'],z['NativeFeedback_interval'])
     checks+=1;calls+=1;del batch,base
    print(ds,'cold/cachehit rawSource/nativeprior/features/finalposterior/boxes exact',flush=True)
   del m;torch.cuda.empty_cache()
  out=dict(status='passed',queries=4,Source_calls=calls,checks=checks,cold_and_visualcache_hit_bitwise=True,native_prior_Native2D_end_to_end_bitwise=True,no_Global6_head_or_aggregation=True,visual=vc.stats(),new_expert=0,new_labels=0,seconds=time.perf_counter()-started,code_sha256=sha(Path(__file__)));atomic(RUN/'bridge.json',out);atomic(R/'results/native2d_nativeprior_full_v1/BRIDGE.json',out)
 finally:lock.close()
if __name__=='__main__':main()

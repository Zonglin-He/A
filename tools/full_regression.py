"""CPU-only bitwise promotion proof against every sealed selected-method output."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from full import K16,OLD,RUN,PUBLIC,read,sha,atomic,check,identity
import numpy as np,torch
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.global6 import posterior
from stvg.paper_panel import derive
from stvg.spatial import SpatialGeometry,evidence_dp

def main():
 p=check(True);torch.set_num_threads(1);start=time.perf_counter();count=0;counts={}
 atomic(RUN/'regression_worker.json',identity())
 source=OLD/'outputs/target_semantic128_collect_v3';ss=read(source/'sealed.json')
 for stage in ['general_region_v1','general_region_recheck_v3']:
  folder=K16/'outputs'/stage;study=read(folder/'plan.json');seal=read(folder/'sealed.json')
  for item in study['items']:
   key=item['alias'];entry=seal['files'][key]
   if stage=='general_region_v1':
    raw=ss['files'][key];path=source/raw['capture_path'];h=raw['capture_sha256']
   else:path=folder/entry['capture_path'];h=entry['capture_sha256']
   assert sha(path)==h and sha(folder/entry['trace_path'])==entry['trace_sha256'] and sha(folder/entry['path'])==entry['sha256']
   cap=torch.load(path,map_location='cpu',weights_only=False);old=torch.load(folder/entry['trace_path'],map_location='cpu',weights_only=False)
   g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids']
   aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
   features,_=extract(cap,aux);assert np.array_equal(features,old['features']['RegionKernel']),(key,'feature')
   q0=posterior(cap['records'],cap['logits'],ids);q,feedback=one_step(features,q0)
   assert np.array_equal(q,old['q']['RegionKernel']),(key,'posterior')
   cap['content_features']=features;arr,_=derive(cap)
   with np.load(folder/entry['path'],allow_pickle=False)as z:
    for a,b in [('Full','RegionKernel'),('Full_interval','RegionKernel_interval'),('Global6','NativeG6'),('Global6_interval','NativeG6_interval'),('Source','Source'),('Source_interval','Source_interval'),('frame_ids','frame_ids'),('positions8','positions8')]:
     assert np.array_equal(arr[a],z[b]),(key,a)
   assert cap['GT_read']is False;count+=1;counts[item['dataset']]=counts.get(item['dataset'],0)+1
   atomic(RUN/'regression_progress.json',dict(completed=count,total=320,seconds=time.perf_counter()-start))
   del cap,old,g,features,q0,q,arr
 assert count==320
 # Frozen spatial and native Source paths are unchanged from old production.
 untouched=['spatial.py','initialization.py','evidence.py','native.py','temporal.py','temporal_hidden.py','global6.py','positions.py','source_loader.py']
 for name in untouched:assert sha(ROOT/'stvg'/name)==sha(K16/'stvg'/name),name
 out=dict(status='passed',queries=count,counts=counts,features_posterior_interval_finalboxes_bitwise=True,Source_Global6_frameids_positions_bitwise=True,unchanged_spatial_and_Source=untouched,new_GPU_model_expert_GT_calls=0,GT_read=False,seconds=time.perf_counter()-start,plan_sha256=sha(RUN/'plan.json'),code_sha256=sha(Path(__file__)))
 atomic(RUN/'regression.json',out);atomic(PUBLIC/'REGRESSION.json',out);print(out)
if __name__=='__main__':main()

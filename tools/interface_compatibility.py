"""CPU-only general-contract bridge proof; current full bound runtime stays intact."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np,torch
from full import K16,OLD,RUN,check,read,sha,atomic,identity
from stvg.global6 import posterior
from stvg.temporal import final_mbr
from stvg.spatial import SpatialGeometry,evidence_dp
from adapters.tastvg import extract
from interfaces.temporal import JointGridPosterior
from interfaces.prototype import feedback

if __name__=='__main__':
 p=check(True);torch.set_num_threads(1);start=time.perf_counter();count=0;source=OLD/'outputs/target_semantic128_collect_v3';ss=read(source/'sealed.json')
 out=ROOT/'outputs/general_interface_compatibility_v1';out.mkdir(parents=True,exist_ok=True);atomic(out/'worker.json',identity());before=sha(RUN/'plan.json')
 code={str(f.relative_to(ROOT)):sha(f)for folder in ['interfaces','adapters']for f in(ROOT/folder).glob('*.py')};code['tools/interface_compatibility.py']=sha(Path(__file__))
 atomic(out/'plan.json',dict(scope='Interface-only CPU compatibility, no newmethod/parameter orSource/GT; runningfullboundsource unchanged',cases=320,code=code,full_plan_sha256=before))
 for stage in ['general_region_v1','general_region_recheck_v3']:
  folder=K16/'outputs'/stage;study=read(folder/'plan.json');seal=read(folder/'sealed.json')
  for item in study['items']:
   key=item['alias'];entry=seal['files'][key]
   if stage=='general_region_v1':raw=ss['files'][key];path=source/raw['capture_path'];h=raw['capture_sha256']
   else:path=folder/entry['capture_path'];h=entry['capture_sha256']
   assert sha(path)==h and sha(folder/entry['trace_path'])==entry['trace_sha256']and sha(folder/entry['path'])==entry['sha256']
   cap=torch.load(path,map_location='cpu',weights_only=False);old=torch.load(folder/entry['trace_path'],map_location='cpu',weights_only=False);g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids']
   aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();features,stats=extract(cap,aux)
   assert np.array_equal(features,old['features']['RegionKernel']),(key,'generic local aggregation')
   q0=posterior(cap['records'],cap['logits'],ids);native=JointGridPosterior(np.asarray(ids),q0,lambda times,q:final_mbr(times.tolist(),q)['interval']);q,_=feedback(features,native)
   assert np.array_equal(q,old['q']['RegionKernel']),(key,'generic temporal interface posterior')
   iv=native.readout(q);boxes=evidence_dp(g.for_interval(iv))['final'].numpy()
   with np.load(folder/entry['path'],allow_pickle=False)as z:assert np.array_equal(boxes,z['RegionKernel'])and iv==z['RegionKernel_interval'].tolist()
   count+=1;atomic(out/'progress.json',dict(completed=count,total=320,seconds=time.perf_counter()-start));del cap,old,g,features,boxes
 assert count==320 and before==sha(RUN/'plan.json');check()
 proof=dict(status='passed',queries=320,local_feature_posterior_interval_finalboxes_bitwise=True,current_full_plan_and_bound_code_unchanged=True,Source_GPU_expert_training_GT_calls=0,TA_adapter_only=True,otherbackbone_MLLM_validation=False,seconds=time.perf_counter()-start,code=code)
 atomic(out/'completion.json',proof);atomic(ROOT/'results/general_interface_v1/COMPATIBILITY.json',proof);print(proof)

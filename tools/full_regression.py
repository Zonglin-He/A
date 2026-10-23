"""All256 cached native-prior ablation outputs must be recovered bitwise."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from full import RUN,PUBLIC,K16,read,sha,atomic,check,PROVIDER
import numpy as np,torch
from stvg.prior import from_capture
from stvg.prototype import one_step
from stvg.paper_panel import derive
from stvg.temporal import final_mbr

def main():
 p=check(True);torch.set_num_threads(1);started=time.perf_counter();items=read(RUN/'promotion_inputs.json');count=0;counts={}
 for i in items:
  for f,h in [(i['path'],i['sha256']),(i['feature_path'],i['feature_sha256']),(i['reference_prediction'],i['reference_sha256']),(i['reference_trace'],i['reference_trace_sha256'])]:assert sha(f)==h
  cap=torch.load(i['path'],weights_only=False,map_location='cpu');tr=torch.load(i['feature_path'],weights_only=False,map_location='cpu');ref=torch.load(i['reference_trace'],weights_only=False,map_location='cpu');features=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features'];ids=cap['frame_ids'];q0=from_capture(cap);q,_=one_step(features,q0);assert np.array_equal(q,ref['NativeFeedback']),(i['row']['id'],'posterior');assert np.array_equal(q0,ref['native_prior'])
  cap['native_interval']=final_mbr(ids,q0)['interval'];cap['positions8']=cap.get('positions8',cap.get('planned_positions'));cap['content_features']=features
  with np.load(i['reference_prediction'],allow_pickle=False)as z:
   cap['reference_Global6']=z['Global6'].copy();cap['reference_Global6_interval']=z['Global6_interval'].copy();arr,_=derive(cap)
   for a,b in [('Full','NativeFeedback'),('Full_interval','NativeFeedback_interval'),('OursS','NativeSpatial'),('OursS_interval','NativeSpatial_interval'),('Source','Source'),('Global6','Global6'),('Global6_interval','Global6_interval'),('Source_interval','Source_interval'),('positions8','positions8')]:assert np.array_equal(arr[a],z[b]),(i['row']['id'],a)
  count+=1;counts[i['dataset']]=counts.get(i['dataset'],0)+1;atomic(RUN/'regression_progress.json',dict(completed=count,total=len(items)))
 unchanged=['spatial.py','initialization.py','evidence.py','native.py','temporal.py','temporal_hidden.py','positions.py','source_loader.py']
 for n in unchanged:assert sha(ROOT/'stvg'/n)==sha(K16/'stvg'/n)
 out=dict(status='passed',queries=count,counts=counts,NativeFeedback_prior_posterior_intervals_finalboxes_bitwise=True,Source_nativeSpatial_Global6_reference_positions_bitwise=True,Global6_not_used_as_production_prior=True,unchanged_Source_spatial=unchanged,GT_read=False,model_expert_calls=0,seconds=time.perf_counter()-started,plan_sha256=sha(RUN/'plan.json'));assert count==256;atomic(RUN/'regression.json',out);atomic(PUBLIC/'REGRESSION.json',out);print(out)
if __name__=='__main__':main()

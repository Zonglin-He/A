"""CPU-only Native2D promotion: every256 cached selected output is reproduced bitwise."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from full import RUN,PUBLIC,K16,read,sha,atomic,check
import numpy as np,torch
from stvg.prototype import one_step
from stvg.global6 import posterior
from stvg.paper_panel import derive
from stvg.temporal import probabilities,mass_table,final_mbr

def main():
 p=check(True);torch.set_num_threads(1);started=time.perf_counter();items=read(RUN/'promotion_inputs.json');count=0;counts={}
 for i in items:
  assert sha(i['path'])==i['sha256']and sha(i['feature_path'])==i['feature_sha256']and sha(i['reference_prediction'])==i['reference_sha256']and sha(i['reference_trace'])==i['reference_trace_sha256']
  cap=torch.load(i['path'],weights_only=False,map_location='cpu');tr=torch.load(i['feature_path'],weights_only=False,map_location='cpu');ref=torch.load(i['reference_trace'],weights_only=False,map_location='cpu');features=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features'];ids=cap['frame_ids'];q0=posterior(cap['records'],cap['logits'],ids);q,_=one_step(features,q0);assert np.array_equal(q,ref['q'][i['reference_q']]),(i['row']['id'],'posterior')
  if'native_interval'not in cap:
   U,nq=mass_table(probabilities(cap['records'],[v[-1]for v in cap['logits']]));assert U==ids;cap['native_interval']=final_mbr(ids,nq)['interval']
  cap['positions8']=cap.get('positions8',cap.get('planned_positions'));cap['content_features']=features;arr,_=derive(cap)
  with np.load(i['reference_prediction'],allow_pickle=False)as z:
   for a,b in [('Full',i['reference_arm']),('Full_interval',i['reference_arm']+'_interval'),('Source','Source'),('Global6','NativeG6'),('Global6_interval','NativeG6_interval'),('Source_interval','Source_interval'),('positions8','positions8')]:assert np.array_equal(arr[a],z[b]),(i['row']['id'],a)
  count+=1;counts[i['dataset']]=counts.get(i['dataset'],0)+1;atomic(RUN/'regression_progress.json',dict(completed=count,total=len(items)))
 unchanged=['spatial.py','initialization.py','evidence.py','native.py','temporal.py','temporal_hidden.py','global6.py','positions.py','source_loader.py']
 for n in unchanged:assert sha(ROOT/'stvg'/n)==sha(K16/'stvg'/n)
 out=dict(status='passed',queries=count,counts=counts,all_posteriors_intervals_Source_Global6_finalboxes_bitwise=True,unchanged_Source_spatial=unchanged,GT_read=False,model_expert_calls=0,seconds=time.perf_counter()-started,plan_sha256=sha(RUN/'plan.json'));assert count==p['regression_queries'];atomic(RUN/'regression.json',out);atomic(PUBLIC/'REGRESSION.json',out);print(out)
if __name__=='__main__':main()

"""Root-only first two paid-input repooling audit; never create an ack by hand."""
from common_v1 import *
import numpy as np,torch
from full_core_v1 import F,build,replay
torch.set_num_threads(1)
def run(ds):
 infer_guard();p=read(F/'plan_v1.json');dest=F/ds;wait=read(dest/'bridge_waiting.json');assert len(wait['keys'])==2 and not(dest/'root_actual_GPU_bridge.json').exists();by={r['id']:r for r in selected.records(ds)};checks=[]
 for key in wait['keys']:
  e=read(dest/'receipts'/(key+'.json'));rawfile=dest/'active'/key/'Source.pt';assert sha(rawfile)==e['raw_origin']['sha256'];raw=torch.load(rawfile,map_location='cpu',weights_only=False,mmap=True);cap,par=selected.suite().load_parent(ds,by[key]);assert par==e['parent']and np.array_equal(raw['Source'],cap['Source'])and all(torch.equal(x,y)for x,y in zip(raw['native_logits'],cap['native_logits']))and raw['records']==cap['records']
  from stvg.evidence import extract_visual_grids
  f,v,ids,shape,_=extract_visual_grids(raw['spatial_inputs'],raw['frame_ids']);assert torch.equal(f,raw['spatial_features'])and torch.equal(v,raw['spatial_valid'])
  a,c,proof,diag=build(ds,by[key],cap,raw);assert proof==e['proof']and diag==e['diagnostics'];b=replay(c)
  with np.load(F/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(x,z[k])and np.array_equal(x,b[k])for k,x in a.items())
  checks.append(dict(raw_sha256=sha(rawfile),prediction_sha256=e['sha256'],compact_sha256=e['compact_sha256'],actual_new_raw_repool=True,postfusion_grid_rederived_exact=True,native_Source_records_logits_exact=True,all_arrays_double_cold_exact=True))
 atomic(dest/'root_actual_GPU_bridge.json',dict(status='passed_actual_two_new_Source_bridge',plan_fingerprint=p['fingerprint'],cases=checks,code_sha256=sha(Path(__file__)),worker=identity(),new_Source=0,new_expert=0,new_model=0,new_GT_scoring=0));print(ds,'actual paid bridge passed')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=COUNTS,required=True);run(ap.parse_args().dataset)


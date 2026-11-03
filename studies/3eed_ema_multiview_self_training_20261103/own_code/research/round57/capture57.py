"""Third independent meaning-preserving point sample; full source remains frozen."""
import argparse,sys,time,torch
from common57 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['capture_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();rows=read(ROOT/p['manifest']);model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);start=time.perf_counter();completed=0
 try:
  for row in rows:
   r=dict(row,seed=seed(row));xyz,ix,nraw=sample_points(r);z,rec=source_capture(model,xyz,ix,row['caption']);z={k:z[k] for k in ['boxes','scores','h','query','tokens','token_mask','point_indices']};z['raw_point_count']=np.array(nraw,np.int64);np.savez_compressed(out/'samples_PRIVATE'/(row['id']+'.npz'),**z);dump(out/'cases_PRIVATE'/(row['id']+'.json'),{**rec,'seed':r['seed'],'raw_point_count':nraw});completed+=1
   if completed%32==0:print('CAPTURE57',completed,336,flush=True)
  assert fingerprint(model)==fp
  costs={'n':completed,'new_full_grounder_forwards':completed,'source_fingerprint':fp,'source_unchanged':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',costs);seal(out,'R57_THIRD_REAL_SOURCE_SAMPLING_NO_GT',**costs)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed':completed,'GT_access':False});raise
if __name__=='__main__':main()

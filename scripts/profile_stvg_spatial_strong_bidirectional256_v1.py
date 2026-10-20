"""One actual cold complete single-seed inference per direction, no GT."""
from pathlib import Path
import sys,time,gc,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_spatial_strong_bidirectional256_v1 import D,P,read,put,load,verify,source_for,capture,sha

def main():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,parse
 from methods.stvg_referdino_opd_v1.predictor import ReferDINOOPDPredictor
 from scripts.run_decota_paper_main_v1 import frames_for
 from main_method.runtime import resource_guard
 setup();p=verify();lock=lease();out={}
 try:
  for ds in ['hc2','vidstg']:
   rec=next(r for r in p['records']if r['dataset']==ds);a=rec['alias'];row=dict(read(R/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'][rec['query_ordinal']]);start=time.perf_counter();parses,parse_s=parse(row['input']['caption']);assert parses==read(R/rec['subject_path'])['parses'];row['parses']=parses;tick=time.perf_counter();frames,ids,_=frames_for(ds,row,collections.OrderedDict());decode_s=time.perf_counter()-tick;tick=time.perf_counter();m=source_for(ds);load_s=time.perf_counter()-tick
   cache=capture(m,frames,ids,row,p['configs'][ds],resource_guard);predictor=ReferDINOOPDPredictor(config=p['configs'][ds],source=m,check=resource_guard);tick=time.perf_counter();adapt=predictor.adapt(cache,key=a,arms=('Live',),seeds=[41]);torch.cuda.synchronize();adapt_s=time.perf_counter()-tick;total=time.perf_counter()-start;peak=max(cache['cost']['peak_GPU_bytes'],torch.cuda.max_memory_allocated());old=load(D/'predictions'/ds/f'{a}.pt');errors=dict(Source=float(abs(old['Source']-cache['Source'].numpy()).max()),E=float(abs(old['E']-cache['E']).max()),Live41=float(abs(old['Live'][41]-adapt['outputs']['Live_41']).max()),Strong=float(abs(old['posterior']['Strong']-cache['posterior']['Strong']).max()));assert max(errors.values())==0,errors
   out[ds]=dict(alias=a,actual_complete_single_seed_wall_seconds=total,includes_text_parse=True,parse_seconds=parse_s,decode_seconds=decode_s,Source_load_seconds=load_s,capture=cache['cost'],adapt_seconds=adapt_s,updates=adapt['cost']['actual_updates'],peak_GPU_bytes=peak,seed=41,deterministic_output_errors=errors,GT_read=False,scope='first prelocked case per direction; not latency macro over128, cold expert subprocess load charged')
   put(D/'COMPLETE_SINGLE_SEED_PROFILE.json',out);print('PROFILE',ds,total,peak,flush=True);del frames,cache,adapt,predictor,m,old;gc.collect();torch.cuda.empty_cache()
  put(P/'COMPLETE_SINGLE_SEED_PROFILE_PUBLIC.json',out)
 finally:lock.close()
if __name__=='__main__':main()

"""Two fixed first-case full-input checks of the selected single-expert runtime."""
import os,sys,time,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE,read,put,sha,load,save,verify

def main(ds):
 import numpy as np,torch
 from methods.stvg_single_referdino_opd_v1.runtime import setup,lease
 from methods.stvg_single_referdino_opd_v1.tuned_gamma import InitializationTunedSingleReferDINOOPDPredictor
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.run_decota_paper_main_v1 import frames_for
 import scripts.run_stvg_single_referdino_bidirectional256_v1 as guard
 from main_method.runtime import resource_guard
 RD=D/'rounding';RP=P/'rounding'
 setup();sys.addaudithook(guard.deny_independent_detector);lock=lease();p=verify();rec=next(x for x in p['records']if x['dataset']==ds);alias=rec['alias'];cfg=read(RP/'SELECTED_CONFIGS_PUBLIC.json')[ds];prior=sum(v['wall_seconds']for v in read(D/'GPU_LEDGER.json')['invocations']);beg=time.perf_counter();failure=None;updates=0
 def check():resource_guard();assert prior+time.perf_counter()-beg<7200
 try:
  expected=load(BASE/f'predictions/{ds}/{alias}.pt');teacher=load(BASE/f'inputs/{ds}/{alias}.pt')['teacher_native'];raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'][rec['query_ordinal']];tick=time.perf_counter();frames,ids,_=frames_for(ds,raw,collections.OrderedDict());decode_s=time.perf_counter()-tick;predictor=InitializationTunedSingleReferDINOOPDPredictor(ds,config=cfg,check=check);torch.cuda.reset_peak_memory_stats();guard.PROFILE_ACTIVE=True;tick=time.perf_counter()
  try:fresh=predictor.capture(frames,ids,raw['input']);out=predictor.adapt(fresh,alias);torch.cuda.synchronize();wall=time.perf_counter()-tick+decode_s
  finally:guard.PROFILE_ACTIVE=False
  updates=out['cost']['actual_updates'];assert updates==cfg['steps'];assert np.array_equal(fresh['Source'].numpy().astype(float),expected['Source']);
  with np.load(RD/'selected_predictions'/ds/f'{alias}.npz')as z:assert np.array_equal(fresh['E'],z['E_Ref'])
  assert torch.equal(fresh['teacher_native']['boxes'],teacher['boxes']);assert all(np.array_equal(fresh['posterior'][k],expected['posterior'][k])and fresh['intervals'][k]==expected['intervals'][k]for k in ['Native','Strong'])
  with np.load(RD/'selected_predictions'/ds/f'{alias}.npz')as z:assert np.array_equal(z['Live'],out['Live'])and z['frame_ids'].tolist()==ids
  assert fresh['teacher_native']['load_audit']['independent_GroundingDINO_checkpoint_loaded']is False;assert all(not x.requires_grad and x.grad is None for x in predictor.source.parameters());save(RD/f'profiles/{ds}_fit.pt',out['fit']);profile=dict(status='pass',alias=alias,configuration=cfg,expert_count=1,complete_single_query_wall_seconds=wall,decode_seconds=decode_s,capture=fresh['cost'],adapt=out['cost'],peak_GPU_bytes=max(torch.cuda.max_memory_allocated(),fresh['cost']['peak_GPU_bytes']),selected_output_bitexact=True,Source_ERef_teacher_Strong_exact=True,historical_detector_or_prediction_cache_reads_during_full_pipeline=0,independent_GroundingDINO_calls=0,not_panel_average=True,GT_read=False);put(RP/f'SELECTED_COMPLETE_PROFILE_{ds}_PUBLIC.json',profile);print('TUNED_COMPLETE_PROFILE_PASS',ds,wall,updates,flush=True)
 except BaseException as e:failure=repr(e);raise
 finally:
  led=read(D/'GPU_LEDGER.json');led['invocations'].append(dict(dataset=ds,phase='selected_complete_profile',wall_seconds=time.perf_counter()-beg,new_primary_fits=0,new_updates=updates,compatibility_updates=0,verification_updates=updates,Source_load_seconds=0,peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_RGB_or_Source_encoding_forwards=2,new_Source_offset_forwards=4,new_expert_forwards=1,failure=failure));put(D/'GPU_LEDGER.json',led);lock.close()
if __name__=='__main__':main(sys.argv[1])

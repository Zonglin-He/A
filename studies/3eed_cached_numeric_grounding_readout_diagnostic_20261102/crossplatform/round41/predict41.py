"""Approved all128 cached five-arm prediction seal; absolutely no task scoring."""
from common41 import *
import time

def main():
 check_authorization();check_states_and_inputs();rr=U_rows();OUT.mkdir(exist_ok=False)
 for sub in ['predictions_PRIVATE','diagnostics_PRIVATE']:(OUT/sub).mkdir()
 start=time.perf_counter();records=[]
 for r in rr:
  oldfile=OLD/'predictions_PRIVATE'/(r['id']+'.npz');sourcefile=SOURCE/'samples_PRIVATE'/(r['id']+'.npz');c=npz(oldfile);z=npz(sourcefile);assert np.array_equal(c['probability_scores'][0],z['scores']);assert np.array_equal(c['boxes'],z['boxes'][c['queries']]);assert np.array_equal(c['queries'],c['probability_scores'].argmax(-1));v=readouts_from_existing_U_cache(c);q=v['queries'];boxes=z['boxes'][q];assert q.shape==(5,) and boxes.shape==(5,6)
  np.savez_compressed(OUT/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),scores=v['scores'],queries=q,boxes=boxes,stable_logmass=v['stable_logmass'],original_R35_queries=c['queries'],original_R35_probability_scores=c['probability_scores'],original_R35_exact_token_logodds=c['exact_token_logodds'],source_probability_projection_replay=c['source_probability_projection_replay'])
  rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'numerical_diagnostics':v['numerical_diagnostics'],'source_arithmetic_bridge':v['source_arithmetic_bridge'],'no_readout_query_change_for_either_weight':v['no_readout_query_change_for_either_weight'],'original_R35_file':str(oldfile.relative_to(ROOT)),'original_R35_sha256':sha(oldfile),'original_source_file':str(sourcefile.relative_to(ROOT)),'original_source_sha256':sha(sourcefile),'native_R30_original_query_exact':int(q[2])==int(c['queries'][1]),'source_historical_original_query_exact':int(q[4])==int(c['queries'][0]),'no_GT_or_new_model_or_optimizer':True};assert rec['native_R30_original_query_exact'] and rec['source_historical_original_query_exact'];dump(OUT/'diagnostics_PRIVATE'/(r['id']+'.json'),rec);records.append(rec)
 dump(OUT/'ALL_DIAGNOSTICS_PRIVATE.json',records);seal(OUT,'R41_all128_cached2x2_plus_sourcebridge_before_task_diagnostic',n=128,arms=list(ARMS),new_GPU_VLM_optimizer_steps=0,root_authorization_sha256=sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),implementation_sha256=sha(EXP/'IMPLEMENTATION_LOCK.json'),original_R35_seal_sha256=sha(OLD/'SEALED.json'),original_sourceU_seal_sha256=sha(SOURCE/'SEALED.json'),within_weight_readout_query_unchanged_all128=proposed_stop_rule(records),seconds=time.perf_counter()-start);print('R41_PREDICTIONS_SEALED',proposed_stop_rule(records))
if __name__=='__main__':main()

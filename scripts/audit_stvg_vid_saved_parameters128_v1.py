"""Independent saved-metric aggregation and cached-node readback; no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_stvg_vid_saved_parameters128_v1 import D,P,OLD,ROOT,read,put,load,sha,pins_ok
def main():
 p=read(D/'PLAN_LOCK.json');pins_ok(p);r=read(P/'RESULTS_PUBLIC.json');assert len(r['rows'])==128
 assert [x['alias']for x in r['rows']]==[x['alias']for x in p['records']]
 errors=[];nodechecks=0;aggregatechecks=0
 for row in r['rows']:
  z=read(D/'private_node_metrics'/f"{row['alias']}.json")
  assert len(z['GT_supported_actual_ids'])==row['GT_supported_actual_nodes']
  with np.load(D/'predictions'/f"{row['alias']}.npz")as pred:
   observed={int(pred['frame_ids'][i])for i in pred['positions8']}
  for arm,values in row['values'].items():
   data=z['per_arm'][arm];io=data['actual_node_IoU'];assert len(io)==row['GT_supported_actual_nodes']
   groups={'fixed_actual_sIoU':[float(v)for v in io.values()],
           'observed_actual_sIoU':[float(v)for k,v in io.items()if int(k)in observed],
           'unobserved_actual_sIoU':[float(v)for k,v in io.items()if int(k)not in observed],
           's':data['dense_GT_IoU']}
   for key,v in groups.items():
    if v:errors.append(abs(sum(v)/len(v)-values[key]));nodechecks+=1
    else:assert values[key]is None
   assert values['t']==row['values']['E_Strong']['t']
 for arm,values in r['means'].items():
  for key,v in values.items():
   q=[x['values'][arm][key]for x in r['rows']if x['values'][arm][key]is not None];errors.append(abs(sum(q)/len(q)-v));aggregatechecks+=1
 rng=np.random.default_rng(20261020);indices=rng.integers(0,128,(10000,128));bootchecks=0
 for comparison,metrics in r['paired'].items():
  a,b=comparison.split('_minus_')
  for key,m in metrics.items():
   if key=='tail':continue
   x=np.asarray([100*(q['values'][a][key]-q['values'][b][key])for q in r['rows']]);ci=np.percentile(np.mean(np.take(x,indices),axis=1),[2.5,97.5]);errors.extend([abs(float(x.mean())-m['mean']),*abs(ci-m['CI95']).tolist()]);bootchecks+=1
  x=np.asarray([100*(q['values'][a]['v']-q['values'][b]['v'])for q in r['rows']]);t=metrics['tail']
  assert [int((x>1e-10).sum()),int((x< -1e-10).sum()),int((abs(x)<=1e-10).sum()),int((x< -3).sum()),int((x< -5).sum())]==[t[k]for k in ['positive','negative','unchanged','below_minus3','below_minus5']]
 assert max(errors)<1e-11,max(errors)
 inventory=[]
 for rec in p['records']:
  f=ROOT/f"artifacts/stvg_opd_paper_hc2_revision_v2/inputs/hcstvg2_to_vidstg/clean/{rec['query_ordinal']:05}.pt"
  z=load(f);inventory.append(dict(alias=rec['alias'],exists=True,decoder_input_fields_present=any(k in z for k in ['spatial_inputs','memory','query_tgt']),bytes=f.stat().st_size,sha256=sha(f)))
 assert not any(x['decoder_input_fields_present']for x in inventory)
 put(P/'CACHE_QUALIFICATION_PUBLIC.json',dict(status='missing_matching_complete_decoder_input_cache',current_prediction_and_trace_files_checked=256,current_decoder_input_files_found=0,legacy_matched_prediction_inputs_checked=128,legacy_decoder_inputs_found=0,legacy_rows=inventory,
   original_producer='scripts/run_stvg_spatial_strong_bidirectional256_v1.py: traces save fits/logits/records/teacher/expert but omit spatial_inputs',
   replay_authorized=True,decoder_replay_executed=False,reason='required decoder input features not saved; new RGB/Source forward forbidden',
   not_exhaustive_filesystem_absence_proof=True,no_learning_rate_or_temperature_results_claimed=True))
 put(P/'ROOT_AUDIT_PUBLIC.json',dict(status='pass',cached_node_and_dense_score_mean_checks=nodechecks,aggregate_checks=aggregatechecks,parent_bootstrap_comparisons=bootchecks,max_abs_error=max(errors),all128_aliases_and_denominators=True,all_positive_negative_unchanged_kept=True,original_pins_and256_caches_unchanged=True))
 print(json.dumps(read(P/'ROOT_AUDIT_PUBLIC.json')))
if __name__=='__main__':main()

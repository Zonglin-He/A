"""Preserve exact signs while exposing coordinate-roundtrip and real clock scopes."""
from common import *
import numpy as np,argparse

def run(ds):
 validate();gt_guard();start=time.perf_counter();aud=read(D/ds/'reporting_receipt.json');assert aud['all_arrays_rederived_exact'];seal=read(D/ds/'sealed.json');rr=read(D/ds/'metrics.json')['private_rows'];delta=np.asarray([100*(r['values']['ClipToImage']['v']-r['values']['FullMAP']['v'])for r in rr]);inside_changed=0;inside_frames=0;maxcoord=0.
 for e in seal['files'].values():
  assert sha(D/e['path'])==e['sha256']
  with np.load(D/e['path'])as z:
   b=z['FullMAP'];lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;inside=(lo>=0).all(1)&(hi<=1).all(1);diff=np.abs(z['ClipToImage']-b);inside_frames+=int(inside.sum());inside_changed+=int((inside&(diff>0).any(1)).sum());maxcoord=max(maxcoord,float(diff[inside].max())if inside.any()else 0.)
 numerical=dict(strict_negative_v_queries=int((delta<0).sum()),largest_negative_v_pp=float(max(0.,-delta.min())),negative_below_minus_1e_10pp=int((delta< -1e-10).sum()),sub_1e_10pp_negative_queries=int(((delta<0)&(delta>=-1e-10)).sum()),all_strict_counts_and_exact_array_hashes_retained=True,no_tolerance_or_score_changed=True,in_image_native_frames=inside_frames,in_image_frames_with_coordinate_roundtrip_difference=inside_changed,max_in_image_coordinate_roundtrip_absolute_difference=maxcoord,epsilon_only_describes_magnitude_not_filtering=True)
 cost=dict(new_model=0,new_Source=0,new_expert=0,new_optimizer=0,conditional_predict_elapsed_wall_seconds=seal['elapsed_wall_seconds'],official_score=read(D/ds/'score_receipt.json'),separate_audit=aud,original_acquisition_cost_nonzero=True,nested_not_additive=True,uncached_deployment=False,actual_root_worker=identity(),root_elapsed_wall_seconds=time.perf_counter()-start,study_private_bytes_at_root=sum(f.stat().st_size for f in D.rglob('*')if f.is_file()),private_scope_limit_bytes=160*2**30)
 assert cost['study_private_bytes_at_root']<cost['private_scope_limit_bytes']
 if ds=='vidstg':
  preservation=read(D/'preserved_serial_runtime_v1/preservation.json');ticks=os.sysconf('SC_CLK_TCK');cpu={}
  for pid,stat in preservation['proc_stats'].items():
   x=stat.split(') ',1)[1].split();cpu[pid]=dict(user_seconds=int(x[11])/ticks,system_seconds=int(x[12])/ticks,startticks=int(x[19]))
  cost.update(original_partial_CPU_work_preserved=preservation['partial_predictions'],original_partial_actual_CPU_times=cpu,old_intentional_exit=read(D/'preserved_serial_runtime_v1/owned_exit.json'),new_v2_prediction_reused_exact=seal['existing_serial_predictions_reused_exact'],new_v2_remaining_CPU_predictions=seal['new_remaining_CPU_predictions'],cache_preflight=read(D/'cache_preflight_v2.json'),original_partial_wall_and_outer_wait_code_missing_not_zero=True,execution_v2_fingerprint=read(D/'execution_v2_plan.json')['fingerprint'],actual_stage_exit_codes={p.stem:read(p)['code']for p in D.glob('vidstg_*_exit_v2.json')})
 atomic(P/ds/'coordinate_roundtrip_diagnostics_v2.json',numerical);atomic(P/ds/'actual_cost_lineage_v2.json',cost);atomic(D/ds/'root_numeric_cost_receipt_v2.json',dict(status='verified_exact_signed_counts_roundtrip_magnitude_and_actual_CPU_model_cost_scope',code_sha256=sha(Path(__file__)),audit_sha256=sha(D/ds/'reporting_receipt.json'),cost_sha256=sha(P/ds/'actual_cost_lineage_v2.json'),diagnostics_sha256=sha(P/ds/'coordinate_roundtrip_diagnostics_v2.json'),new_GT_scoring=0));print(json.dumps(dict(dataset=ds,numerical=numerical)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);run(p.parse_args().dataset)

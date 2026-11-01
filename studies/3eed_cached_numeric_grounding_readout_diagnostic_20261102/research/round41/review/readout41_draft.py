"""R41 proposal only. Pure cached-array functions, no file/model/GT access.

No experiment has been run by importing or writing this draft. A later root
execution authorization must bind this exact code and the existing U128 seal.
The stable branch uses saved pre-probability token log-odds, never logit(p).
"""
import numpy as np

PRIMARY_ARMS=('source_native_replay','source_token_logodds',
              'R30_native','R30_token_logodds')
ARMS=PRIMARY_ARMS+('source_historical_stored_probability',)

def _topset(v):
 v=np.asarray(v)
 if v.shape!=(256,) or not np.isfinite(v).all():raise ValueError('finite full256 score vector required')
 return np.flatnonzero(v==v.max()).tolist()

def readouts_from_existing_U_cache(cache):
 """Use the already sealed, same-GPU-forward U128 score pair, without rerun.

 source_probability_projection_replay is the probability produced alongside
 exact_token_logodds[0]. probability_scores[0] instead is the original source
 cache mass, and is kept as a separate arithmetic bridge, not mixed into the
 2x2. The R35 R30_U_rows head has been independently verified bitexact to R30.
 """
 names=np.asarray(cache['arm_names']).tolist()
 if names!=['source','R30_U_rows','U_rows_matched_denominator','partial_output_keep']:
  raise ValueError('unexpected immutable R35 U128 head ordering')
 p=np.asarray(cache['probability_scores']);z=np.asarray(cache['exact_token_logodds'])
 replay=np.asarray(cache['source_probability_projection_replay'])
 if p.shape!=(4,256) or z.shape!=(4,256) or replay.shape!=(256,):raise ValueError('full256 vectors required')
 if p.dtype!=np.float32 or z.dtype!=np.float32 or replay.dtype!=np.float32:raise ValueError('preserve original GPU float32 inputs')
 if not np.isfinite(p).all() or not np.isfinite(z).all() or not np.isfinite(replay).all():raise ValueError('nonfinite scores')
 # Copy stored values. Float64 transport never recomputes the head or GPU LSE.
 scores=np.stack([replay,z[0],p[1],z[1],p[0]]).astype(np.float64)
 queries=scores.argmax(-1).astype(np.int64)
 logmass=-np.logaddexp(0.,-z[:2].astype(np.float64))
 diagnostics=[]
 for j,(native,stable) in enumerate([(replay,z[0]),(p[1],z[1])]):
  stable_top=_topset(stable);lm_top=_topset(logmass[j]);order=np.argsort(stable,kind='stable')
  same_top=stable_top==lm_top;monotone=bool(np.all(np.diff(logmass[j][order])>=0))
  if not (same_top and monotone):raise ValueError('stable logmass monotonic/top-set measurement check failed')
  p_recovered=np.exp(logmass[j]);identity_error=float(np.max(np.abs(native.astype(np.float64)-p_recovered)))
  if identity_error>=2e-5:raise ValueError('saved probability/logodds pair does not replay within historical tolerance')
  native_top=_topset(native);qn=int(native.argmax());qs=int(stable.argmax())
  diagnostics.append({'weight':'source' if j==0 else 'R30',
   'native_query':qn,'stable_query':qs,'native_topset':native_top,'stable_topset':stable_top,
   'native_exact_top_tie_n':len(native_top),'stable_exact_top_tie_n':len(stable_top),
   'native_values_exactly_one':int(np.count_nonzero(native==1)),
   'native_values_above_one':int(np.count_nonzero(native>1)),
   'changed_by_readout':qn!=qs,
   'stable_winner_in_native_topset':qs in native_top,
   'change_is_native_top_tie_break':qn!=qs and qs in native_top,
   'change_is_non_tie_numerical_reversal':qn!=qs and qs not in native_top,
   'logmass_topset_equal_logodds':same_top,'logmass_sorted_monotone':monotone,
   'logmass_not_subtracted_near_equal_LSEs':True,
   'probability_sigmoid_logodds_maxerror':identity_error})
 bridge={'source_replay_query':int(replay.argmax()),'source_historical_query':int(p[0].argmax()),
         'query_changed':int(replay.argmax())!=int(p[0].argmax()),
         'max_probability_abs_difference':float(np.max(np.abs(replay.astype(float)-p[0].astype(float)))),
         'not_TTA_or_readout_factor_gain':True}
 return {'arm_names':ARMS,'scores':scores,'queries':queries,
         'stable_logmass':logmass,'numerical_diagnostics':diagnostics,
         'source_arithmetic_bridge':bridge,
         'no_readout_query_change_for_either_weight':all(not d['changed_by_readout'] for d in diagnostics)}

def proposed_stop_rule(records):
 """Stop means stable vs native is identical within each of the two weights.

 It does not require source to equal adapted R30. Historical TTA changes are
 independent of whether the new numerical readout changes either prediction.
 No task label or new score is used by this mechanism-only stop condition.
 """
 if len(records)!=128:raise ValueError('all locked U128 rows required')
 return all(r['no_readout_query_change_for_either_weight'] for r in records)

if __name__=='__main__':
 raise SystemExit('DRAFT ONLY: no file reader or scoring runner; root approval required before execution')

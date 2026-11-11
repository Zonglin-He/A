"""Exact derived suffix counts; keep producer reader-only counter unmodified."""
from common76 import *
P=BASE/'target336_attempt001';assert read(P/'SEALED.json')['n']==336;trials=0;accepted=0;steps=0
for r in target_rows():
 s=read(P/'steps_PRIVATE'/(r['id']+'.json'))
 for branch in s['updates'].values():
  for st in branch:steps+=1;trials+=len(st['trials']);accepted+=st['accepted']
obj=steps*2+trials;explicit=336*10;producer=read(P/'COUNTS.json');assert producer['new_grounding_suffix_forwards']==explicit and steps==2016
rec={'state':'PASS_R76_DERIVED_EXACT_NATIVE_GROUNDING_SUFFIX_CALL_COUNTS','n':336,'optimizer_steps':steps,'accepted':accepted,'backtracking_trial_objective_calls':trials,'step_old_and_final_objective_calls':steps*2,'all_TTA_objective_calls':obj,'explicit_frozen_and_endpoint_grounding_readers':explicit,'total_actual_target_grounding_suffix_forwards':obj+explicit,'auxiliary_fusion_only_calls_in_objectives':2*obj,'initial_frozen_auxiliary_fusion_only_calls':336*2,'counter_limitation':'producer_new_grounding_suffix_forwards_counts_explicit_readers_only;use_derived_total_for_actual_cost','producer_predictions_or_weights_changed':False};dump(H/'DERIVED_COST_AUDIT.json',rec);print(rec)

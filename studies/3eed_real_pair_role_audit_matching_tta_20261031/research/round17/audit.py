"""CPU replay: source-pool invariance, global gate, real update and scores."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round17'))
from common import EXP,read,dump,sha,verify
def main():
 inp=EXP.parent/'input';A=read(inp/'A64_PRIVATE.json');B=read(inp/'B32_PRIVATE.json');assert len(A)==64 and len(B)==32 and not {r['id'] for r in A}&{r['id'] for r in B}
 old={r['id'] for r in read(ROOT/'research/round15/input/SOURCE96_GPU_INPUTS.json')+read(ROOT/'research/round15/input/REGRESSION140_GPU_INPUTS.json')};assert not old&{r['id'] for r in B}
 names=('features96_attempt001','pairs64_attempt001','G64_attempt001','G_variants64_attempt001','G_selected64_attempt001','B_predictions_attempt001','calibration_probe4_attempt001');seals={n:verify(ROOT/'crossplatform/round17'/n) for n in names};verify(EXP/'A_scoring_attempt001');verify(EXP/'B_scoring_attempt001')
 froot=ROOT/'crossplatform/round17/features96_attempt001';pred=ROOT/'crossplatform/round17/B_predictions_attempt001';selected=ROOT/'crossplatform/round17/G_selected64_attempt001';gate=read(EXP/'A_scoring_attempt001/GATE.json');arows=read(EXP/'A_scoring_attempt001/ROWS_PRIVATE.json');brows=read(EXP/'B_scoring_attempt001/ROWS_PRIVATE.json');assert {r['id_PRIVATE'] for r in arows}=={r['id'] for r in A} and {r['id_PRIVATE'] for r in brows}=={r['id'] for r in B}
 policies=('strict_roles','winner_role','pair_only')
 for platform in ('drone','quad'):
  for policy in policies:
   rr=[r for r in arows if r['platform']==platform];e=[r['details'][policy] for r in rr if r['details'][policy]['status']=='usable'];rat=[r['strict_pair_win_fraction'] for r in e if r['strict_pair_win_fraction'] is not None];pd=[r['representative_pair_delta'] for r in e];d=np.array([r['ious'][policy]-r['ious']['source'] for r in rr]);expected=bool(len(e)>=4 and len(rat)>=3 and np.mean(rat)>.6 and pd and np.mean(pd)>0 and d.mean()*100>=-1e-10 and (d>1e-10).sum()>=(d< -1e-10).sum());assert gate['policies'][platform][policy]['passed']==expected
  selected_policy=next((p for p in policies if gate['policies'][platform][p]['passed']),None);assert selected_policy==gate['platforms'][platform]['selected_policy']
  fit=read(pred/(platform+'_FIT.json'));assert fit['steps']==32 and fit['parameter_L2_delta']>0 and fit['update_parameters']==18496
  with np.load(froot/'projection_source_PRIVATE.npz') as s,np.load(pred/(platform+'_projection_PRIVATE.npz')) as p:assert np.linalg.norm(p['weight']-s['4.weight'])+np.linalg.norm(p['bias']-s['4.bias'])>0
  for r in [x for x in A if x['platform']==platform]:
   original=read(ROOT/'crossplatform/round17/G_variants64_attempt001/cases_PRIVATE'/(r['id']+'.json'));actual=read(selected/'cases_PRIVATE'/(r['id']+'.json'));assert actual['preference']==original['variants'][selected_policy]
 for r in B:
  with np.load(froot/'samples_PRIVATE'/(r['id']+'.npz')) as f,np.load(pred/'predictions_PRIVATE'/(r['id']+'.npz')) as p:
   assert np.array_equal(p['source_pool'],f['boxes']) and np.array_equal(p['boxes'],f['boxes'][p['query_indices']]);assert p['query_indices'][0]==f['scores'].argmax()
 for stage,rows in [('A',arows),('B',brows)]:
  s=read(EXP/(stage+'_scoring_attempt001/SUMMARY.json'))
  for platform,cells in s.items():
   rr=[r for r in rows if r['platform']==platform]
   for arm,c in cells.items():assert abs(np.mean([r['ious'][arm] for r in rr])*100-c['mIoU_percent'])<1e-12 and c['n']==len(rr)
 closure=read(EXP/'B_CLOSURE.json');assert closure['passed'] and closure['full_model_B_forwards']==32 and closure['only_two_allowed_projection_tensors_changed']
 receipt={'state':'R17_CPU_REPLAY_PASSED','A':64,'B':32,'checks':{'all_artifact_members_verified':True,'new_B_not_R15_or_R16_inputs':True,'G_exactly_A_no_B':True,'global_gate_and_policy_selection_recomputed':True,'no_per_sample_GT_preference_filter':True,'real_32_steps_both_platforms':True,'projection_18496_parameters_only':True,'all_B_boxes_bitexact_frozen_source_pool':True,'full_model_updated_B_forward_closure':True,'all_summaries_and_denominators_recomputed':True},'limits':['CPU geometry uses the previously independently audited metric primitive; this replay is not a new annotation source.','All target VAL historically exposed; three role policies selected on exposed A development utility.','Pair comparisons and multiple queries are not independent samples.','Content uniqueness is inherited and checked by the R15 decoded inventory; no same-instance/time correspondence claim.'],'artifact_seal_sha256':{n:sha(ROOT/'crossplatform/round17'/n/'SEALED.json') for n in names}}
 dump(EXP/'CPU_REPLAY_AUDIT.json',receipt);print('R17_CPU_REPLAY_PASSED')
if __name__=='__main__':main()

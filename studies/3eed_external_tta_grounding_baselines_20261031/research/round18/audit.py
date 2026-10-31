"""Independent execution of CPU artifact replay; no new target predictions."""
from pathlib import Path
import sys,collections,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round18'))
from common import EXP,read,dump,sha,verify,check_plan
def main():
 plan=check_plan();names=('formal96_attempt001','engineering2_attempt001','reg_online32_attempt001','cost_probe4_attempt001');seals={n:verify(ROOT/'crossplatform/round18'/n) for n in names};verify(EXP/'scored96_attempt001')
 inputs=read(EXP.parent/'input/ALL96_PRIVATE.json');formal=ROOT/'crossplatform/round18/formal96_attempt001';rows=read(EXP/'scored96_attempt001/ROWS_PRIVATE.json');trace=read(formal/'TRACE_PRIVATE.json');assert len(rows)==len(trace)==len(inputs)==96;assert {r['id_PRIVATE'] for r in rows}=={r['id'] for r in inputs};assert len({r['id'] for r in inputs})==96
 for r in inputs:
  p=formal/'predictions_PRIVATE'/(r['id']+'.npz');f=ROOT/'crossplatform/round17/features96_attempt001/samples_PRIVATE'/(r['id']+'.npz')
  with np.load(p,allow_pickle=False) as x,np.load(f,allow_pickle=False) as source:
   assert np.array_equal(x['source_pool'],source['boxes']) and np.array_equal(x['source_scores'],source['scores']);ns=x['arm_names'].tolist()
   for n in ('own_G_direct_no_grounder_update','own_G_episode_projection32'):
    if n in ns:j=ns.index(n);assert np.array_equal(x['arm_boxes_raw6'][j],x['source_pool'][x['query_indices'][j]])
 for r in trace:
  for n,d in r['diagnostics'].items():
   assert d['nonselected_parameters_unchanged']
   if n!='own_G_episode':assert (d['parameter_delta_L2']>0)==(d['steps']>0)
   if n=='MEMO' and d['steps']:
    assert d['common_groups']>=2 and min(d['coverage'])>=.5 and d['gradient_parameters']>1e8
   if n=='SAR' and d['steps']:assert d['first_reliable'] and d['second_reliable']
 for repository,commit in plan['official_commits'].items():assert subprocess.check_output(['git','-C',str(ROOT/'crossplatform/round18/official'/repository),'rev-parse','HEAD'],text=True).strip()==commit
 assert sha(ROOT/'crossplatform/round18/official/SAR/sam.py')==plan['official_SAM_sha256']
 summary=read(EXP/'scored96_attempt001/SUMMARY.json')
 for split,platforms in summary.items():
  for platform,cells in platforms.items():
   rr=[r for r in rows if r['split']==split and r['platform']==platform]
   for arm,c in cells.items():assert c['n']==len(rr) and abs(c['mIoU_percent']-np.mean([r['ious'][arm] for r in rr])*100)<1e-12
 # Native online readout precedes the first update in each source-reset platform.
 B=read(EXP.parent/'input/B32_PRIVATE.json');online=ROOT/'crossplatform/round18/reg_online32_attempt001'
 for platform in ('drone','quad'):
  r=next(r for r in B if r['platform']==platform)
  with np.load(online/'predictions_PRIVATE'/(r['id']+'.npz')) as x,np.load(formal/'predictions_PRIVATE'/(r['id']+'.npz')) as y:assert np.array_equal(x['arm_boxes_raw6'][0],y['arm_boxes_raw6'][0])
 cp=read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json');assert sha(cp['source_checkpoint_path'])==plan['source_checkpoint_sha256']
 report={'state':'R18_CPU_ARTIFACT_REPLAY_PASSED','checks':{'all_prediction_and_metric_seals':True,'all96_inputs_and_negative_rows_retained':True,'all_frozen_sources_bitexact_R17':True,'own_matching_method_candidate_boxes_frozen':True,'all_nonselected_parameters_verified_by_runner':True,'actual_parameter_deltas_match_updates':True,'MEMO_geometry_correspondence_and_mass_gates':True,'SAR_two_pass_reliability_or_valid_skip':True,'official_code_commits_and_SAM_hash':True,'all_metric_denominators_recomputed':True,'online_first_preupdate_equals_source_each_platform':True,'source_checkpoint_immutable':True},'artifact_sha256':{n:sha(ROOT/'crossplatform/round18'/n/'SEALED.json') for n in names},'limitations':['Same previously audited CPU geometry primitive; not a second set of independent annotations.','No identity guarantee for geometric candidate matches or token category proxies.','Episodic current-input and shared A-to-B/online protocols are separate.','Reg is an inspired port with stated RoI/yaw/class/equation substitutions, not official original reproduction.']}
 dump(EXP/'CPU_REPLAY_AUDIT.json',report);print('R18_CPU_REPLAY_PASSED')
if __name__=='__main__':main()

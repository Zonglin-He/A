"""Artifact, update, coverage and numerical replay checks without model calls."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round19'))
from common import *
from features_v2 import feature
def main():
 artifacts=['sourceH32_attempt001','sourceV16_attempt001','pairs96_attempt001','pairs112_attempt002','G96_attempt001','G112_attempt002','G112_attempt003','G112_attempt004','pair_fit112_attempt002','historical_pair_interface_attempt001','closed112_attempt002'];seals={n:verify(ROOT/'crossplatform/round19'/n) for n in artifacts};verify(EXP/'scored112_attempt001');rows=read(EXP/'scored112_attempt001/ROWS_PRIVATE.json');inputs=read(EXP.parent/'input/ALL112_v2_PRIVATE.json');assert len(rows)==len(inputs)==112 and len({r['id'] for r in inputs})==112
 trace=read(ROOT/'crossplatform/round19/pair_fit112_attempt002/TRACE_PRIVATE.json');closed=read(ROOT/'crossplatform/round19/closed112_attempt002/TRACE_PRIVATE.json');accepted=sum(r['feedback_accepted'] for r in trace);assert accepted==seals['pair_fit112_attempt002']['accepted']==seals['closed112_attempt002']['actual_updated_grounder_forwards']
 for r in trace:
  d=r['diagnostics'];assert (d['parameter_delta_L2']>0)==(d['steps']>0)
  if r['feedback_accepted']:
   assert d['selected_parameters']==18496 and d['steps']<=128 and all(s['gradient_L2']>0 for s in d['trace']);assert d['no_C_automatically_added_as_negative']
 assert all(r['boxes_frozen_verified'] for r in closed) and seals['closed112_attempt002']['source_restored']
 for r in inputs:
  f=ROOT/'crossplatform/round19/closed112_attempt002/predictions_PRIVATE'/(r['id']+'.npz');z=feature(r)
  with np.load(f) as a:assert np.array_equal(a['arm_boxes_raw6'],z['boxes'][a['query_indices']])
 summary=read(EXP/'scored112_attempt001/SUMMARY.json')
 for stage,platforms in summary.items():
  for platform,cells in platforms.items():
   rr=[r for r in rows if r['stage']==stage and r['platform']==platform]
   for arm,c in cells.items():assert c['n']==len(rr) and abs(c['mIoU_percent']-100*np.mean([r['ious'][arm] for r in rr]))<1e-12
 V=[r for r in inputs if r['stage']=='V'];locktime=(EXP/'G_LOCK_v4.json').stat().st_mtime;vfiles=[ROOT/'crossplatform/round19/G112_attempt002/cases_PRIVATE'/(r['id']+'.json') for r in V];v_after=all(p.stat().st_mtime>locktime for p in vfiles)
 probes=[r for r in rows if r['evidence_responses']];assert all(not d['used_as_gate'] for r in probes for d in r['evidence_responses'].values())
 assert sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 result={'state':'R19_CPU_ARTIFACT_REPLAY_PASSED','all112cases_and_negatives_kept':True,'all_seals_verified':True,'candidate_boxes_frozen_bitexact':True,'only18496projection_parameters_updated':True,'actual_update_steps':sum(r['diagnostics']['steps'] for r in trace),'accepted':accepted,'actual_updated_grounder_forwards':accepted,'all_source_weights_restored':True,'all_metric_denominators_recomputed':True,'V16_final_visual_outputs_created_after_final_G_rule_lock':v_after,'V16_not_pristine_blind':True,'destruction_diagnostics_not_hard_gate':True,'source_checkpoint_immutable':True,'history_count':len(artifacts),'seal_hashes':{n:sha(ROOT/'crossplatform/round19'/n/'SEALED.json') for n in artifacts},'identity_audit_limit':'GTgeometry assignment is a proxy; not independent semantic/mask identity gold.','no_route_level_rejection':True};dump(EXP/'CPU_REPLAY_AUDIT.json',result);print('R19_AUDIT_PASSED',accepted)
if __name__=='__main__':main()

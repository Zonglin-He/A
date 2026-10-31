"""Post-run CPU replay of splits, gate, box membership, scores and parameter scope.
Does not create alternative predictions, tune G or fit a model.
"""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round16'))
from common import EXP,read,sha,dump,verify
def main():
 A=read(EXP.parent/'input/A32_PRIVATE.json');B=read(EXP.parent/'input/B32_PRIVATE.json');assert len(A)==len(B)==32 and not {r['id'] for r in A}&{r['id'] for r in B}
 folders=[ROOT/'crossplatform/round16'/x for x in ('features64_attempt001','G_A32_attempt001','G_A32_attempt002','G_A32_attempt003','G_A32_attempt004','B_predictions_attempt001')]+[EXP/'A_scoring_attempt001',EXP/'B_scoring_attempt001']
 seals={str(f.relative_to(ROOT)):verify(f) for f in folders};features=folders[0];pred=folders[5];groot=folders[4]
 arows=read(EXP/'A_scoring_attempt001/ROWS_PRIVATE.json');brows=read(EXP/'B_scoring_attempt001/ROWS_PRIVATE.json');gate=read(EXP/'A_scoring_attempt001/GATE.json');assert {r['id_PRIVATE'] for r in arows}=={r['id'] for r in A};assert {r['id_PRIVATE'] for r in brows}=={r['id'] for r in B}
 assert {f.stem for f in (groot/'cases_PRIVATE').glob('*.json')}=={r['id'] for r in A}
 for platform in ('drone','quad'):
  rows=[r for r in arows if r['platform']==platform];eligible=[r for r in rows if r['details']['preference_status']=='usable'];ratios=[r['details']['strict_pair_win_fraction'] for r in eligible if r['details']['strict_pair_win_fraction'] is not None];pd=[r['details']['representative_pair_delta'] for r in eligible];d=np.array([r['ious']['direct_G']-r['ious']['source'] for r in rows]);expected=bool(len(eligible)>=4 and ratios and np.mean(ratios)>.5 and pd and np.mean(pd)>0 and d.mean()*100>=-1e-10 and (d>1e-10).sum()>=(d< -1e-10).sum());assert gate['platforms'][platform]['passed']==expected
  fit=read(pred/(platform+'_FIT.json'));assert fit['steps']==(32 if expected else 0) and (fit['parameter_L2_delta']>0)==expected
  with np.load(features/'projection_source_PRIVATE.npz',allow_pickle=False) as z, np.load(pred/(platform+'_projection_PRIVATE.npz'),allow_pickle=False) as p:
   delta=np.linalg.norm(p['weight']-z['4.weight'])+np.linalg.norm(p['bias']-z['4.bias']);assert (delta>0)==expected
 for row in B:
  sid=row['id']
  with np.load(features/'samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as f,np.load(pred/'predictions_PRIVATE'/(sid+'.npz'),allow_pickle=False) as p:
   assert np.array_equal(p['source_pool'],f['boxes']) and np.array_equal(p['boxes'],f['boxes'][p['query_indices']]);assert p['query_indices'][0]==f['scores'].argmax()
 for stage,rows in [('A',arows),('B',brows)]:
  summary=read(EXP/(stage+'_scoring_attempt001/SUMMARY.json'))
  for platform in ('drone','quad'):
   rr=[r for r in rows if r['platform']==platform]
   for arm,cell in summary[platform].items():
    values=np.array([r['ious'][arm] for r in rr]);assert abs(cell['mIoU_percent']-values.mean()*100)<1e-12 and cell['n']==16
 closure=read(EXP/'B_CLOSURE.json');assert closure['passed'] and closure['prediction_seal_sha256']==sha(pred/'SEALED.json')
 result={'state':'R16_CPU_replay_passed','checks':{'all_seals_and_members':True,'A_B_disjoint_ids':True,'G_cases_exactly_A_no_B':True,'all64_score_rows_and_negatives_retained':True,'gate_recomputed_no_relabeling':True,'parameter_delta_matches_gate':True,'all_selected_boxes_bitexact_source_pool':True,'summary_recomputed':True,'full_model_closure_or_no_update_skip':True},'limitations':['Same scorer geometry primitive as the previously audited official-transform implementation; this is a separate replay, not a second independent ground-truth annotation.','Content uniqueness inherits R15 decoded-content audit; scene/object/time independence is not established.','Forced verbalizer confidence uncalibrated, relative readout engineering exposed first15 A before task labels.','A labels only apply prespecified gate; target-development screening is not wholly target-label-free.'],'seals':{str(f.relative_to(ROOT)):sha(f/'SEALED.json') for f in folders}}
 dump(EXP/'CPU_REPLAY_AUDIT.json',result);print('CPU_REPLAY_PASSED')
if __name__=='__main__':main()

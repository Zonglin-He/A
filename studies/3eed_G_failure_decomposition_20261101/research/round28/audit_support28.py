"""CPU physical-support witness recomputation without task labels."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
from observations27 import load
def main():
 verify(BASE/'blind_review_attempt001');keys={x['blind_case']:x for x in read(EXP.parent/'input/BLIND_KEY_PRIVATE.json')};labels={x['case']:x for x in read(BASE/'blind_review_attempt001/ANNOTATIONS.json')};lookup={x['id']:x for x in rows()};records=read(EXP/'POINT_SUPPORT24_ROWS.json');checked=0
 for x in records:
  if 'details' not in x:continue
  k=keys[x['case']];r=lookup[k['id_PRIVATE']];frame,scene,b=load(r);P=scene['visible_points_model'];owner=scene['point_owner'];targets={k['labels_to_mask_PRIVATE'][s] for s in labels[x['case']]['target_labels']};tp=np.isin(owner,list(targets));op=(owner>=0)&~tp;z=feature(r);p=paths(r);prep=read(p['prep']/'cases_PRIVATE'/(r['id']+'.json'));W=load_npz(p['prep']/'arrays_PRIVATE'/(r['id']+'.npz'))['weights'];e=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'))['targets_PRIVATE'];cue=W@np.array([e[str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in prep['mask_pool_PRIVATE']]);M=np.flatnonzero(W.sum(1)>0);best=M[abs(cue[M]-cue[M].max())<=1e-9];qs={'source':int(z['scores'].argmax()),'cue_max_source_tiebreak':int(best[z['scores'][best].argmax()]),'mapped_final':int(M[(z['scores'][M]+2*cue[M]).argmax()]),'original_final':int((z['scores']+2*cue).argmax())}
  for name,q in qs.items():
   box=z['boxes'][q];inbox=np.less_equal(np.abs(P-box[:3]),box[3:]/2).all(1);t=int(inbox[tp].sum());o=int(inbox[op].sum());n=int(inbox.sum());v=x['details'][name];assert (t,o,n)==(v['target_points'],v['other_foreground_points'],v['all_box_visible_points']);checked+=1
 out={'state':'PASSED','resolved_visual_cases':21,'source_cuemax_mappedfinal_originalfinal_point_witnesses_checked':checked,'no_GT_geometry_or_IoU_in_support_recomputation':True,'same_projection_masks_assumption_disclosed':True};dump(EXP/'POINT_SUPPORT_CPU_AUDIT.json',out);print('SUPPORT_AUDIT28',checked)
if __name__=='__main__':main()

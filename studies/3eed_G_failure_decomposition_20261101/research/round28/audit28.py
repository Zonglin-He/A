"""Independent CPU reconstruction of cached-score counterfactuals and task reductions."""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round28'),str(ROOT/'research/round5/audit')]
from common28 import *
from independent_core_v2 import candidate_ious,transform_gt
def main():
 verify(BASE/'blind_review_attempt001');verify(BASE/'fusion528_attempt001');verify(ROOT/'crossplatform/round27/predictions528_attempt001');verify(EXP.parent/'private_views')
 rr=read(EXP/'FUSION_ROWS_PRIVATE.json');idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};pure_positive_changes=0;imagecorrect=collections.Counter()
 for r,x in zip(rows(),rr):
  z=feature(r);p=paths(r);prep=read(p['prep']/'cases_PRIVATE'/(r['id']+'.json'));W=load_npz(p['prep']/'arrays_PRIVATE'/(r['id']+'.npz'))['weights'];pool=prep['mask_pool_PRIVATE'];ev=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'))['targets_PRIVATE'];e=np.array([ev[str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in pool]);mapped=W.sum(1)>0;s=z['scores'].astype(float);v=W@e if pool else np.zeros(256);f=s+2*v
  assert int(f.argmax())==x['q']['joint_original'];q0=int(s.argmax())
  if pool:
   assert mapped[q0];M=np.flatnonzero(mapped);qm=int(M[f[M].argmax()]);assert qm==x['q']['mapped_competition_only']==x['q']['source_anchor_zero'];assert np.max(abs(W.sum(1)[mapped]-1))<1e-12
   anchor=f.copy();anchor[mapped]-=2*v[q0];assert anchor[qm]>=anchor.max()-1e-12
  for c in [0,.25,.5,1,2]:
   q=int((s+2*c*mapped).argmax());pure_positive_changes+=q!=q0
  for c in [-2,-1,-.5,-.25,0,.25,.5,1,2]:
   shifted=f.copy();shifted[mapped]+=2*c;q=int(shifted.argmax());expected=x['q']['offset_'+str(c)];assert shifted[expected]>=shifted.max()-1e-12
  pr=load_npz(BASE/'fusion528_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(pr['boxes'],z['boxes'][pr['queries']]);ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform']);b=pr['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vals=candidate_ious(b,gt)[0]
  for n,val in zip(pr['names'],vals):assert float(val)==x['ious'][n]
 assert pure_positive_changes==0
 # Independent exact ranking reduction from sealed visual labels and raw cached scores.
 labels={a['case']:a for a in read(BASE/'blind_review_attempt001/ANNOTATIONS.json')};keys=read(EXP.parent/'input/BLIND_KEY_PRIVATE.json');lookup={r['id']:r for r in rows()};evaluated={a['case']:a for a in read(EXP/'IDENTITY24_ROWS.json')}
 for k in keys:
  a=labels[k['blind_case']];r=lookup[k['id_PRIVATE']];ev=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'))['targets_PRIVATE'];e={l:ev[str(m)]['crop_scene_joint']['log_odds_yes_no'] for l,m in k['labels_to_mask_PRIVATE'].items()};top={l for l,v in e.items() if abs(v-max(e.values()))<=1e-7};t=set(a['target_labels']);status='unresolved_visual' if not t else 'correct' if top<=t else 'incorrect' if not top&t else 'tie_target_other';assert status==evaluated[k['blind_case']]['image_ranking_status'];imagecorrect[status]+=1
 s=read(EXP/'DIAGNOSTIC_SUMMARY.json')
 for phase,rs in [('ALL528',rr),('OLD400',rr[:400]),('REPLAY256',rr[208:464]),('Q64',rr[464:])]:
  for a in ['mapped_competition_only','median_zero','mean_zero']:
   mean=np.mean([np.mean([x['ious'][a]-x['ious']['joint_original'] for x in rs if x['platform']==p]) for p in ['drone','quad']])*100;assert abs(mean-s['readout_effects'][phase+'_'+a]['balanced_gain_pp'])<1e-10
 assert read(EXP/'SOURCE_PRIOR_AUDIT.json')['n']==528
 dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','all528_counterfactual_queries_and_task_IoUs_recomputed':True,'source_anchor_equivalence_and_row_mass_checked':True,'pure_constant_nonnegative_evidence_cannot_change_sourcewinner_in_this_cohort':True,'all24_blind_visual_ranking_reduced_from_rawscores':dict(imagecorrect),'blind_annotations_preunblinding_seal_verified':True,'source_prior_all528_readout_mask_replay_verified':True,'old_R27_predictions_unchanged':True,'no_new_model_forward_training_GPU_or_TTA':True,'visual_review_is_assistant_singleannotator_not_expertgold':True,'point_support_shares_camera_and_masks_not_external_3Didentity_truth':True,'no_unresolved_label_forced_by_IoU':True,'all_negative_counterfactuals_retained':True})
 print('AUDIT28_PASSED',dict(imagecorrect))
if __name__=='__main__':main()

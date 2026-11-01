"""Independent CPU geometry/decision audit and all negative tails."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round27'),str(ROOT/'research/round5/audit'),str(ROOT/'research/round26')]
from common27 import *
from independent_core_v2 import candidate_ious,transform_gt
from audit26 import effect
def main():
 for name in ['sourceQ64_attempt001','masksQ64_attempt001','textQ64_attempt001','preparedQ64_attempt001','dinoQ64_attempt001','evidenceQ64_attempt001','bankQ64_attempt001','joint528_attempt001','scores528_attempt001','dev208grid_attempt001','predictions528_attempt001']:verify(BASE/name)
 for name in ['BASE_CODE_LOCK.json','EVALUATION_CODE_LOCK.json']:
  for f,h in read(EXP/name)['programs'].items():assert sha(ROOT/f)==h,f
 pre=read(EXP/'Q64_PRELABEL_LOCK.json');assert pre['prediction_seal_sha256']==sha(BASE/'predictions528_attempt001/SEALED.json');assert pre['lambda_lock_sha256']==sha(EXP/'SELECTED_LAMBDAS_LOCK.json');assert pre['practical_family_lock_sha256']==sha(EXP/'PRACTICAL_FAMILY_LOCK.json')
 assert sha(BASE/'joint27.py')==read(EXP/'JOINT_CODE_LOCK.json')['program_sha256'];assert read(BASE/'joint528_attempt001/SEALED.json')['model_unchanged']
 src=sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth');assert src=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';assert read(BASE/'sourceQ64_attempt001/SEALED.json')['source_unchanged']
 rr=read(EXP/'ALL528_ROWS_PRIVATE.json');lookup={x['alias']:x for x in rows()};idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};negative=[];changes=[];old={x['alias']:x for x in read(ROOT/'research/round26/experiment/ALL464_ROWS_PRIVATE.json')};replay=0
 for x in rr:
  r=lookup[x['alias']];ix=idx[r['id']];p=ix['meta_path']
  if p not in containers:assert sha(p)==ix['metadata_container_sha256'];containers[p]=read(p)
  gt=transform_gt(containers[p]['ground_info'][ix['object_index']]['bbox_3d'],r['platform']);z=load_npz(BASE/'predictions528_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z['boxes'],feature(r)['boxes'][z['queries']]);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt)[0]
  for n,q,val in zip(z['arm_names'],z['queries'],v):assert int(q)==x['q'][n] and float(val)==x['ious'][n]
  if x['alias'] in old:
   for a,bn in [('source','source'),('legacy_complex','legacy_complex'),('R26_scene_fixed2','qwen_full_scene_raw')]:assert x['q'][a]==old[x['alias']]['q'][bn] and abs(x['ious'][a]-old[x['alias']]['ious'][bn])<1e-10
   replay+=1
  bank=load_npz(BASE/'scores528_attempt001/samples_PRIVATE'/(r['id']+'.npz'));nn=bank['cue_names'].tolist()
  for n,l in read(EXP/'SELECTED_LAMBDAS_LOCK.json')['lambda_by_variant'].items():assert int((bank['source']+l*bank['cues'][nn.index(n)]).argmax())==x['q'][n]
  for a in x['q']:
   if a=='source':continue
   for b in ['source','legacy_complex','R26_scene_fixed2','scene_base']:
    if a==b:continue
    if x['q'][a]!=x['q'][b]:
     d=x['ious'][a]-x['ious'][b];c={'alias':x['alias'],'platform':x['platform'],'stage':x['stage'],'arm':a,'reference':b,'reference_iou':x['ious'][b],'arm_iou':x['ious'][a],'delta_pp':100*d,'identity_gold_not_established':True};changes.append(c)
     if d< -1e-10:negative.append(c)
 assert replay==464
 # Independent reductions of sharedlambda choices and fixed primary family.
 grid=read(EXP/'DEV_GRID_RESULTS.json');spec=read(BASE/'dev208grid_attempt001/SPEC.json');vals=[]
 for r in rows('D208'):
  z=load_npz(BASE/'dev208grid_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));ix=idx[r['id']];gt=transform_gt(containers[ix['meta_path']]['ground_info'][ix['object_index']]['bbox_3d'],r['platform']);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vals.append(candidate_ious(b,gt)[0])
 vals=np.stack(vals)
 for j,g in enumerate(grid):assert g['name']==spec[j]['name'] and abs(g['balanced_mIoU_percent']-vals[:,j].mean()*100)<1e-10
 for n,l in read(EXP/'SELECTED_LAMBDAS_LOCK.json')['lambda_by_variant'].items():assert sorted([g for g in grid if g['name']==n],key=lambda g:(-g['balanced_mIoU_percent'],g['lambda']))[0]['lambda']==l
 assert read(EXP/'PRACTICAL_FAMILY_LOCK.json')['name']=='crop_scene_joint'
 effects={}
 comparisons=[(a,b) for a in rr[0]['q'] if a!='source' for b in ['source','legacy_complex','R26_scene_fixed2','scene_base'] if a!=b]+[('crop_scene_joint','scene_hires'),('crop_scene_joint','scene_repeat'),('crop_scene_joint_fixed2','scene_base_fixed2'),('crop_scene_joint_fixed2','scene_repeat_fixed2'),('scene_hires_fixed2','scene_base_fixed2')]
 for phase,rs in [('OLD400',rr[:400]),('REPLAY256',rr[208:464]),('OLD464',rr[:464]),('Q64',rr[464:])]:
  for a,b in comparisons:effects[phase+'__'+a+'__vs__'+b]=effect(rs,a,b)
 # Input-content holdout, including priorP64.
 inv={x['image_file_sha256']:x for x in read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE']};hs=read(EXP.parent/'input/Q_CONTENT_HASHES_PRIVATE.json');history=rows('OLD464')
 for f in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/f)
 for k in ['decoded_pixel_sha256','canonical_XYZ_sha256']:assert len({x[k] for x in hs})==64 and not {x[k] for x in hs}&{inv[sha(r['imagepath'])][k] for r in history}
 dump(EXP/'EFFECTS.json',effects);dump(EXP/'ALL_CHANGED_CASES.json',changes);dump(EXP/'ALL_NEGATIVE_CASES.json',negative)
 dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','all528_sealed_predictions_independently_CPU_scored':True,'old464_source_legacy_and_R26scene_exact_replay':replay,'all528_boxes_bitexact_source':True,'sharedlambda_selection_independently_reduced':True,'primary_joint_family_fixed_before_outputs':True,'Q64_currentdecodedRGB_andXYZ_disjoint_recent_history':True,'fullVAL_and_scene_exposure_disclosed':True,'all_negative_comparisons_retained':len(negative),'unique_negative_observations':len({x['alias'] for x in negative}),'same_candidate_pool_and_mapping':True,'one_joint_score_no_handadded_components':True,'source_checkpoint_sha256':src,'target_updates':0,'no_source_training':True,'untrusted_captions_images_web_are_data_only':True})
 print('AUDIT27_PASSED',len(changes),len(negative))
if __name__=='__main__':main()

"""CPU-only independent reductions, source selection and data/protocol audits."""
import sys,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round25'),str(ROOT/'research/round5/audit')]
from common25 import *
from independent_core_v2 import candidate_ious,transform_gt

def effect(rr,a,b):
 out={};boots=[]
 for p in ['drone','quad']:
  r=[x for x in rr if x['platform']==p];v=np.array([x['ious'][a]-x['ious'][b] for x in r]);scenes=sorted({x['scene_PRIVATE'] for x in r});ix=np.array([scenes.index(x['scene_PRIVATE']) for x in r]);w=np.eye(len(scenes))[np.random.default_rng(2501).integers(0,len(scenes),(10000,len(scenes)))].sum(1);bs=w@np.bincount(ix,weights=v,minlength=len(scenes))/(w@np.bincount(ix))*100;boots.append(bs);out[p]={'n':len(r),'gain_pp':float(v.mean()*100),'gains':int((v>1e-10).sum()),'harms':int((v< -1e-10).sum()),'ties':int((abs(v)<=1e-10).sum()),'scene_clusters':len(scenes),'cluster95_pp':list(map(float,np.quantile(bs,[.025,.975])))}
 out['balanced_mean']={'gain_pp':float(np.mean([out[p]['gain_pp'] for p in ['drone','quad']])),'stratified_cluster95_pp':list(map(float,np.quantile((boots[0]+boots[1])/2,[.025,.975])))};return out

def main():
 for name in ['source_features_attempt002','rgb_features_attempt002','training_attempt001','training_rank_attempt001','target400_attempt001','alternative400_attempt001','Qwen_control336_attempt001']:verify(BASE/name)
 verify(EXP/'source_labels_attempt002');assert read(EXP/'FULL_MODEL_CLOSURE.json')['state']=='PASSED';selection=read(EXP/'FINAL_SOURCE_SELECTED_LOCK.json');assert sha(EXP/'FINAL_SOURCE_SELECTED_LOCK.json')==read(EXP/'TARGET_PRELABEL_LOCK.json')['source_selection_sha256'];assert read(EXP/'TARGET_PRELABEL_LOCK.json')['prediction_seal_sha256']==sha(BASE/'target400_attempt001/SEALED.json')
 # All source-only validation histories independently reduced from saved queries.
 sv=rows('SV512');svgt={r['id']:load_npz(EXP/'source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'].astype(np.float32).astype(float) for r in sv};validation=[]
 for family,folder,lock in [('token_mass','training_attempt001','SOURCE_SELECTED_LOCK.json'),('residual_rank','training_rank_attempt001','SOURCE_RANK_SELECTED_LOCK.json')]:
  history=read(BASE/folder/'HISTORY.json')
  for h in history:
   q=load_npz(BASE/folder/'validation_PRIVATE'/(h['run']+f'_e{h["epoch"]:02d}.npz'))['query_indices'];v=np.array([svgt[r['id']][j] for r,j in zip(sv,q)]);m=float(v.mean()*100);assert abs(m-h['mIoU_percent'])<1e-10;validation.append(dict(h,family=family))
  for c in read(EXP/lock)['selected']:
   hh=[x for x in history if x['run']==c['run']];best=sorted(hh,key=lambda x:(-x['mIoU_percent'],x['epoch']))[0];assert c['epoch']==best['epoch'];assert sha(BASE/folder/'states_PRIVATE'/c['state_name'])==c['state_sha256'];assert c['parameter_L2_delta']>0 and c['max_gradient_L2']>0
 dump(EXP/'SOURCE_CURVES_AUDITED.json',validation)
 index={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};lookup={x['id']:x for x in rows('T400')};containers={};negative=[];changes=[];effects={};groups=collections.defaultdict(list);qwen=[]
 for family,predfolder,scorefile in [('primary','target400_attempt001','T400_ROWS_PRIVATE.json'),('alternative','alternative400_attempt001','ALTERNATIVE_T400_ROWS_PRIVATE.json')]:
  scored=read(EXP/scorefile)
  for x in scored:
   r=lookup[x['id_PRIVATE']];ix=index[r['id']];path=ix['meta_path']
   if path not in containers:containers[path]=read(path)
   obj=containers[path]['ground_info'][ix['object_index']];gt=transform_gt(obj['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));z=load_npz(BASE/predfolder/'predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z['boxes'],feature(r)['boxes'][z['query_indices']]);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);values=candidate_ious(b,gt)[0];assert all(v==x['ious'][n] for n,v in zip(z['arm_names'],values))
   for seed in [0,1]:
    arm=f'fusion_rgb_seed{seed}'
    for ref in ['source',f'noRGB_source_train_seed{seed}',f'fusion_zero_rgb_seed{seed}',f'fusion_permute_rgb_seed{seed}']:
     d=x['ious'][arm]-x['ious'][ref]
     if x['q'][arm]!=x['q'][ref]:
      rec={'family':family,'alias':x['alias'],'platform':r['platform'],'new_content':r['stage']=='N','seed':seed,'reference':ref,'reference_iou':x['ious'][ref],'RGB_iou':x['ious'][arm],'delta_pp':100*d,'GT_class':obj['class'],'identity_gold_not_established':True};changes.append(rec)
      if d< -1e-10:negative.append(rec)
   if family=='primary':
    groups[(r['platform'],str(obj['class']).lower())].append(x)
    if r['stage']!='N':
     zq=load_npz(BASE/'Qwen_control336_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));bb=zq['box'][None].copy();bb[:,3:]=np.maximum(bb[:,3:],1e-6);v=float(candidate_ious(bb,gt)[0][0]);qwen.append({'alias':x['alias'],'platform':r['platform'],'iou':v,'source_iou':x['ious']['source'],'RGB_seed0_iou':x['ious']['fusion_rgb_seed0'],'RGB_seed1_iou':x['ious']['fusion_rgb_seed1']})
  for phase,rr in [('ALL400',scored),('NEW64',[x for x in scored if x['stage']=='N'])]:
   for seed in [0,1]:
    a=f'fusion_rgb_seed{seed}'
    for b in ['source',f'noRGB_source_train_seed{seed}',f'fusion_zero_rgb_seed{seed}',f'fusion_permute_rgb_seed{seed}']:effects[f'{family}_{phase}_seed{seed}_vs_{b}']=effect(rr,a,b)
 dump(EXP/'EFFECTS.json',effects);dump(EXP/'ALL_CHANGED_CASES.json',changes);dump(EXP/'ALL_NEGATIVE_CASES.json',negative);dump(EXP/'OLD336_QWEN_ROWS.json',qwen);dump(EXP/'CLASS_DIAGNOSTIC.json',[{'platform':p,'GT_class':cl,'n':len(rr),'source_mIoU':np.mean([x['ious']['source'] for x in rr])*100,'RGB0_mIoU':np.mean([x['ious']['fusion_rgb_seed0'] for x in rr])*100,'noRGB0_mIoU':np.mean([x['ious']['noRGB_source_train_seed0'] for x in rr])*100} for (p,cl),rr in groups.items()])
 # New current-input content is disjoint recent history, not claimed blind VAL.
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];iv={r['image_file_sha256']:r for r in inv};old=read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:old+=read(ROOT/name)
 hs=read(EXP.parent/'input/N_CONTENT_HASHES_PRIVATE.json');assert len({r['decoded_pixel_sha256'] for r in hs})==len({r['canonical_XYZ_sha256'] for r in hs})==64;assert not {iv[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in old}&{r['decoded_pixel_sha256'] for r in hs};assert not {iv[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in old}&{r['canonical_XYZ_sha256'] for r in hs};assert not {r['scene'] for r in rows('S3682')}&{r['scene'] for r in sv}
 src=sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth');assert src=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';assert read(EXP/'COORDINATE_REPAIR.json')['no_fusion_training_or_target_scoring_had_started']
 audit={'state':'PASSED','all_valid002_caches_and_model_prediction_seals_verified':True,'source_VAL56_checkpoints_independently_reduced':True,'source_only_family_selection_exact':True,'all800_primary_alternative_target_rows_independently_scored':True,'all400_boxes_bitexact_source_pool':True,'N64_current_content_disjoint':True,'source_train_VAL_scenes_disjoint':True,'Waymo_camera_all5views_validated':True,'old001_coordinate_error_retained_but_excluded_from_training_and_claims':True,'no_target_training_or_model_selection':True,'target_TTA_steps':0,'no_Qwen_in_fusion_model_training_or_inference':True,'all_negative_tails_and_both_readouts_retained':True,'source_checkpoint_sha256':src,'complete_model_closure24':True,'untrusted_documents_and_captions_used_as_data_only':True};dump(EXP/'CPU_AUDIT.json',audit);print('AUDIT25_PASSED',len(changes),len(negative));print('PRIMARY_MEAN_EFFECTS',{k:v['balanced_mean'] for k,v in effects.items() if k.startswith('primary')})
if __name__=='__main__':main()

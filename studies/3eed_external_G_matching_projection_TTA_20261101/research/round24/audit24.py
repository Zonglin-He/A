"""Independent CPU integrity/effect audit; task labels remain scoring-only."""
import sys,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'research/round24')]
from common24 import *
from evaluate24 import score,summary

def ci(rr,a,b):
 out={};bs=[]
 for platform in ['drone','quad']:
  r=[x for x in rr if x['platform']==platform];scenes=sorted({x['scene_PRIVATE'] for x in r});ix=np.array([scenes.index(x['scene_PRIVATE']) for x in r]);delta=np.array([x['ious'][a]-x['ious'][b] for x in r]);rng=np.random.default_rng(2401);counts=np.bincount(ix);weights=np.eye(len(scenes))[rng.integers(0,len(scenes),(10000,len(scenes)))].sum(1);boot=(weights@np.bincount(ix,weights=delta,minlength=len(scenes)))/(weights@counts)*100;bs.append(boot);out[platform]={'gain_pp':float(delta.mean()*100),'gains':int((delta>1e-10).sum()),'harms':int((delta< -1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),'clusters':len(scenes),'scene_cluster95_gain_pp':list(map(float,np.quantile(boot,[.025,.975])))}
 out['balanced_mean']={'gain_pp':float(np.mean([out[p]['gain_pp'] for p in ['drone','quad']])),'stratified_scene_cluster95_gain_pp':list(map(float,np.quantile((bs[0]+bs[1])/2,[.025,.975])))};return out

def aabb_iou(a,b):
 alo=a[:3]-np.maximum(a[3:],1e-6)/2;ahi=a[:3]+np.maximum(a[3:],1e-6)/2;blo=b[:3]-np.maximum(b[3:],1e-6)/2;bhi=b[:3]+np.maximum(b[3:],1e-6)/2;inter=np.maximum(0,np.minimum(ahi,bhi)-np.maximum(alo,blo)).prod();return float(inter/max(np.maximum(a[3:],1e-6).prod()+np.maximum(b[3:],1e-6).prod()-inter,1e-12))

def main():
 names=['capture336_attempt001','textM64_attempt001','preparedM64_attempt001','dinoM64_attempt001','evidenceM64_attempt001','bankM64_attempt001','screen_attempt001','closureM64_attempt001','clip336_attempt001','memory_attempt001','memory_selected_scoresM64_attempt001','finalM64_attempt001']
 for n in names:verify(BASE/n)
 rr=rows('M64');scored=score(BASE/'finalM64_attempt001',rr);saved=read(EXP/'M64_ROWS_PRIVATE.json');assert scored==saved;assert summary(scored)==read(EXP/'M64_SUMMARY.json')
 # Exact content exclusion against previous recent experiments; includes R23 K64.
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};old=rows('D208')+rows('V64')
 for name in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:old+=read(ROOT/name)
 im={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in old};pc={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in old};h=read(EXP.parent/'input/M_CONTENT_HASHES_PRIVATE.json');assert len(h)==len({r['decoded_pixel_sha256'] for r in h})==len({r['canonical_XYZ_sha256'] for r in h})==64;assert not im&{r['decoded_pixel_sha256'] for r in h};assert not pc&{r['canonical_XYZ_sha256'] for r in h}
 pre=read(EXP/'M64_PRELABEL_LOCK.json');assert pre['prediction_seal_sha256']==sha(BASE/'finalM64_attempt001/SEALED.json');assert pre['selection_sha256']==sha(EXP/'LAYER_SELECTED_LOCK.json');sel=read(EXP/'LAYER_SELECTED_LOCK.json');assert sel['best']==sel['per_layer'][0];source=sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth');assert source=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 # Recompute direct M Qwen choice and frozen box pool correspondence.
 changed=[];allnegative=[];comparisons=[('Qwen_G_direct','source'),('CLIP_G_direct','Qwen_G_direct'),('no_Qwen_G_direct','Qwen_G_direct'),('selected_TTA_plus_current_G','Qwen_G_direct'),('query_mlp_transfer','source'),('text_mlp_transfer','source'),('query_last_transfer','source'),('rank_adapter_transfer','source'),('same_A_cache_plus_current_G','Qwen_G_direct')]
 for r,x in zip(rr,scored):
  z=feature(r);e=load_npz(BASE/'bankM64_attempt001/samples_PRIVATE'/(r['id']+'.npz'));assert int((z['scores']+2*e['vision']+2*e['position']+2*e['relation']).argmax())==x['q']['Qwen_G_direct'];a=load_npz(BASE/'preparedM64_attempt001/arrays_PRIVATE'/(r['id']+'.npz'));weights=a['weights'];prep=read(BASE/'preparedM64_attempt001/cases_PRIVATE'/(r['id']+'.json'))
  for arm,ref in comparisons:
   d=x['ious'][arm]-x['ious'][ref]
   if x['q'][arm]!=x['q'][ref]:
    qa,qb=x['q'][arm],x['q'][ref];ma=int(weights[qa].argmax()) if weights.shape[1] and weights[qa].max()>0 else None;mb=int(weights[qb].argmax()) if weights.shape[1] and weights[qb].max()>0 else None
    rec={'alias':x['alias'],'platform':x['platform'],'arm':arm,'reference':ref,'reference_iou':x['ious'][ref],'new_iou':x['ious'][arm],'delta_pp':float(d*100),'predicted_boxes_mutual_aabb_iou':aabb_iou(z['boxes'][qa],z['boxes'][qb]),'same_dominant_RGB_instance_mask':ma==mb if ma is not None and mb is not None else None,'association_is_not_identity_gold':True};changed.append(rec)
    if d< -1e-10:allnegative.append(rec)
 effects={a+'_vs_'+b:ci(scored,a,b) for a,b in comparisons};dump(EXP/'EFFECTS.json',effects);dump(EXP/'ALL_CHANGED_CASES.json',changed);dump(EXP/'ALL_NEGATIVE_CASES.json',allnegative)
 fits=read(BASE/'screen_attempt001/FIT_RECEIPTS.json');assert len(fits)==32 and all(x['parameter_L2_delta']>0 and x['trace'][0]['gradient_L2']>0 for x in fits);closures=read(BASE/'closureM64_attempt001/CLOSURE_AUDIT.json');assert len(closures)==4 and all(x['box_max_error']==0 and x['source_restored_bitexact'] and x['nonmatching_state_unchanged'] for x in closures)
 selectedchanges=[r for r in changed if r['arm']=='selected_TTA_plus_current_G'];mechanism={'selected_hybrid_vs_direct_G_query_changes':len(selectedchanges),'same_associated_mask_changes':sum(r['same_dominant_RGB_instance_mask'] is True for r in selectedchanges),'different_associated_mask_changes':sum(r['same_dominant_RGB_instance_mask'] is False for r in selectedchanges),'unknown_association_changes':sum(r['same_dominant_RGB_instance_mask'] is None for r in selectedchanges),'identity_gold_not_available':True,'all_changes':selectedchanges};dump(EXP/'MECHANISM_AUDIT.json',mechanism)
 audit={'state':'PASSED','source_sha256':source,'all_seals_verified':names,'scoring_independently_recomputed_exact':True,'M64_content_disjoint':True,'scenes':read(EXP.parent/'input/EXPOSURE.json')['scenes'],'D208_unlabeled_G_fit_only':True,'V64_exposed_labels_selected_layers':True,'D208_exposed_labels_selected_CLIP_scale':True,'no_current_M_GT_or_G_used_to_fit':True,'full_VAL_historical_exposure_disclosed':True,'all336_source_features_and_all64_final_boxes_frozen':True,'all_four_interfaces_nonzero_gradients_and_parameter_updates':True,'full_model_closure256forwards':True,'source_original_fingerprint_restored':True,'all_negatives_preserved':True,'runtime_G_and_fit_do_not_read_task_annotations':True,'source_actual_input_XYZ_only':True,'source_training_dataset_also_returns_XYZ':True,'use_color_flag_not_proof_of_RGB_input':True};dump(EXP/'CPU_AUDIT.json',audit);print('AUDIT24_PASSED');print('TTA_EXTRA',effects['selected_TTA_plus_current_G_vs_Qwen_G_direct']);print('MECHANISM',mechanism)
if __name__=='__main__':main()

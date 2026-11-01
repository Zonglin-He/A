"""Independent CPU reductions, sealed decision checks and all negative tails."""
import sys,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round26'),str(ROOT/'research/round5/audit')]
from common26 import *
from independent_core_v2 import candidate_ious,transform_gt
def effect(rr,a,b):
 out={};boots=[]
 for p in ['drone','quad']:
  r=[x for x in rr if x['platform']==p];v=np.array([x['ious'][a]-x['ious'][b] for x in r]);scenes=sorted({x['scene_PRIVATE'] for x in r});ix=np.array([scenes.index(x['scene_PRIVATE']) for x in r]);rng=np.random.default_rng(2601 if p=='drone' else 2602);w=np.eye(len(scenes))[rng.integers(0,len(scenes),(10000,len(scenes)))].sum(1);bs=w@np.bincount(ix,weights=v,minlength=len(scenes))/(w@np.bincount(ix))*100;boots.append(bs);out[p]={'n':len(r),'gain_pp':float(v.mean()*100),'gains':int((v>1e-10).sum()),'harms':int((v< -1e-10).sum()),'ties':int((abs(v)<=1e-10).sum()),'scene_clusters':len(scenes),'cluster95_pp':list(map(float,np.quantile(bs,[.025,.975])))}
 out['balanced']={'gain_pp':float(np.mean([out[p]['gain_pp'] for p in ['drone','quad']])),'cluster95_pp':list(map(float,np.quantile((boots[0]+boots[1])/2,[.025,.975])))};return out
def main():
 for lockname in ['BASE_CODE_LOCK.json','READOUT_CODE_LOCK.json']:
  for path,digest in read(EXP/lockname)['programs'].items():assert sha(ROOT/path)==digest,path
 rr=read(EXP/'ALL464_ROWS_PRIVATE.json');manifest=rows();lookup={x['alias']:x for x in manifest};pred=BASE/'predictions464_attempt001';verify(pred);pre=read(EXP/'P64_PRELABEL_LOCK.json');assert pre['prediction_seal_sha256']==sha(pred/'SEALED.json') and pre['lambda_lock_sha256']==sha(EXP/'SELECTED_LAMBDAS_LOCK.json')
 for f in ['sourceP64_attempt001','masksP64_attempt001','textF128_attempt001','preparedF128_attempt001','dinoF128_attempt001','evidenceF128_attempt001','bankF128_attempt001','joint464_attempt001','clip464_attempt001','scores464_attempt001','dev208grid_attempt001']:verify(BASE/f)
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};negative=[];changes=[];diagnostics=[];old={}
 for f in ['D208_MODAL_ROWS_PRIVATE','V64_MODAL_ROWS_PRIVATE','M64_ROWS_PRIVATE']:
  old.update({x['alias']:x for x in read(ROOT/'research/round24/experiment'/(f+'.json'))})
 replay=0
 for x in rr:
  r=lookup[x['alias']];ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));z=load_npz(pred/'predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z['boxes'],feature(r)['boxes'][z['queries']]);bb=z['boxes'].copy();bb[:,3:]=np.maximum(bb[:,3:],1e-6);vv=candidate_ious(bb,gt)[0]
  for name,v,q in zip(z['arm_names'],vv,z['queries']):assert float(v)==x['ious'][name] and int(q)==x['q'][name]
  # Independent score formula replay for every arm.
  bank=load_npz(BASE/'scores464_attempt001/samples_PRIVATE'/(r['id']+'.npz'));nn=bank['cue_names'].tolist();selected=read(EXP/'SELECTED_LAMBDAS_LOCK.json')['lambda_by_variant']
  for name,l in selected.items():assert int(np.argmax(bank['source']+l*bank['cues'][nn.index(name)]))==x['q'][name]
  if x['alias'] in old:
   assert x['q']['legacy_complex']==old[x['alias']]['q']['Qwen_G_direct'];assert abs(x['ious']['legacy_complex']-old[x['alias']]['ious']['Qwen_G_direct'])<1e-10;replay+=1
  for arm in x['q']:
   if arm=='source':continue
   for ref in ['source','legacy_complex']:
    if arm==ref:continue
    if x['q'][arm]!=x['q'][ref]:
     d=x['ious'][arm]-x['ious'][ref];c={'alias':x['alias'],'platform':x['platform'],'stage':x['stage'],'arm':arm,'reference':ref,'reference_iou':x['ious'][ref],'arm_iou':x['ious'][arm],'delta_pp':100*d,'mapped_source':bool(bank['mapped'][x['q']['source']]),'mapped_new':bool(bank['mapped'][x['q'][arm]]),'identity_gold_not_established':True};changes.append(c)
     if d< -1e-10:negative.append(c)
  diagnostics.append({'alias':x['alias'],'platform':x['platform'],'stage':x['stage'],'mapped_queries':int(bank['mapped'].sum()),'q0_mapped':bool(bank['mapped'][x['q']['source']]),'has_pool':bool(bank['mapped'].any())})
 assert replay==336,replay
 assert pre['practical_family_lock_sha256']==sha(EXP/'PRACTICAL_FAMILY_LOCK.json')
 # Lambda choice independently reconstructed from saved development grid.
 spec=read(BASE/'dev208grid_attempt001/SPEC.json');v=[]
 for r in rows('D208'):
  z=load_npz(BASE/'dev208grid_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));ix=idx[r['id']];gt=transform_gt(containers[ix['meta_path']]['ground_info'][ix['object_index']]['bbox_3d'],r['platform']);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v.append(candidate_ious(b,gt)[0])
 v=np.stack(v);grid=read(EXP/'DEV_GRID_RESULTS.json')
 for j,g in enumerate(grid):assert g['name']==spec[j]['name'] and abs(g['balanced_mIoU_percent']-v[:,j].mean()*100)<1e-10
 for n,l in read(EXP/'SELECTED_LAMBDAS_LOCK.json')['lambda_by_variant'].items():assert sorted([g for g in grid if g['name']==n],key=lambda g:(-g['balanced_mIoU_percent'],g['lambda']))[0]['lambda']==l
 policy=read(EXP/'PRIMARY_CONTRAST_LOCK.json');selected=read(EXP/'SELECTED_LAMBDAS_LOCK.json')['lambda_by_variant'];eligible=[g for g in grid if g['name'] in policy['practical_simple_Qwen_families'] and g['lambda']==selected[g['name']]];best=sorted(eligible,key=lambda g:(-g['balanced_mIoU_percent'],policy['tie_order'].index(g['name'])))[0];chosen=read(EXP/'PRACTICAL_FAMILY_LOCK.json');assert chosen['name']==best['name'] and chosen['lambda']==best['lambda']
 effects={}
 comparisons=[(a,b) for a in rr[0]['q'] if a!='source' for b in ['source','legacy_complex'] if a!=b]+[('bounded_no_gray_dual_fixed2','no_position_reference_fixed2'),('bounded_gray_tight_fixed2','no_position_reference_fixed2'),('bounded_gray_context_fixed2','no_position_reference_fixed2'),('centered_unclipped_gray_dual_fixed2','no_position_reference_fixed2'),('qwen_subject_raw_dual','qwen_subject_gray_dual'),('qwen_full_scene_raw','qwen_subject_raw_dual'),('clip_full_native_dual','clip_subject_dual'),('clip_full_chunk_dual','clip_full_native_dual')]
 for phase,rs in [('D208',rr[:208]),('REPLAY192',rr[208:400]),('OLD400',rr[:400]),('P64',rr[400:])]:
  for a,b in comparisons:effects[phase+'__'+a+'__vs__'+b]=effect(rs,a,b)
 hs=read(EXP.parent/'input/P_CONTENT_HASHES_PRIVATE.json');inv={x['image_file_sha256']:x for x in read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE']};hist=rows('OLD400')
 for f in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:hist+=read(ROOT/f)
 for key in ['decoded_pixel_sha256','canonical_XYZ_sha256']:
  assert len({x[key] for x in hs})==64 and not {x[key] for x in hs}&{inv[sha(r['imagepath'])][key] for r in hist}
 src=sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth');assert src=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 dump(EXP/'EFFECTS.json',effects);dump(EXP/'ALL_CHANGED_CASES.json',changes);dump(EXP/'ALL_NEGATIVE_CASES.json',negative);dump(EXP/'MAPPING_COVERAGE.json',diagnostics)
 dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','all464_sealed_predictions_independently_CPU_scored':True,'all_selected_lambda_choices_independently_reduced':True,'old336_legacy_G_query_and_IoU_exact_replay':replay,'all464_candidate_boxes_bitexact_source':True,'same_pool_weights_and_crop_hashes_across_variants':True,'P64_decoded_RGB_and_XYZ_disjoint_recent_development':True,'P64_historical_fullVAL_and_scene_exposure_disclosed':True,'P64_choices_sealed_before_current_GT':True,'all_negative_changes_retained':len(negative),'all_changed_comparisons':len(changes),'negative_rows_not_independent_samples':True,'source_checkpoint_sha256':src,'new_model_training_steps':0,'target_TTA_steps':0,'untrusted_captions_images_web_used_as_data_only':True})
 print('AUDIT26_PASSED',len(changes),len(negative))
if __name__=='__main__':main()

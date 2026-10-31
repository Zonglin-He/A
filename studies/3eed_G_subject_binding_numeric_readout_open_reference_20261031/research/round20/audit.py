"""CPU evidence replay, independent counts, no model calls or new selection."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round20'))
from common_v2 import *
def main():
 names=['sourceW32_attempt001','sourceZ64_attempt001','text144_attempt001','textZ64_attempt001','pairs144_attempt001','pairsZ64_attempt001','dino144_attempt001','dinoZ64_attempt001','visual4_attempt001','visual144_attempt001','visualZ64_attempt001','direct144_attempt001','direct208_attempt002'];seals={n:verify(ROOT/'crossplatform/round20'/n) for n in names};verify(EXP/'scored144_attempt001');verify(EXP/'scored208_attempt002');inputs=read(EXP.parent/'input/ALL208_PRIVATE.json');rows=read(EXP/'scored208_attempt002/ROWS_PRIVATE.json');assert len(inputs)==len(rows)==208 and len({r['id'] for r in inputs})==208;summary=read(EXP/'scored208_attempt002/SUMMARY.json');attrs={'explicit_field_cases':0,'field_scored_cases':0,'old18_connected':0};references={'queries':0,'with_stable_detection':0,'multiple_stable_candidates':0,'environment_queries':0,'environment_with_stable_detection':0};valid=0;pairrows=[];refrows=[]
 oldaliases={x['alias'] for x in read(ROOT/'research/round19/experiment/TEXT_BINDING_PATCH_AUDIT.json')['rows']};sourcefp=[]
 for r in inputs:
  isZ=r['stage']=='Z';folder=lambda name:ROOT/'crossplatform/round20'/(name+'Z64_attempt001' if isZ else name+'144_attempt001');g=read(folder('text')/'cases_PRIVATE'/(r['id']+'.json'));v=read(folder('visual')/'cases_PRIVATE'/(r['id']+'.json'));d=read(folder('dino')/'cases_PRIVATE'/(r['id']+'.json'));p=read(ROOT/'crossplatform/round20/direct208_attempt002/cases_PRIVATE'/(r['id']+'.json'))
  with np.load(ROOT/'crossplatform/round20/direct208_attempt002/predictions_PRIVATE'/(r['id']+'.npz')) as z:assert np.array_equal(z['arm_boxes_raw6'],feature(r)['boxes'][z['query_indices']])
  if g['explicit_appearance_field']:
   attrs['explicit_field_cases']+=1;assert g['explicit_appearance_field']['text'] in g['subject_query'];attrs['field_scored_cases']+=bool(v['masks_PRIVATE'])
   if r['alias'] in oldaliases:assert v['masks_PRIVATE'] and p['actual_subject_attribute_question_involved'];attrs['old18_connected']+=1
  assert not v['free_generation_or_parsing_used'] and p['no_format_rejection'] and not p['parameters_updated']
  for ms in v['masks_PRIVATE'].values():
   assert len(ms)==2
   for m in ms:
    for k in ('full','category','gray_full','gray_category'):assert m[k]['generated_tokens']==0 and np.isfinite(m[k]['log_odds_yes_no']);valid+=1
  for ref in g['references']:
   det=d['objects'].get(ref['entity'])
   if det:
    n=len(det['stable_indices']);env=ref['head'].casefold() in ('tree','trees','wall','walls','building','buildings','fence','graffiti','pole','sign');references['queries']+=1;references['with_stable_detection']+=n>0;references['multiple_stable_candidates']+=n>1;references['environment_queries']+=env;references['environment_with_stable_detection']+=env and n>0;refrows.append({'alias':r['alias'],'stage':r['stage'],'platform':r['platform'],'reference_index':ref['entity'],'environmental_noun':env,'stable_regions':n,'semantic_identity_not_certified':True})
  for q in p['pair_diagnostics']:pairrows.append({'alias':r['alias'],'stage':r['stage'],'platform':r['platform'],**{k:v for k,v in q.items() if not k.endswith('_PRIVATE')}})
 assert attrs['old18_connected']==18
 for phase,pp in summary.items():
  for platform,arms in pp.items():
   rr=[r for r in rows if r['phase']==phase and r['platform']==platform]
   for arm,c in arms.items():assert c['n']==len(rr) and abs(c['mIoU_percent']-100*np.mean([r['ious'][arm] for r in rr]))<1e-12
 # The additional64 actual RGB/XYZ must not duplicate any current/prior set.
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};content={r['id']:(lookup[sha(r['imagepath'])]['decoded_pixel_sha256'],lookup[sha(r['imagepath'])]['canonical_XYZ_sha256']) for r in inputs};Z=[content[r['id']] for r in inputs if r['stage']=='Z'];other=[content[r['id']] for r in inputs if r['stage']!='Z']
 for k in (0,1):assert len({x[k] for x in Z})==64 and not {x[k] for x in Z}&{x[k] for x in other}
 assert sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';asset=read(ROOT/'crossplatform/round20/assets/DINO_ASSET_RECEIPT.json');assert sha(ROOT/'crossplatform/round20/assets/grounding_dino/model.safetensors')==asset['model_sha256'];assert seals['visual144_attempt001']['free_generated_tokens']==seals['visualZ64_attempt001']['free_generated_tokens']==0
 dump(EXP/'SUBJECT_ATTRIBUTE_AND_REFERENCE_AUDIT.json',{'subject_attributes':attrs,'references':references,'numeric_visual_forwards_in_final_sets':valid,'visual_format_rejections':0,'no_free_form_visual_generation':True,'all_numeric_outputs_finite':True,'reference_proposal_coverage_not_accuracy':True,'reference_rows':refrows});dump(EXP/'ALL_PAIR_DIAGNOSTICS.json',pairrows)
 receipt={'state':'R20_CPU_ARTIFACT_AUDIT_PASSED','all208_unique_rows_and_all_negatives_kept':True,'all_candidate_boxes_from_immutable_source_pool':True,'source_checkpoint_immutable':True,'all_seals_verified':True,'all_metric_denominators_recomputed':True,'newZ64_decoded_content_disjoint':True,'old18_subject_fields_actually_in_scored_questions':True,'readout_does_not_parse_strings':True,'DINO_asset_LFS_sha256_verified':True,'TTA_optimizer_steps':0,'current_direct_only':True,'seal_hashes':{n:sha(ROOT/'crossplatform/round20'/n/'SEALED.json') for n in names},'limits':['AlltargetVALhistoricallyexposed. W32development after firstgate failure; Z64fixed before newG outputs.','Semantic reference identity and point-mask/query mapping are not certified by detector scores.','Two Dronescenes/threeQuadscenes; clusteruncertainty reported, no fullVAL claims.']};dump(EXP/'CPU_REPLAY_AUDIT.json',receipt);print('R20_AUDIT_PASSED',attrs,references)
if __name__=='__main__':main()

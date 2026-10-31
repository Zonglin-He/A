"""Pre-label relative-preference readout of the same sealed binary evidence.
No extra VLM calls. Cancels candidate-specific YES/NO verbalizer bias by a fixed
same-scene caption counterfactual. This is a proposed preference, not a label.
The original absolute-confidence output and every refusal remain untouched.
"""
import sys,time,math
import numpy as np
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round13')]
from input_io import load_case
from scene_observation import build_scene_observation
from preferences import component_mapping

def relative_pref(old,scene,boxes,scores):
 ev=old['preference']['all_mask_evidence'];mapping=component_mapping(scene,boxes);obj=scene['objects'];comp=mapping['components'];out={'status':'unusable','reason':None,'positive_queries':[],'negative_queries':[],'direct_query':int(scores.argmax()),'target_mask':None,'relation_status':'unknown','absolute_attempt_reason':old['preference']['reason'],'component_diagnostics':mapping['component_diagnostics'],'query_mask_assignment':mapping['assignment'].tolist()}
 if not ev:out['reason']='no_instances';return out
 def odds(e,k):
  p=e[k]['probabilities'];return math.log(max(p['YES'],1e-12))-math.log(max(p['NO'],1e-12))
 full=np.array([odds(e,'full') for e in ev]);wrong=np.array([odds(e,'wrong_caption') for e in ev]);contrast=full-wrong
 t=int(full.argmax());winner=int(contrast.argmax());donor_winner=int(wrong.argmax());out.update(full_logodds=full.tolist(),wrong_logodds=wrong.tolist(),caption_contrast=contrast.tolist(),target_mask=t)
 if t!=winner or t==donor_winner:out['reason']='full_vs_counterfactual_rank_not_discriminating';return out
 others=np.delete(contrast,t)
 if contrast[t]<math.log(2) or (len(others) and contrast[t]-others.max()<math.log(2)):out['reason']='insufficient_counterfactual_margin';return out
 loc=ev[t]['location']['status'];out['relation_status']=loc
 if loc=='contradict':out['reason']='explicit_location_contradicts';return out
 pos=np.flatnonzero(mapping['assignment']==t)
 if not len(comp[t]) or not len(pos):out['reason']='target_component_has_no_3D_query';return out
 points=scene['visible_points_model'];tp=points[comp[t]];tc=np.median(tp,axis=0);te=np.ptp(tp,axis=0);negative_masks=[]
 for m,o in enumerate(obj):
  if m==t or not len(comp[m]) or o['predicted_class_id']!=obj[t]['predicted_class_id'] or full[t]-full[m]<math.log(2):continue
  pp=points[comp[m]];sep=np.linalg.norm(np.median(pp,axis=0)-tc);inter=np.count_nonzero(o['mask_bool']&obj[t]['mask_bool']);union=np.count_nonzero(o['mask_bool']|obj[t]['mask_bool'])
  if sep>max(.5,.5*(np.linalg.norm(te)+np.linalg.norm(np.ptp(pp,axis=0)))) and inter/max(union,1)<.05:negative_masks.append(m)
 neg=np.flatnonzero(np.isin(mapping['assignment'],negative_masks))
 if not len(neg):out['reason']='no_separated_same_class_relative_negative';return out
 out.update(status='usable',reason='counterfactual_relative_fullquery_component_preference',positive_queries=pos.tolist(),negative_queries=neg.tolist(),negative_masks=negative_masks,direct_query=int(pos[np.argmax(scores[pos])]))
 return out
def main():
 check_plan();lock=read(EXP/'RELATIVE_PRELABEL_LOCK.json')
 for rel,h in lock['programs'].items():assert sha(ROOT/rel)==h
 oldroot=ROOT/'crossplatform/round16/G_A32_attempt002';verify(oldroot)
 out=ROOT/'crossplatform/round16/G_A32_attempt003';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();inp=EXP.parent/'input'
 rows=read(inp/'A32_PRIVATE.json');cases={x['id']:x for x in read(inp/'A_CASES_PRIVATE.json')};index=read(inp/'A_FRAME_INDEX_PRIVATE.json');counts={};start=time.perf_counter()
 for r in rows:
  old=read(oldroot/'cases_PRIVATE'/(r['id']+'.json'));frame,source,bindings=load_case(r,cases[r['id']],index);h,w=frame['rgb'].shape[:2];scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=frame['instances'],image_sha256=bindings['image_sha256']);pref=relative_pref(old,scene,source['boxes_raw6'][0],source['text_scores'][0]);counts[pref['reason']]=counts.get(pref['reason'],0)+1
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'id_PRIVATE':r['id'],'alias':old['alias'],'platform':r['platform'],'bindings':bindings,'preference':pref})
 dump(out/'registered_prelabel_lock.json',lock);seal(out,'R16_A32_relative_preferences_before_GT_no_fitting',n=32,new_model_calls=0,reasons=counts,binary_evidence_seal_sha256=sha(oldroot/'SEALED.json'),seconds=time.perf_counter()-start,programs=lock['programs']);print('RELATIVE_PREFERENCES_SEALED',counts,flush=True)
if __name__=='__main__':main()

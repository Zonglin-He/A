"""Native partial preference: no global target winner or full pseudo identity.
An observed same-class component pair must reverse order under a fixed caption
counterfactual. Unknown candidates are neither positives nor negatives.
"""
import sys,time,math
import numpy as np
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round13')]
from input_io import load_case
from scene_observation import build_scene_observation
from preferences import component_mapping

def make_local(old,scene,boxes,scores):
 ev=old['preference']['all_mask_evidence'];mp=component_mapping(scene,boxes);obj=scene['objects'];comp=mp['components'];out={'status':'unusable','reason':None,'positive_queries':[],'negative_queries':[],'direct_query':int(scores.argmax()),'target_mask':None,'relation_status':'unknown','absolute_attempt_reason':old['preference']['reason'],'component_diagnostics':mp['component_diagnostics'],'query_mask_assignment':mp['assignment'].tolist(),'pair_candidates':[]}
 def odds(e,k):
  p=e[k]['probabilities'];return math.log(max(p['YES'],1e-12))-math.log(max(p['NO'],1e-12))
 full=np.array([odds(e,'full') for e in ev]);wrong=np.array([odds(e,'wrong_caption') for e in ev]);points=scene['visible_points_model'];pairs=[];mapped=[]
 for m in range(len(obj)):
  if len(comp[m]) and np.any(mp['assignment']==m):mapped.append(m)
 out['mapped_mask_count']=len(mapped)
 for m in mapped:
  if ev[m]['location']['status']=='contradict':continue
  for n in mapped:
   if m==n or obj[m]['predicted_class_id']!=obj[n]['predicted_class_id']:continue
   fg=float(full[m]-full[n]);reverse=float(wrong[n]-wrong[m])
   if min(fg,reverse)<math.log(2):continue
   a,b=points[comp[m]],points[comp[n]];sep=np.linalg.norm(np.median(a,axis=0)-np.median(b,axis=0));limit=max(.5,.5*(np.linalg.norm(np.ptp(a,axis=0))+np.linalg.norm(np.ptp(b,axis=0))));inter=np.count_nonzero(obj[m]['mask_bool']&obj[n]['mask_bool']);union=np.count_nonzero(obj[m]['mask_bool']|obj[n]['mask_bool'])
   if sep<=limit or inter/max(union,1)>=.05:continue
   pairs.append({'positive_mask':m,'negative_mask':n,'full_logodds_gap':fg,'wrong_caption_reverse_gap':reverse,'minimum_gap':min(fg,reverse)})
 out['pair_candidates']=pairs
 if not pairs:out['reason']='no_mapped_sameclass_caption_reversal_pair';return out
 pair=sorted(pairs,key=lambda p:(-p['minimum_gap'],p['positive_mask'],p['negative_mask']))[0];m=pair['positive_mask'];n=pair['negative_mask'];pos=np.flatnonzero(mp['assignment']==m);neg=np.flatnonzero(mp['assignment']==n);q0=int(scores.argmax());qpos=int(pos[np.argmax(scores[pos])]);qneg=int(neg[np.argmax(scores[neg])])
 out.update(status='usable',reason='local_sameclass_caption_reversal_preference',positive_queries=pos.tolist(),negative_queries=neg.tolist(),target_mask=m,negative_mask=n,relation_status=ev[m]['location']['status'],pair_positive_query=qpos,pair_negative_query=qneg,source_affected_by_local_pair=bool(q0 in neg),direct_query=qpos if q0 in neg else q0,selected_pair=pair)
 return out
def main():
 check_plan();lock=read(EXP/'LOCAL_PREFERENCE_PRELABEL_LOCK.json')
 for rel,h in lock['programs'].items():assert sha(ROOT/rel)==h
 oldroot=ROOT/'crossplatform/round16/G_A32_attempt002';verify(oldroot);out=ROOT/'crossplatform/round16/G_A32_attempt004';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();inp=EXP.parent/'input'
 rows=read(inp/'A32_PRIVATE.json');cases={x['id']:x for x in read(inp/'A_CASES_PRIVATE.json')};index=read(inp/'A_FRAME_INDEX_PRIVATE.json');counts={};start=time.perf_counter()
 for r in rows:
  old=read(oldroot/'cases_PRIVATE'/(r['id']+'.json'));frame,source,bindings=load_case(r,cases[r['id']],index);h,w=frame['rgb'].shape[:2];scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=frame['instances'],image_sha256=bindings['image_sha256']);pref=make_local(old,scene,source['boxes_raw6'][0],source['text_scores'][0]);counts[pref['reason']]=counts.get(pref['reason'],0)+1
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'id_PRIVATE':r['id'],'alias':old['alias'],'platform':r['platform'],'bindings':bindings,'preference':pref})
 dump(out/'registered_prelabel_lock.json',lock);seal(out,'R16_A32_local_pair_preferences_before_GT_no_fitting',n=32,new_model_calls=0,reasons=counts,binary_evidence_seal_sha256=sha(oldroot/'SEALED.json'),seconds=time.perf_counter()-start,programs=lock['programs']);print('LOCAL_PREFERENCES_SEALED',counts,flush=True)
if __name__=='__main__':main()

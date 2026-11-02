"""Current source Top1 versus distinct same-class visible instance candidates."""
import sys,time
import numpy as np
from common_v2 import *
from observations import load

sys.path.insert(0,str(ROOT/'crossplatform/round17'))
from pair_geometry import prepare
def choose(scene,z,raw):
 q0=int(z['scores'].argmax());support=np.array(raw['point_support_count']);cov=np.array(raw['point_support_fraction']);pi=np.array(raw['projection_iou']);objects=scene['objects'];M=len(objects);out={'q0_PRIVATE':q0,'pairs':[],'q0_mapping_count':0,'reason':'no_q0_mapping'}
 if not M:return out
 okay=(support>=3)&(cov>=.15)&(pi>=.1)&(z['boxes'][:,3:]>0).all(1)[:,None];am=np.flatnonzero(okay[q0]);out['q0_mapping_count']=len(am)
 if len(am)!=1:out['reason']='q0_mapping_ambiguous' if len(am) else 'no_q0_mapping';return out
 a=int(am[0]);out['source_mask_PRIVATE']=a
 for b in range(M):
  if b==a or objects[a]['predicted_class_id']!=objects[b]['predicted_class_id']:continue
  ma=objects[a]['mask_bool'];mb=objects[b]['mask_bool'];mi=float((ma&mb).sum()/max((ma|mb).sum(),1));pa=objects[a]['points_model'];pb=objects[b]['points_model']
  sep=float(np.linalg.norm(np.median(pa,0)-np.median(pb,0))) if len(pa) and len(pb) else None
  if mi>=.2 or sep is None or sep<.5 or support[q0,b]>.2*max(support[q0,a],1):continue
  qs=np.flatnonzero(okay[:,b]&(support[:,a]<=.2*np.maximum(support[:,b],1)));qs=qs[qs!=q0]
  if not len(qs):continue
  qb=max(qs,key=lambda q:(float(z['scores'][q]),-int(q)))
  out['pairs'].append({'mask_indices_PRIVATE':[a,b],'qA_PRIVATE':q0,'qB_PRIVATE':int(qb),'qB_score':float(z['scores'][qb]),'alternative_query_count':len(qs),'mask_iou':mi,'visible_surface_median_separation_m':sep,'identity_not_Gold':True})
 out['pairs']=sorted(out['pairs'],key=lambda p:(-p['qB_score'],p['qB_PRIVATE']))[:2]
 for i,p in enumerate(out['pairs']):p['pair_alias']=f'pair_{i:02d}'
 out['reason']='pairs_available' if out['pairs'] else 'no_distinct_sameclass_competitor_with_mapping';return out
def main():
 locked('PREPARE_Z_LOCK.json');verify(ROOT/'crossplatform/round20/sourceZ64_attempt001');out=ROOT/'crossplatform/round20/pairsZ64_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();stats={};start=time.perf_counter()
 for i,r in enumerate(read(EXP.parent/'input/Z64_PRIVATE.json')):
  frame,scene,binding=load(r);z=feature(r);geometry=prepare(scene,z['boxes'],r['id']);p=choose(scene,z,geometry);p.update(id_PRIVATE=r['id'],alias=r['alias'],platform=r['platform'],stage=r['stage'],bindings=binding);dump(out/'cases_PRIVATE'/(r['id']+'.json'),p);key=r['stage']+'_'+r['platform'];stats.setdefault(key,{});stats[key][p['reason']]=stats[key].get(p['reason'],0)+1
  if (i+1)%16==0:print('PAIRS',i+1,flush=True)
 dump(out/'COUNTS.json',stats);seal(out,'R19_current_Top1_pairs_prepared_no_GT',n=64,seconds=time.perf_counter()-start);print(stats)
if __name__=='__main__':main()

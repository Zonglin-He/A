"""Recompute newly selected entity phrases with same frozen3EED and anchors."""
from structural_core87 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
start=time.perf_counter();base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.set_num_threads(4);fp=fingerprint(base);O=B87/'structural_evidence336_attempt002';(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False);(O/'records_PRIVATE').mkdir();calls=0;rows=[]
with torch.no_grad():
 for nr,r in enumerate(read(H87/'experiment/DEV336_PRIVATE.json')):
  sid=r['id'];graph=read(B87/'role_inventory336_attempt003/cases_PRIVATE'/(sid+'.json'))['graph'];meta=read(B86/'evidence366_attempt002/cases_PRIVATE'/(sid+'.json'));old=load(B85/'prediction366_attempt002/samples_PRIVATE'/(sid+'.npz'));previous={e['phrase']:old['entity_scores'][e['source_entity_index']] for e in meta['graph']['entities']};z0,z,d=item80(r);es=[]
  for e in graph['entities']:
   if e['phrase'] in previous:score=previous[e['phrase']]
   else:score=phrase85(base,z,e['phrase']).cpu().numpy();calls+=1
   es.append(score)
  es=np.array(es).reshape(-1,256);ev=load(B85/'observations366_attempt001/samples_PRIVATE'/(sid+'.npz'));orig=load(B87/'evidence336_attempt002/samples_PRIVATE'/(sid+'.npz'));A=orig['alias'];prior=old['scores'][old['arm_names'].tolist().index('frozen_bi')];sample={'alias':A,'source_boxes':z0['boxes'],'source_entity_scores':es};rec={}
  for v in MODES87+['wrong']:
   p,info,b=target87(prior,es,graph,ev,A,mode='full' if v=='wrong' else v,seed=sid,mismatch=v=='wrong');sample[v+'_pi']=p;sample[v+'_beliefs']=b;rec[v]=info
  np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),**sample);dump(O/'records_PRIVATE'/(sid+'.json'),rec);rows.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],**rec['full']})
  if (nr+1)%64==0:print('STRUCTURAL_PREP',nr+1,flush=True)
assert fingerprint(base)==fp;dump(H87/'STRUCTURAL_EVIDENCE_PREPARATION_ROWS.json',rows);dump(O/'COUNTS.json',{'n':336,'new_own_model_phrase_suffix_calls':calls,'seconds':time.perf_counter()-start,'GT_read':False});seal(O,'R87_CORRECTED_ROLE_TARGETS_SOURCE_FROZEN_PHRASES_SEALED',n=336);print(read(O/'COUNTS.json'))

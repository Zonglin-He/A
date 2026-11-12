"""Freeze soft neighborhoods and all component targets before task scoring."""
from core87 import *
import time
start=time.perf_counter();O=B87/'evidence336_attempt002';(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False);(O/'records_PRIVATE').mkdir();stats=[]
for n,r in enumerate(read(H87/'experiment/DEV336_PRIVATE.json')):
 sid=r['id'];ev,graph,scores=prepared87(r);z0=prefix(r);g=load(casepath(r));A=alias87(z0['boxes'],ev['support'],z0['vis0'][0],g['cell'],g['visible']);sample={'alias':A,'source_boxes':z0['boxes']};rec={};old=load(B85/'prediction366_attempt002/samples_PRIVATE'/(sid+'.npz'));prior=old['scores'][old['arm_names'].tolist().index('frozen_bi')]
 for v in MODES87+['wrong']:
  p,info,belief=target87(prior,scores,graph,ev,A,mode='full' if v=='wrong' else v,seed=sid,mismatch=v=='wrong');sample[v+'_pi']=p;sample[v+'_beliefs']=belief;rec[v]=info
 np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),**sample);dump(O/'records_PRIVATE'/(sid+'.json'),rec);stats.append({'row':'DEV'+str(n+1).zfill(4),'platform':r['platform'],**rec['full']});
 if (n+1)%64==0:print('PREPARE87',n+1,flush=True)
dump(H87/'EVIDENCE_PREPARATION_ROWS.json',stats);dump(O/'COUNTS.json',{'n':336,'targets_per_input':9,'seconds':time.perf_counter()-start,'new_model_calls':0,'GT_read':False});seal(O,'R87_SOFT_ALIAS_AND_ALL_ENTITY_TARGETS_SEALED_BEFORE_TTA',n=336);print(read(O/'COUNTS.json'))

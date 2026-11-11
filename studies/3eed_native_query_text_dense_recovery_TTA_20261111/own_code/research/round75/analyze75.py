"""Postseal overall task, held semantic recovery and clustered descriptive results."""
from common75 import *
import collections
P=ROOT/'crossplatform/round75/predicted336_attempt001';tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');methods=[];aux=[];frames=collections.defaultdict(list);raw=rows();scenes=sorted({(r['platform'],r['scene']) for r in raw});sm={x:f'RECORDING_{i+1:02d}' for i,x in enumerate(scenes)};sr=[]
for scope,tt in [('ALL336',tasks)]+[(s,[x for x in tasks if x['split']==s]) for s in sorted({x['split'] for x in tasks})]:
 for pl in ['balanced','drone','quad']:
  uu=tt if pl=='balanced' else [x for x in tt if x['platform']==pl]
  if not uu:continue
  for k in ARMS:
   a=np.array([x['IoU'][k] for x in uu]);ref='held_mask_source' if k.startswith('held_mask_') else 'frozen_mask_probability';b=np.array([x['IoU'][ref] for x in uu]);delta=a-b;methods.append({'scope':scope,'platform':pl,'method':k,'n':len(uu),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100),'reference':ref,'gain_vs_reference_pp':float(delta.mean()*100),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'query_changes':sum(x['queries'][k]!=x['queries'][ref] for x in uu),'rescue25':int(((a>.25)&(b<=.25)).sum()),'break25':int(((a<=.25)&(b>.25)).sum()),'largest_positive_contribution_pp':float(delta.max()/len(uu)*100),'remainder_after_largest_pp':float((delta.sum()-delta.max())/len(uu)*100)})
for x,r in zip(tasks,raw):
 a=read(P/'diagnostics_PRIVATE'/(r['id']+'.json'))
 for kind,v in a.items():aux.append({'row':x['row'],'platform':x['platform'],'split':x['split'],'method':kind,**v})
 key=(x['platform'],sha(r['imagepath']),sha(r['pointpath']));frames[key].append(x)
for t,alias in sm.items():
 tt=[x for x in tasks if (x['platform'],x['scene_PRIVATE'])==t]
 for k in ARMS:sr.append({'platform':t[0],'recording_alias':alias,'method':k,'n':len(tt),'mIoU_percent':float(np.mean([x['IoU'][k] for x in tt])*100)})
fr=[]
for i,(key,tt) in enumerate(sorted(frames.items())):
 for k in ARMS:fr.append({'frame_alias':f'FRAME_{i+1:04d}','platform':key[0],'method':k,'descriptions':len(tt),'mean_IoU':float(np.mean([x['IoU'][k] for x in tt]))})
auxsum=[]
for pl in ['balanced','drone','quad']:
 for kind in ['source']+ADAPT:
  aa=[x for x in aux if x['method']==kind and (pl=='balanced' or x['platform']==pl)];keys=['held_row_KL','held_column_KL','held_row_top1_agreement','held_column_top1_agreement','full_matrix_L2'];auxsum.append({'platform':pl,'method':kind,'n':len(aa),**{k:float(np.mean([x[k] for x in aa])) for k in keys}})
steps=read(H/'ALL_OPTIMIZER_STEP_ROWS.json');counts={k:{'n':sum(s['method']==k for s in steps),'accepted':sum(s['method']==k and s['accepted'] for s in steps),'rejected':sum(s['method']==k and not s['accepted'] for s in steps),'loss_before_mean':float(np.mean([s['loss_before'] for s in steps if s['method']==k])),'loss_after_mean':float(np.mean([s['loss_after'] for s in steps if s['method']==k]))} for k in ADAPT}
for name,v in [('ALL_METHOD_ROWS',methods),('REGION_FREE_HELD_SEMANTIC_ROWS',aux),('HELD_SEMANTIC_SUMMARY',auxsum),('ALL_SCENE_ROWS',sr),('ALL_FRAME_CLUSTER_ROWS',fr),('OPTIMIZER_SUMMARY',counts)]:dump(H/(name+'.json'),v)
print({'n':336,'recordings':len(scenes),'frames':len(fr)//len(ARMS),'methods':[(x['method'],x['mIoU_percent'],x['gain_vs_reference_pp'],x['improved'],x['worsened']) for x in methods if x['scope']=='ALL336' and x['platform']=='balanced'],'held':auxsum[:4],'optimizer':counts})

"""Independent saved-scalar aggregation for source/target, all negative tails."""
from pathlib import Path
import sys,itertools,collections,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round76'))
from common76 import read,dump,sha,load,source_rows
H=ROOT/'research/round78';TR=ROOT/'crossplatform/round78/trained_source_attempt001';SC=H/'score_dev336_attempt001';ss=read(SC/'SEALED.json')
for f,h in ss['files'].items():assert sha(SC/f)==h
tasks=read(SC/'ALL_TASK_ROWS_PRIVATE.json');arms=read(SC/'SCORING_RECEIPT.json')['arm_names'];summary=read(SC/'TASK_SUMMARY.json');checks=0
for scope,sub in [('ALL336',tasks)]+[(s,[r for r in tasks if r['split']==s]) for s in sorted({r['split'] for r in tasks}) if s!='ALL336']:
 for pl in ['drone','quad']:
  rr=[r for r in sub if r['platform']==pl]
  for a in arms:
   v=np.asarray([r['IoU'][a] for r in rr]);z=summary[scope][pl][a];assert abs(v.mean()*100-z['mIoU_percent'])<1e-12 and np.mean(v>.25)*100==z['Acc25_percent'] and np.mean(v>.5)*100==z['Acc50_percent'];checks+=3
scenes=read(SC/'ALL_SCENE_SUMMARIES_PRIVATE.json');neg=read(SC/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');expected=[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tasks for a,b in itertools.permutations(arms,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];assert expected==[(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in neg]
method=[]
for scope,data in summary.items():
 for pl in ['drone','quad']:
  for a in arms:method.append({'scope':scope,'platform':pl,'arm':a,**data[pl][a]})
scenevalues={pl:sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in ['drone','quad']};sr=[]
for r in scenes:
 for a in arms:sr.append({'scope':r['scope'],'platform':r['platform'],'scene_alias':('D' if r['platform']=='drone' else 'Q')+'_S'+str(scenevalues[r['platform']].index(r['scene_PRIVATE'])+1).zfill(2),'n':r['n'],'arm':a,'mIoU_percent':r['mIoU_percent'][a]})
source=source_rows('VAL128');sel=read(TR/'SELECTED_SOURCE_ONLY.json');eps=sel['shared_epoch'];srcpred={b:load(TR/'val_PRIVATE'/(b+'_e'+str(eps)+'.npz')) for b in ['joint','meta']};old=load(ROOT/'crossplatform/round76/trained_source_attempt002/val_PRIVATE/bidirectional_e06.npz');src=[];groups=sorted({r['scene'] for r in source})
for i,r in enumerate(source):
 yy=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];rr={'R76_frozen':float(old['IoU'][i])}
 for b in ['joint','meta']:
  for j in [0,1]:
   a=b+str(j);q=int(srcpred[b]['scores'][i,j].argmax());v=float(yy[q]);assert v==srcpred[b]['IoU'][i,j];rr[a]=v
 for a,v in rr.items():src.append({'row':f'SOURCE128_{i+1:05d}','recording_alias':'W_S'+str(groups.index(r['scene'])+1).zfill(3),'arm':a,'IoU':v})
source_summary={}
for a in ['R76_frozen','joint0','joint1','meta0','meta1']:
 v=np.array([r['IoU'] for r in src if r['arm']==a]);source_summary[a]={'n':128,'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100)}
src_rows={r['row']:{x['arm']:x['IoU'] for x in src if x['row']==r['row']} for r in src};srcneg=[{'row':row,'arm':a,'reference':b,'delta_pp':100*(z[a]-z[b])} for row,z in src_rows.items() for a,b in itertools.permutations(z,2) if z[a]-z[b]<-1e-10];srcscene=[{'recording_alias':s,'arm':a,'n':len([r for r in src if r['recording_alias']==s and r['arm']==a]),'mIoU_percent':float(np.mean([r['IoU'] for r in src if r['recording_alias']==s and r['arm']==a])*100)} for s in sorted({r['recording_alias'] for r in src}) for a in source_summary]
hist=read(TR/'HISTORY.json');publichist=[{k:v for k,v in r.items() if k not in ['state_path','state_sha256']} for r in hist];details=[]
for b in ['joint','meta']:
 for epoch in range(1,5):details.extend([{'branch':b,'epoch':epoch,**r} for r in read(TR/'val_PRIVATE'/(b+'_e'+str(epoch)+'_SCALARS.json'))])
grad=read(H/'SOURCE_GRADIENT_SCALAR_ROWS.json');gs={}
for b in ['joint','meta']:
 rr=[r for r in grad if r['branch']==b];gs[b]={'n':len(rr),'mean_gradient_cosine':float(np.mean([r['aux_ground_gradient_cosine'] for r in rr if r['aux_ground_gradient_cosine'] is not None])),'ground_loss_favorable_first_order_n':sum(r['ground_gradient_dot_SGD_delta']<0 for r in rr),'ground_loss_unfavorable_first_order_n':sum(r['ground_gradient_dot_SGD_delta']>0 for r in rr)}
frame=[]
for pl in ['drone','quad']:
 # Frames may carry multiple descriptions; frame IDs are local ordinal aliases.
 lookup={r['row']:r for r in tasks if r['platform']==pl};delta=np.array([r['IoU']['meta1']-r['IoU']['meta0'] for r in lookup.values()]);frame.append({'platform':pl,'n':len(delta),'meta_gain_pp':float(delta.mean()*100),'improved':int((delta>1e-10).sum()),'worsened':int((delta<-1e-10).sum()),'largest_positive_contribution_platform_pp':float(max(0,delta.max())*100/len(delta)),'largest_negative_contribution_platform_pp':float(min(0,delta.min())*100/len(delta))})
for name,x in [('ALL_METHOD_ROWS',method),('ALL_SCENE_ROWS',sr),('SOURCE128_TASK_SCALARS',src),('SOURCE128_ORDERED_NEGATIVES',srcneg),('SOURCE128_RECORDING_SCALARS',srcscene),('SOURCE_TRAINING_HISTORY_ROWS',publichist),('SOURCE_ALL_EPOCH_DIAGNOSTIC_ROWS',details),('SOURCE128_SUMMARY',source_summary),('SOURCE_GRADIENT_SUMMARY',gs),('TARGET_CONTRIBUTION_ROWS',frame)]:dump(H/(name+'.json'),x)
out=H/'independent_scalar_audit_attempt001';out.mkdir();record={'state':'PASS_R78_ALL_SAVED_TARGET_SOURCE_STATUS_AND_ORDERED_NEGATIVE_SCALARS','n':336,'task_values':4032,'ordered_negatives':len(neg),'source_task_values':len(src),'source_ordered_negatives':len(srcneg),'source_recording_rows':len(srcscene),'checks':checks,'source_selected_epoch':eps,'source_only_selection':True,'no_new_GT_geometry_or_model_operations':True};dump(out/'AUDIT_RESULT.json',record);print({'source':source_summary,'source_gradient':gs,'target':summary['ALL336']['balanced_mIoU_percent']});print(record)

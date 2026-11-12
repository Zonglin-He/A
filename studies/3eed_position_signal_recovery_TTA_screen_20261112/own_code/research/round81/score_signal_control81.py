"""Score only new full-signal shuffled predictions; copied controls remain labeled."""
from core81 import *
import itertools
P=B81/'signal_control2304_attempt001';ss=read(P/'SEALED.json');a=read(H81/'independent_signal_control_audit_attempt001/AUDIT_RESULT.json');assert a['state'].startswith('PASS_') and a['prediction_seal_sha256']==sha(P/'SEALED.json');old=ROOT/'research/round69/score_all2304_attempt001';lf=old/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(lf)==read(old/'SEALED.json')['files'][lf.name];y={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(lf)};cohort=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');prior={}
for part in ['VAL1280','TRAIN1024']:
 for r in read(H81/('score_'+part.lower()+'_attempt001/ALL_TASK_ROWS_PRIVATE.json')):prior[r['id_PRIVATE']]=r
rows=[];private=[];trialrows=[];endpoints=read(H81/'SIGNAL_CONTROL_ENDPOINT_SCALAR_ROWS.json')
for i,r in enumerate(allrows80()[336:]):
 sid=r['id'];f=P/'samples_PRIVATE'/(sid+'.npz');assert sha(f)==ss['files'][f.relative_to(P).as_posix()];z=load(f);oldr=prior[sid];ious={k:oldr['IoU'][k] for k in ['frozen_bi','frozen_direct3','all__ALL']};ious['all__ALL__shuffle']=float(y[sid][int(z['query'])]);row={'row':'E'+str(i+1).zfill(5),'cohort':cohort[sid],'platform':r['platform'],**{k+'_IoU':v for k,v in ious.items()}};rows.append(row);private.append({**row,'scene_PRIVATE':r['scene']})
 for j,t in enumerate(read(P/'records_PRIVATE'/(sid+'.json'))['attempts']):trialrows.append({'row':row['row'],'cohort':row['cohort'],'platform':r['platform'],'arm':'all__ALL__shuffle','attempt':j+1,**t})
summary={};scenes=[];neg=[];arms=['frozen_bi','frozen_direct3','all__ALL','all__ALL__shuffle']
for part,n in [('VAL1280',1280),('TRAIN1024',1024)]:
 rr=[r for r in rows if r['cohort']==part];assert len(rr)==n;s={}
 for pl in ['drone','quad']:
  pr=[r for r in rr if r['platform']==pl];s[pl]={}
  for arm in arms:
   v=np.array([r[arm+'_IoU'] for r in pr]);s[pl][arm]={'n':len(pr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100)}
  prs=[r for r in private if r['cohort']==part and r['platform']==pl]
  for j,sc in enumerate(sorted({r['scene_PRIVATE'] for r in prs})):
   sr=[r for r in prs if r['scene_PRIVATE']==sc]
   for arm in arms:scenes.append({'cohort':part,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'arm':arm,'n':len(sr),'mIoU_percent':float(np.mean([r[arm+'_IoU'] for r in sr])*100)})
 s['balanced_mIoU_percent']={a:float(np.mean([s[p][a]['mIoU_percent'] for p in ['drone','quad']])) for a in arms};s['correct_minus_shuffle_pp']=s['balanced_mIoU_percent']['all__ALL']-s['balanced_mIoU_percent']['all__ALL__shuffle'];summary[part]=s
 for r in rr:
  for arm,ref in itertools.permutations(arms,2):
   if 'all__ALL__shuffle' not in [arm,ref]:continue
   delta=100*(r[arm+'_IoU']-r[ref+'_IoU'])
   if delta<-1e-10:neg.append({'row':r['row'],'cohort':part,'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':delta})
for n,x in [('SIGNAL_CONTROL_TASK_ROWS',rows),('SIGNAL_CONTROL_SUMMARY',summary),('SIGNAL_CONTROL_SCENE_ROWS',scenes),('SIGNAL_CONTROL_NEGATIVE_ROWS',neg),('SIGNAL_CONTROL_ATTEMPT_ROWS',trialrows)]:dump(H81/(n+'.json'),x)
dump(H81/'SIGNAL_CONTROL_SCORE_RECEIPT.json',{'state':'PASS_R81_NEW_FULL_SHUFFLE_CONTROL_POSTSEAL_SAVED_LABEL_JOIN','n':2304,'new_prediction_task_values':2304,'copied_existing_reference_values':2304*3,'unique_new_inputs':0,'postscore_exploratory_control':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'no_new_GT_geometry':True});print({p:s['balanced_mIoU_percent'] for p,s in summary.items()})

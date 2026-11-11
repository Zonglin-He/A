"""Independent saved-only scalar, negative, auxiliary and frame arithmetic."""
from common75 import *
from scipy.special import logsumexp
import itertools
P=ROOT/'crossplatform/round75/predicted336_attempt001';tt=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');lab={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};checks=0;negative=[]
for x in tt:
 for k in ARMS:assert x['IoU'][k]==lab[x['id_PRIVATE']][x['queries'][k]];checks+=1
 for k,ref in itertools.permutations(ARMS,2):
  d=x['IoU'][k]-x['IoU'][ref]
  if d< -1e-10:negative.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':d*100})
assert negative==read(H/'score_dev336_attempt001/ALL_ORDERED_NEGATIVE_COMPARISONS.json')
for r in read(H/'ALL_METHOD_ROWS.json'):
 uu=tt if r['scope']=='ALL336' else [x for x in tt if x['split']==r['scope']];uu=uu if r['platform']=='balanced' else [x for x in uu if x['platform']==r['platform']];a=np.array([x['IoU'][r['method']] for x in uu]);assert len(uu)==r['n'];assert abs(a.mean()*100-r['mIoU_percent'])<1e-12;assert abs((a>.25).mean()*100-r['Acc25_percent'])<1e-12;checks+=3
for a in read(H/'REGION_FREE_HELD_SEMANTIC_ROWS.json'):
 x=tt[int(a['row'][1:])-1];z=load(P/'diagnostics_PRIVATE'/(x['id_PRIVATE']+'.npz'));s=z['source_S_full'];t=z[a['method']+'_S_held'];rr=s-logsumexp(s,axis=1,keepdims=True);cc=s-logsumexp(s,axis=0,keepdims=True);tr=t-logsumexp(t,axis=1,keepdims=True);tc=t-logsumexp(t,axis=0,keepdims=True);assert abs(a['held_row_KL']-np.mean(np.sum(np.exp(rr)*(rr-tr),axis=1)))<1e-12;assert abs(a['held_column_KL']-np.mean(np.sum(np.exp(cc)*(cc-tc),axis=0)))<1e-12;assert a['held_row_top1_agreement']==np.mean(np.argmax(s,axis=1)==np.argmax(t,axis=1));checks+=3
for a in read(H/'HELD_SEMANTIC_SUMMARY.json'):
 r=[x for x in read(H/'REGION_FREE_HELD_SEMANTIC_ROWS.json') if x['method']==a['method'] and (a['platform']=='balanced' or x['platform']==a['platform'])]
 for k in ['held_row_KL','held_column_KL','held_row_top1_agreement','held_column_top1_agreement','full_matrix_L2']:assert abs(a[k]-np.mean([x[k] for x in r]))<1e-12;checks+=1
summary=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']
for pl in ['drone','quad']:
 uu=[x for x in tt if x['platform']==pl]
 for k in ARMS:
  a=np.array([x['IoU'][k] for x in uu]);s=summary[pl][k];assert abs(a.mean()*100-s['mIoU_percent'])<1e-12
  for st in ['unknown','fallback','no_parameter_update']:assert s[st+'_n']==sum(x['statuses'][k][st] for x in uu)
steps=read(H/'ALL_OPTIMIZER_STEP_ROWS.json');assert len(steps)==1008 and sum(x['accepted'] for x in steps)==1002 and not any(x['loss_after']>x['loss_before'] for x in steps if x['accepted'])
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();r={'state':'PASS_R75_ALL336_SAVED_TASK_NEGATIVES_AND_HELD_SEMANTIC_NUMBERS','n':336,'task_values':5376,'ordered_negatives':len(negative),'checks':checks,'accepted':1002,'no_new_GT_geometry_or_model_operations':True};dump(out/'AUDIT_RESULT.json',r);print(r)

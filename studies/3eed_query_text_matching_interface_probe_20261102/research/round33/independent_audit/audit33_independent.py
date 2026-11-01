"""R33 independent NumPy/CPU replay. No torch import, optimizer, or protected inputs."""
import sys,json,time,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'research/round5/audit')]
from common30 import read,dump,sha,verify,load_npz,rows,feature
from independent_core_v2 import transform_gt,candidate_ious
OUT=Path(__file__).resolve().parent
BASE=ROOT/'crossplatform/round33/interface_screen_attempt001'
EXP=ROOT/'research/round33/experiment'

def lse(x):
 m=x.max(-1);return m+np.log(np.exp(x-m[:,None]).sum(-1))
def project(s,x):
 x=x.astype(float)
 for j in ['0','2','4']:
  x=x@s[j+'.weight'].astype(float).T+s[j+'.bias'].astype(float)
  if j!='4':x=np.maximum(x,0)
 return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def head(s,layer,z):
 q=project(s,z['h']) if layer=='query_mlp' else z['query'].astype(float)
 t=project(s,z['text_h']) if layer=='text_mlp' else z['tokens'].astype(float)
 v=q@t.T/.07;mask=z['token_mask'].astype(bool)
 lo=lse(v[:,mask])-lse(v[:,~mask]);return lo,1/(1+np.exp(-lo))
def grouped(r,s,z):
 if r['stage']=='Q':return int(s.argmax())
 groups=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE']
 used=set();choices=[]
 for g in groups.values():
  used.update(g);rep=max(g,key=lambda j:(float(z['scores'][j]),-j));choices.append((float(s[g].max()),-rep,rep))
 for j in range(256):
  if j not in used:choices.append((float(s[j]),-j,j))
 return max(choices)[2]
def close(a,b,tol=1e-9):assert abs(float(a)-float(b))<tol,(a,b,tol)

def gradients():
 b=ROOT/'research/round33/gradient_diagnostic/attempt001';verify(b)
 ss=read(b/'STATE_SUMMARY.json');obs=read(b/'OBSERVATION_ROWS.json');pairs=read(b/'PAIRWISE_ROWS.json');result=[]
 for s in ss:
  z=load_npz(b/'gradients_PRIVATE'/(s['platform']+'_'+s['state']+'.npz'))
  G=z['gradient_by_observation'].astype(float);gp=z['gradient_pref'].astype(float);gk=z['gradient_keep'].astype(float)
  n=np.linalg.norm(G,axis=1);d=G@G.T;aliases=z['aliases'].tolist()
  close(np.linalg.norm(gp),s['pref_gradient_norm'],1e-10);close(np.linalg.norm(gk),s['keep_gradient_norm'],1e-10)
  close(np.linalg.norm(G.sum(0)-gp)/max(np.linalg.norm(gp),1e-30),s['perobs_sum_vs_original_pref_relative_error'],1e-12)
  for key,mask in [('all_exact_nonzero',n>0),('relative_active_1e6',(n>0)&(n>=n.max()*1e-6)),('absolute_active_1e12',(n>0)&(n>=1e-12))]:
   ii=np.flatnonzero(mask);dots=np.array([d[i,j] for k,i in enumerate(ii) for j in ii[k+1:]]);mass=np.array([n[i]*n[j] for k,i in enumerate(ii) for j in ii[k+1:]])
   v=s[key];assert v['observations']==len(ii) and v['pairs']==len(dots) and v['negative_pairs']==int((dots<0).sum())
   close(dots.sum()/mass.sum(),v['norm_weighted_cosine'],1e-12);close(np.maximum(-dots,0).sum()/abs(dots).sum(),v['negative_dot_energy_fraction'],1e-12)
  for o in [x for x in obs if (x['platform'],x['state'])==(s['platform'],s['state'])]:
   i=aliases.index(o['feedback_alias']);close(n[i],o['gradient_norm'],1e-10)
  for p in [x for x in pairs if (x['platform'],x['state'])==(s['platform'],s['state'])]:
   i=aliases.index(p['a']);j=aliases.index(p['b']);close(d[i,j],p['dot'],1e-9)
   if n[i]*n[j]:close(d[i,j]/(n[i]*n[j]),p['cosine'],1e-12)
   else:assert p['cosine'] is None
  if s['state']=='source':assert np.count_nonzero(gk)==0 and s['pref_keep_cosine'] is None
  else:close(gp@gk/(np.linalg.norm(gp)*np.linalg.norm(gk)),s['pref_keep_cosine'],1e-12)
  assert s['model_state_sha256_before']==s['model_state_sha256_after'] and s['optimizer_calls']==0 and s['parameters_unchanged'] and s['parameter_grad_buffers_still_None']
  result.append({'platform':s['platform'],'state':s['state'],'negative_dot_energy_fraction':s['all_exact_nonzero']['negative_dot_energy_fraction'],'pref_keep_cosine':s['pref_keep_cosine']})
 audit=read(ROOT/'research/round33/gradient_diagnostic/CPU_AUDIT.json');assert audit['state']=='PASSED' and audit['sealed_source']==sha(b/'SEALED.json')
 return {'state':'PASS','states':len(ss),'observation_rows':len(obs),'pair_rows':len(pairs),'results':result,'existing_CPU_AUDIT_rechecked':True,'gradient_generation_not_rerun':True}

def main():
 start=time.perf_counter();verify(BASE);feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={x['id_PRIVATE'] for x in feedback};assert len(feedback)==22 and len(fids)==19
 rr=[dict(r,split=split) for split in ['A208','DEV64'] for r in rows(split)];assert len(rr)==272 and set(r['stage'] for r in rr).isdisjoint({'R','S','T','U','F'})
 zm={r['id']:feature(r) for r in rr};initial=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz')
 src={layer:{k[len(prefix):]:v for k,v in initial.items() if k.startswith(prefix)} for layer,prefix in [('query_mlp','image.'),('text_mlp','text.')]}
 receipts=read(BASE/'FIT_RECEIPTS.json');assert len(receipts)==8 and sum(x['steps'] for x in receipts)==128
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};gt={}
 for r in rr:
  ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt[r['id']]=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'))
 scoremax=0.;lossmax=0.;normmax=0.;readouts=0;states=[];alltask=[];prefrows=[];contract={};pred={};task={};feas={}
 for rec in receipts:
  pl=rec['platform'];layer=rec['layer'];name=rec['name'];A=[r for r in rr if r['platform']==pl and r['split']=='A208'];ff=[f for f in feedback if f['platform']==pl];U=[r['id'] for r in A if r['id'] not in fids]
  assert len(ff)=={'drone':10,'quad':12}[pl] and len(U)==rec['keep_observations']=={'drone':96,'quad':93}[pl]
  contract[pl]={'preference_denominator':len(ff),'keep_n':len(U),'keep_sorted_id_sha256_PRIVATE':hashlib.sha256('\n'.join(sorted(U)).encode()).hexdigest()}
  assert rec['steps']==16 and rec['params']==184960
  s=load_npz(BASE/'states_PRIVATE'/(pl+'_'+name+'.npz'));assert s.keys()==src[layer].keys() and sum(x.size for x in s.values())==184960
  assert all(s[n].shape==src[layer][n].shape and np.isfinite(s[n]).all() for n in s)
  norm=np.sqrt(sum(np.square(s[n].astype(float)-src[layer][n].astype(float)).sum() for n in s));normmax=max(normmax,abs(norm-rec['parameter_delta_L2']));assert abs(norm-rec['parameter_delta_L2'])<2e-5 and norm>0
  initlo={r['id']:head(src[layer],layer,zm[r['id']])[0] for r in A}
  initloss=np.mean([np.logaddexp(0,-(initlo[f['id_PRIVATE']][f['positive_group_PRIVATE']].max()-initlo[f['id_PRIVATE']][f['negative_group_PRIVATE']].max())/.1) for f in ff]);lossmax=max(lossmax,abs(initloss-rec['trace'][0]['preference']));assert abs(initloss-rec['trace'][0]['preference'])<3e-4 and rec['trace'][0]['keep']==0
  if layer=='query_mlp' and rec['lr']==.0005:
   r30=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));assert all(np.array_equal(s[n],r30[n]) for n in s) and rec['R30_query_state_reproduction_maxerror']==0
  model_lo={};pm=[];finalq={}
  for split in ['A208','DEV64']:
   part=[r for r in rr if r['platform']==pl and r['split']==split];p=load_npz(BASE/'predictions_PRIVATE'/(pl+'_'+name+'_'+split+'.npz'));assert p['row_ids_PRIVATE'].tolist()==[r['id'] for r in part] and p['scores'].shape==(len(part),256)
   pred[pl,name,split]=p
   for j,r in enumerate(part):
    z=zm[r['id']];lo,pr=head(s,layer,z);err=float(abs(pr-p['scores'][j]).max());scoremax=max(scoremax,err);assert err<2e-5,(pl,name,r['alias'],err)
    q=int(p['queries'][j]);assert q==grouped(r,p['scores'][j],z);readouts+=1;finalq[r['id']]=q;model_lo[r['id']]=lo
    q0=int(z['scores'].argmax());b=z['boxes'][[q0,q]].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt[r['id']])[0]
    row={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':pl,'split':split,'layer':layer,'lr':rec['lr'],'model':name,'source_iou':float(v[0]),'iou':float(v[1]),'delta_vs_source_pp':float(100*(v[1]-v[0])),'q_PRIVATE':q,'sourceq_PRIVATE':q0,'query_changed':q!=q0,'has_feedback':r['id'] in fids};alltask.append(row);task[pl,name,r['id']]=float(v[1])
    if split=='DEV64':pm.append(float(v[1]))
  gaps=[];bp=bf=0
  for f in ff:
   lo=model_lo[f['id_PRIVATE']];gap=float(lo[f['positive_group_PRIVATE']].max()-lo[f['negative_group_PRIVATE']].max());gaps.append(gap)
   pos=finalq[f['id_PRIVATE']] in f['positive_group_PRIVATE'];bp+=int(f['label']=='B' and gap>0);bf+=int(f['label']=='B' and pos)
   prefrows.append({'model':name,'platform':pl,'id_PRIVATE':f['id_PRIVATE'],'label':f['label'],'gap':gap,'direction_satisfied':gap>0,'final_positive':pos})
  assert sum(g>0 for g in gaps)==rec['pairs_satisfied'] and bp==rec['B_pairs_satisfied'] and bf==rec['B_final_positive'];feas[pl,name]=(bp,bf)
  states.append({'platform':pl,'name':name,'layer':layer,'lr':rec['lr'],'DEV_mIoU_percent':100*np.mean(pm),'B_pair':bp,'B_final':bf,'state_parameter_delta_L2':float(norm),'final_preference_loss_CPU':float(np.logaddexp(0,-np.array(gaps)/.1).mean())})
 table=[]
 for layer in ['query_mlp','text_mlp']:
  for lr in [.0001,.0005]:
   name=layer+'_lr'+str(lr).replace('.','p');st=[x for x in states if x['name']==name];pm={x['platform']:x['DEV_mIoU_percent'] for x in st};B=sum(x['B_pair'] for x in st);F=sum(x['B_final'] for x in st)
   table.append({'name':name,'layer':layer,'lr':lr,'platform_mIoU_percent':pm,'balanced_mIoU_percent':float(np.mean(list(pm.values()))),'B_pairs_satisfied':B,'B_final_positive':F,'feasible':B>=5 and F>=4})
 saved=read(EXP/'DEV_INTERFACE_TABLE.json');assert len(saved)==4
 for a,b in zip(table,saved):
  assert a.keys()==b.keys()
  for k in a:
   if k=='platform_mIoU_percent':
    for pl in a[k]:close(a[k][pl],b[k][pl],1e-10)
   elif isinstance(a[k],float):close(a[k],b[k],1e-10)
   else:assert a[k]==b[k],(k,a[k],b[k])
 pick=lambda t:sorted(t,key=lambda x:(-x['balanced_mIoU_percent'],x['layer'],x['lr']))[0] if t else None
 lock=read(EXP/'SELECTED_INTERFACE_LOCK.json');assert pick([x for x in table if x['feasible']])==lock['selected_feasible_primary'];assert pick([x for x in table if x['layer']=='text_mlp'])==lock['besttext_unconstrained_diagnostic'];assert lock['source_R30_baseline_is_selected'] and not lock['text_feasible_candidate_exists'];assert lock['fit_prediction_seal_sha256']==sha(BASE/'SEALED.json')
 origdev=read(EXP/'DEV_INTERFACE_ROWS_PRIVATE.json');dmap={(r['alias'],r['layer'],r['lr']):r for r in origdev}
 for r in [x for x in alltask if x['split']=='DEV64']:
  o=dmap[r['alias'],r['layer'],r['lr']];close(o['iou'],r['iou'],1e-12);assert o['q_PRIVATE']==r['q_PRIVATE']
 r30={x['id_PRIVATE']:x for x in read(ROOT/'research/round30/experiment/ALL336_TASK_ROWS_PRIVATE.json')};repro=0
 for r in alltask:
  if r['model']=='query_mlp_lr0p0005':
   o=r30[r['id_PRIVATE']];assert r['q_PRIVATE']==o['q']['keep_selected'];close(r['iou'],o['ious']['keep_selected'],1e-12);repro+=1
 assert repro==272
 summary=[]
 for name in [x['name'] for x in table]:
  for split in ['A208','DEV64']:
   for pl in ['drone','quad']:
    for subset in (['all','feedback','no_feedback'] if split=='A208' else ['all']):
     rs=[r for r in alltask if r['model']==name and r['split']==split and r['platform']==pl and (subset=='all' or r['has_feedback']==(subset=='feedback'))];dd=np.array([r['delta_vs_source_pp'] for r in rs]);summary.append({'name':name,'split':split,'platform':pl,'subset':subset,'n':len(rs),'mIoU_percent':100*np.mean([r['iou'] for r in rs]),'mean_gain_vs_source_pp':float(dd.mean()),'improved':int((dd>1e-8).sum()),'harmed':int((dd< -1e-8).sum()),'query_changed':sum(r['query_changed'] for r in rs)})
 grad=gradients();dump(OUT/'GRADIENT_AUDIT_RESULT.json',grad);dump(OUT/'FIXED_CONTRACT_PRIVATE.json',contract);dump(OUT/'DEV_TABLE_INDEPENDENT.json',table);dump(OUT/'ALL1088_TASK_ROWS_PRIVATE.json',alltask);dump(OUT/'ALL88_PREFERENCE_ROWS_PRIVATE.json',prefrows);dump(OUT/'SUBSET_SUMMARY.json',summary)
 result={'state':'PASS_WITH_EXECUTION_SCOPE_LIMIT','models':8,'steps_from_code_and_receipts':128,'parameters_each':184960,'CPU_recomputed_score_vectors':readouts,'task_values_adapted':len(alltask),'source_task_values_also_recomputed':len(rr),'DEV_task_values_matched':len(origdev),'preference_states_checked':len(prefrows),'R30_state_bitexact_platforms':2,'R30_prediction_and_task_exact':repro,'max_score_replay_error':scoremax,'max_initial_loss_replay_error':lossmax,'max_parameter_delta_norm_error':normmax,'fixed_contract':{'denominators':{'drone':10,'quad':12},'keep':{'drone':96,'quad':93},'feedback_pairs':22,'feedback_observations':19},'selection_DEV_only_exact':True,'all_negatives_retained':True,'gradient_independent_recheck':grad['state'],'U_FINAL_inputs_labels_media_not_accessed':True,'no_GPU_training_autograd_called':True,'update_scope_evidence':'Only an isolated six-array MLP module is constructed and optimized in the inspected fit path, using frozen precomputed features. No full model is instantiated. Thus other branches cannot be changed by this path; no new full-model GPU closure was executed for unselected text heads.','limitations':['Independent audit replays saved CPU heads and geometry; it does not retrain states or independently trace every optimizer step.','Final text-head fitting is incomplete under the registered16-step/two-rate budget. This is not evidence all text-side adaptation is ineffective.','DEV is exposed historical development; no new U128 confirmation is warranted because the selected state exactly equals the prior R30 reference.','Data-access claims follow inspected control flow and audit scope, not an OS-level historical file-access trace.'],'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

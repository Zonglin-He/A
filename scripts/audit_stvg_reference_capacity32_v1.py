"""Independent scalar geometry, exhaustive oracle, Adam and metric audit. CPU."""
import os,sys,time,collections,itertools
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_reference_capacity32_v1 import D,P,verify,read,put,load,sha
from scripts.audit_stvg_sparse_state_route32_v1 import loss_numpy,independent_paired
from scripts.audit_stvg_dino_residual_advantage64_v1 import scalar

def projected(base,delta):
 if np.array_equal(delta,np.zeros(4)):return np.asarray(base).copy()
 a=np.asarray(base);return np.column_stack((a[:,:2]+delta[:2],np.exp(np.log(a[:,2:])+delta[2:])))

def metric(b,row,truth,span,iv,ds,planned):
 ids=np.asarray(row['frame_ids']);fids=np.array(sorted(truth));W,H=row['input']['width'],row['input']['height'];a=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*np.array([W,H,W,H])
 if ds=='hc2':a=np.maximum(a,0)
 px=np.column_stack([np.interp(fids,ids,a[:,k])for k in range(4)]);gg=np.array([truth[int(f)]for f in fids]);ins=np.maximum(0,np.minimum(px[:,2:],gg[:,2:])-np.maximum(px[:,:2],gg[:,:2])).prod(1);un=np.maximum(0,px[:,2:]-px[:,:2]).prod(1)+np.maximum(0,gg[:,2:]-gg[:,:2]).prod(1)-ins;v=np.divide(ins,un,out=np.zeros_like(ins),where=un>0);v[(fids<ids[0])|(fids>ids[-1])]=0;lo,hi=iv;g,h=span;over=max(0,min(hi,h)-max(lo,g));den=max(hi,h)-min(lo,g);inside=(fids>=max(lo,g))&(fids<min(hi,h));known=[i for i,f in enumerate(ids)if int(f)in truth];vals={i:scalar(b[i],truth[int(ids[i])],row,ds)for i in known};mean=lambda v:float(np.mean(v))if len(v)else None
 return dict(v=float(v[inside].sum()/max(den,1)),t=over/(hi-lo+h-g-over),s=float(v.mean()),observed_actual_s=mean([v for i,v in vals.items()if i in planned]),unobserved_actual_s=mean([v for i,v in vals.items()if i not in planned]),benchmark_interpolated_s=mean(v[~np.isin(fids,ids)]))

def main():
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 torch.set_num_threads(4);start=time.perf_counter();p=verify(True);bar=read(D/'GLOBAL_OUTPUT_BARRIER.json');assert bar['count']==32 and bar['not_GT_free'];res=read(P/'RESULTS_PUBLIC.json');checks=collections.Counter();errors=collections.Counter();pub=[]
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 def near(x,y,name,tol=1e-10):
  e=float(np.max(abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors[name],e);assert e<tol,(name,e)
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];x=load(D/'inputs'/(a+'.pt'));z=load(D/'predictions'/(a+'.pt'));ss=load(ROOT/f'artifacts/stvg_dino_endpoint_confirm32_v1/source/{a}.pt');row=plans[ds][q];tr=gt[ds][0][q];span=gt[ds][1][q];rr=next(v for v in res['rows']if v['alias']==a);S=x['Source'];E=x['E'];iv=x['interval'];allprobes=ss['expert']['observations'];pools=[np.asarray(allprobes['original',po]['probe']['boxes'],float).reshape(-1,4)for po in x['candidate_positions']]
  assert len(x['candidate_oracle']['landscape'])==int(np.prod([len(v)+1 for v in pools]));all_vs=[]
  for candidate in x['candidate_oracle']['landscape']:
   diffs=[]
   for po,pool,k in zip(x['candidate_positions'],pools,candidate['assignment']):
    if k>=0:
     b=pool[k];diffs.append(np.r_[b[:2]-S[po,:2],np.log(b[2:])-np.log(S[po,2:])])
   delta=np.mean(diffs,axis=0)if diffs else np.zeros(4);near(delta,candidate['delta'],'combination_delta');b=projected(S,delta);v=metric(b,row,tr,span,iv,ds,x['planned_positions'])['v'];near(v,candidate['v'],'oracle_metric');all_vs.append(v);checks['exhaustive_DINO_assignments']+=1
  best=max(all_vs+[rr['values']['E_MBR']['v']]);near(best,rr['values']['Candidate_combo_oracle']['v'],'finite_oracle_maximum');assert best>=rr['values']['E_MBR']['v']-1e-12
  for po,t in zip(x['positions'],x['targets']):
   fid=x['frame_ids'][po];g=np.array(tr[fid])/np.array([row['input']['width'],row['input']['height']]*2);target=np.r_[(g[:2]+g[2:])/2,g[2:]-g[:2]];near(target,t,'GT_label');assert po in x['planned_positions'];checks['true_exact_anchors']+=1
  for k in ['known4','matched']:
   near(projected(S,x['GT_estimates'][k]['delta'])if(k=='known4'and x['positions'])or(k=='matched'and x['matched_positions'])else E,z['outputs']['GT_E_'+k],'GT_E_mapping')
  if x['GT_subset_oracle']is not None:
   vv=[]
   for ca in x['GT_subset_oracle']['landscape']:
    active=[np.r_[t[:2]-S[po,:2],np.log(t[2:])-np.log(S[po,2:])]for po,t,k in zip(x['positions'],x['targets'],ca['assignment'])if k>=0];delta=np.mean(active,axis=0)if active else np.zeros(4);near(delta,ca['delta'],'GT_subset_delta');v=metric(projected(S,delta),row,tr,span,iv,ds,x['planned_positions'])['v'];near(v,ca['v'],'GT_subset_metric');vv.append(v);checks['GT_subset_assignments']+=1
   near(max(vv),rr['values']['GT_subset_oracle_known4']['v'],'GT_subset_maximum')
  for arm,f in z['fits'].items():
   ps=x['positions']if arm=='GT_state_known4'else x['matched_positions'];targets=x['targets']if arm=='GT_state_known4'else x['matched_GT_targets']if arm=='GT_state_matched'else x['matched_DINO_targets'];m=np.zeros(256);v=np.zeros(256);state=np.zeros(256);steps=0
   assert f['GT_used']==(arm!='DINO_state_matched')
   for j,t in enumerate(f['rounds']):
    near(t['before'],state,'state_path');near(t['loss_before'],loss_numpy(f['path'][j]['boxes'],ps,targets),'fit_loss');near(t['loss_after'],loss_numpy(t['post_boxes'],ps,targets),'fit_loss');native=np.asarray(t['post_native'],float);ret=np.concatenate([E[:,:2]+native[:,:2]-S[:,:2],E[:,2:]*(native[:,2:]/S[:,2:])],1);near(ret,t['post_boxes'],'E_retaining',2e-12);g=np.asarray(t['gradient']).astype(np.float32).astype(float)
    if t['updated']:
     steps+=1;m=.9*m+.1*g;v=.999*v+.001*g*g;new=(state-p['Adam'][ds]*(m/(1-.9**steps))/(np.sqrt(v/(1-.999**steps))+1e-8)).astype(np.float32).astype(float);near(new,t['after'],'Adam',1e-6);state=t['after'].copy()
    else:assert np.array_equal(t['before'],t['after']);state=t['after'].copy()
    assert np.array_equal(t['post_boxes'],f['path'][j+1]['boxes']);checks['scheduled_state_steps']+=1
   assert np.array_equal(f['final'],z['outputs'][arm]);checks['state_readouts']+=1
  if z['gradient_probe']is not None:assert z['gradient_probe']['absolute_error']<z['gradient_probe']['tolerance'];checks['fixed_GPU_FD_receipts']+=1
  assert z['zero_Source_E_exact']and z['compute']['PosDecoder_calls']==2*z['compute']['decoder_trajectory_calls'];checks['frozen_model_prefix_receipts']+=1
  for arm,b in z['outputs'].items():
   actual=metric(b,row,tr,span,iv,ds,x['planned_positions'])
   for key,val in actual.items():
    other=rr['values'][arm][key]
    if val is None:assert other is None
    else:near(val,other,'independent_metrics')
    checks['independent_metrics']+=1
  pub.append(dict(alias=a,dataset=ds,oracle_gain_pp=100*(rr['values']['Candidate_combo_oracle']['v']-rr['values']['E_MBR']['v']),GT_known_anchors=len(x['positions']),matched_anchors=len(x['matched_positions']),GT_state_steps=rr['fits']['GT_state_known4']['updated_steps']))
 for ds,g in res['groups'].items():
  rr=[r for r in res['rows']if r['dataset']==ds];assert len(rr)==16
  for pair,endpoints in g['paired'].items():
   a,b=pair.split('_minus_')
   for ep,z in endpoints.items():
    vals=[100*(r['values'][a][ep]-r['values'][b][ep])if r['values'][a][ep]is not None and r['values'][b][ep]is not None else None for r in rr];m,ci,n=independent_paired(vals);assert z['parents']==n
    if n:near(z['mean'],m,'parent_mean');near(z['CI95'],ci,'parent_bootstrap')
    checks['paired_parent_bootstraps']+=1
 verify(True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),max_errors=dict(errors),CPU_seconds=time.perf_counter()-start,GT_interventions_explicit=True,not_full_independent_GPU_replay=True,protected_baseline_formal_registry_hashes_unchanged=True));put(P/'CAPACITY_AUDIT_ROWS_PUBLIC.json',dict(status='pass',rows=pub));print('independent oracle/Adam/metric audit pass',dict(checks),dict(errors),flush=True)
if __name__=='__main__':main()

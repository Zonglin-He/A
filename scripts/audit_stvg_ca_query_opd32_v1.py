"""Independent NumPy Q/K, field/ROI likelihood, actual policy and metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_ca_query_opd32_v1 import D,P,FEAT,ANCHOR,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def audit_put(path,value):
 if path.exists():assert read(path)==value,(path,"existing audit receipt differs")
 else:put(path,value)

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_outputs_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();cosine_checks=[];maximum=dict(full_attention=0.,content_field=0.,cosine_absolute=0.,cosine_global_relative=0.,likelihood=0.,probability=0.,state=0.)
 for f in bar['files']+read(D/'GLOBAL_FEATURE_BARRIER.json')['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 def soft(x):y=np.exp(x-np.max(x,axis=-1,keepdims=True));return y/y.sum(axis=-1,keepdims=True)
 def independent_feedback(prob,valid,b,eps):
  n,k=b.shape[:2];h,w=valid.shape[-2:];areas=[]
  for t in range(n):
   vh=valid[t].any(1).sum();vw=valid[t].any(0).sum();xylo=(b[t,:,:2]-b[t,:,2:]/2)*[vw,vh];xyhi=(b[t,:,:2]+b[t,:,2:]/2)*[vw,vh];dx=np.maximum(np.minimum(xyhi[:,0,None],np.arange(w)+1)-np.maximum(xylo[:,0,None],np.arange(w)),0).clip(0,1);dy=np.maximum(np.minimum(xyhi[:,1,None],np.arange(h)+1)-np.maximum(xylo[:,1,None],np.arange(h)),0).clip(0,1);areas.append(dy[:,:,None]*dx[:,None,:]*valid[t])
  area=np.stack(areas);mass=area.sum((-1,-2));ok=mass>1e-12;safe=ok.copy();safe[~safe.any(1)]=True;lp=np.where(safe,-.5*np.square(eps).sum(-1),-np.inf);prior=soft(lp)
  if not len(prob):return prior,prior,np.ones((n,k)),ok,np.zeros(n,bool)
  per=np.einsum('anhw,nkhw->ank',prob,area)/np.maximum(mass/valid.sum((-1,-2))[:,None],1e-300)[None];L=per.mean(0);pos=ok&(L>0);psafe=pos.copy();psafe[~psafe.any(1)]=safe[~psafe.any(1)];logL=np.zeros_like(L);logL[pos]=np.log(L[pos]);pt=soft(np.where(psafe,lp+logL,-np.inf));hi=np.where(ok,L,-np.inf).max(1);lo=np.where(ok,L,np.inf).min(1);info=ok.any(1)&pos.any(1)&((hi-lo)>1e-12);pt[~info]=prior[~info];return pt,prior,L,ok,info
 for row in p['records']:
  a=row['alias'];o=load(D/'predictions'/(a+'.pt'));f=load(D/'features'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));old=load(ANCHOR/'predictions'/(a+'.pt'));E=f['E'].numpy();valid=f['valid'].numpy();n,h,w=valid.shape;V=h*w;conditions=f['conditions'];keys=[]
  for name,c in conditions.items():
   assert torch.equal(c['boxes'],old['Source']if name=='native'else old['raw_calls']['persistent_'+name]['boxes']);checks['hook_output_exact_old_decoder']+=1
   for view,rec in enumerate(c['records']):
    H=rec['heads'];dh=rec['head_dim'];q=rec['q'].numpy().astype(np.float64)[0].reshape(-1,H,dh);k=rec['k'].numpy().astype(np.float64).reshape(-1,len(q),H,dh);att=soft(np.einsum('nhd,snhd->nhs',q,k)/np.sqrt(dh)).mean(1)[:,None];err=float(abs(att-rec['attention'].numpy()).max());maximum['full_attention']=max(maximum['full_attention'],err);assert err<=1e-5;checks['native_all_key_attention_reconstruction']+=1
    if name!='native':assert torch.equal(rec['k'],conditions['native']['records'][view]['k']);checks['conditioned_keys_identical']+=1
  for t in range(n):
   rec=conditions['native']['records'][t%2];H=rec['heads'];dh=rec['head_dim'];keys.append(rec['k'].numpy().astype(np.float64)[:V,t//2].reshape(V,H,dh)[...,:dh//2])
  K=np.stack(keys);pv=valid.reshape(n,V)
  for mode in ['DINO','E']:
   qs=[]
   for t in f['positions']:qs.append(conditions[mode]['records'][t%2]['q'].numpy().astype(np.float64)[0,t//2].reshape(H,dh)[...,:dh//2])
   if qs:logits=np.einsum('ahd,nvhd->anhv',np.stack(qs),K)/np.sqrt(dh);prob=soft(np.where(pv[None,:,None],logits,-np.inf)).mean(2).reshape(len(qs),n,h,w)
   else:prob=np.empty((0,n,h,w))
   err=float(abs(prob-f['fields'][mode].numpy()).max())if len(qs)else 0.;maximum['content_field']=max(maximum['content_field'],err);assert err<1e-12;assert torch.equal(o['fields'][mode],f['fields'][mode]);checks['independent_content_transfer_fields']+=1
  if len(f['positions']):
   maps=x['maps'].numpy().astype(np.float32);ff=maps.reshape(n,256,V).transpose(0,2,1);ff/=np.maximum(np.linalg.norm(ff,axis=-1,keepdims=True),1e-12);out=[]
   for t,b in zip(f['positions'],f['targets']):
    # The inherited original cosine map deliberately uses its original padded-grid box geometry.
    b=np.asarray(b,dtype=np.float32).astype(np.float64);lo=(b[:2]-b[2:]/2)*[w,h];hi=(b[:2]+b[2:]/2)*[w,h];dx=np.maximum(np.minimum(hi[0],np.arange(w)+1)-np.maximum(lo[0],np.arange(w)),0).clip(0,1);dy=np.maximum(np.minimum(hi[1],np.arange(h)+1)-np.maximum(lo[1],np.arange(h)),0).clip(0,1);anchor=(dy[:,None]*dx[None])*valid[t];logits=(ff@ff[t].T)/.07;logits[...,~pv[t]]=-np.inf;m=(soft(logits)@anchor.reshape(V).astype(np.float32)).reshape(n,h,w)*valid;total=m.astype(np.float64).sum((-1,-2),keepdims=True);uniform=valid/valid.sum((-1,-2),keepdims=True);out.append(np.where(total>0,m/np.maximum(total,1e-300),uniform))
   cos=np.stack(out);reference=o['fields']['cosine'].numpy();err=float(abs(cos-reference).max());scale=float(abs(reference).max());maximum['cosine_absolute']=max(maximum['cosine_absolute'],err);maximum['cosine_global_relative']=max(maximum['cosine_global_relative'],err/max(scale,1e-300));threshold=32*np.finfo(np.float32).eps*scale+1e-12;cosine_checks.append(dict(alias=a,max_error=err,global_reference_max=scale,threshold=threshold,error_threshold_ratio=err/threshold,failed_original_abs3e_7=bool(err>=3e-7)));assert err<=threshold
   from vg_tta.stvg_soft_correspondence_v1 import propagate
   from vg_tta.stvg_ca_query_opd32_v1 import normalized_cosine
   exact=normalized_cosine(propagate(x['maps'],f['valid'],f['positions'],torch.as_tensor(f['targets']).double()),f['valid']);assert torch.equal(exact,o['fields']['cosine']);checks['original_cosine_same_anchor_normalization']+=1;checks['exact_saved_cosine_execution_recipe']+=1
  for mode,fit in o['fits'].items():
   fld=o['fields']['E'if mode=='Equery_live'else'cosine'if mode=='Cosine_like_live'else'DINO'].numpy();vv=valid.copy()
   if mode=='Dino_shuffle':fld=fld[:,o['permutation']];vv=vv[o['permutation']]
   eta=np.zeros((n,2));allowed=np.ones(n,bool);allowed[o['planned_positions']]=False
   for t in fit['rounds']:
    center=np.zeros_like(eta)if mode=='Dino_fixed'else eta;eps=o['noise'][t['step']].numpy();d=center[:,None]+eps;b=np.repeat(E[:,None],len(eps[0]),axis=1);b[:,:,:2]+=E[:,None,2:]*.25*d;assert abs(b-t['candidates'].numpy()).max()<1e-12
    pt,pr,L,ok,info=independent_feedback(fld,vv,b,eps);maximum['likelihood']=max(maximum['likelihood'],float(abs(L-t['feedback']['likelihood'].numpy()).max()));assert abs(L-t['feedback']['likelihood'].numpy()).max()<1e-10;assert abs(pt-t['feedback']['prob'].numpy()).max()<1e-12;assert abs(pr-t['feedback']['prior'].numpy()).max()<1e-12 and np.array_equal(ok,t['feedback']['valid']);safe=ok.copy();safe[~safe.any(1)]=True;sp=soft(np.where(safe,-.5*np.square(d-eta[:,None]).sum(-1),-np.inf));g=((sp-pt)[...,None]*d).sum(1);g[~(allowed&info)]=0;new=eta-.5*g;maximum['state']=max(maximum['state'],float(abs(new-t['after'].numpy()).max()));assert abs(new-t['after'].numpy()).max()<1e-12;eta=new;out=E.copy();out[:,:2]+=.25*E[:,2:]*eta;out[o['planned_positions']]=E[o['planned_positions']];assert abs(out-fit['path'][t['step']+1]['boxes'].numpy()).max()<1e-12;checks['independent_likelihood_KL_actual_rounds']+=1
   assert np.array_equal(fit['final'].numpy(),o['outputs'][mode]);assert np.array_equal(fit['final'].numpy()[:,2:],E[:,2:])and np.array_equal(fit['final'].numpy()[o['planned_positions']],E[o['planned_positions']]);checks['fixed_size_planned_final']+=1
  assert torch.equal(o['fits']['Dino_fixed']['path'][1]['boxes'],o['fits']['Dino_live']['path'][1]['boxes']);checks['same_first_step']+=1
  for mode,field in [('DINO',o['fields']['DINO'].numpy()),('E',o['fields']['E'].numpy())]:
   dd=o['direct'][mode];eps=np.concatenate([v.numpy()for v in o['noise']],1);b=dd['candidates'].numpy();pt,_,_,_,info=independent_feedback(field,valid,b,eps);best=pt.argmax(1);out=E.copy();use=info;use[o['planned_positions']]=False;out[use]=b[np.arange(n)[use],best[use]];assert abs(pt-dd['feedback']['prob'].numpy()).max()<1e-12 and abs(out-dd['boxes'].numpy()).max()<1e-12;checks['same_budget_direct']+=1
  if not f['positions']:
   for arm,b in o['outputs'].items():assert np.array_equal(b,E);checks['no_reference_noop']+=1
  independent_metrics(o,plans[row['dataset']][row['query_ordinal']],gt[row['dataset']][0][row['query_ordinal']],gt[row['dataset']][1][row['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   a,b=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])==want;checks['independent_parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();audit_put(P/'ROOT_AUDIT_RECOVERY.json',dict(status='audit_only_recovery_pass',original_audit_sha256=sha(D/'audit_recovery001/ORIGINAL_AUDIT.py'),original_failure=read(D/'audit_recovery001/PRESERVED_FAILURE.json'),rule='32*eps_FP32*global_reference_max+1e-12',unchanged_model_candidates_predictions_metrics_decision=True,independent_NumPy_and_exact_execution_recipe_separate_checks=True,not_attributed_solely_to_ROI_cast=True,per_alias_cosine_checks=cosine_checks));audit_put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),maximum_errors=maximum,no_GPU_model_rerun=True));audit_put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maximum,flush=True)
if __name__=='__main__':main()

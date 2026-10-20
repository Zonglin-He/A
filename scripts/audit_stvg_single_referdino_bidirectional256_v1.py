"""Independent initialization/OPD audit, then saved-score aggregation audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time,math,itertools,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_bidirectional256_common_v1 import ROOT,D,P,OLD,ROWS,read,put,sha,load,verify
from scripts.audit_stvg_spatial_strong_bidirectional256_v1 import array,softmax,giou
def pregt():
 p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['parents']==256;beg=time.perf_counter();errs={k:0. for k in ['initialization','chart','proposal','actions','cost','weights','gradient','state_increment']};fits=steps=0;supports=[]
 for rec in p['records']:
  ds,a=rec['dataset'],rec['alias'];z=load(D/f'predictions/{ds}/{a}.pt');tr=load(D/f'traces/{ds}/{a}.pt');inp=load(D/f'inputs/{ds}/{a}.pt');f=tr['fit'];cfg=p['configs'][ds];target=array(f['target']);src=array(inp['Source']).astype(float);pos=f['positions'];assert not any(k in inp for k in ['expert','observations','original_input'])
  assert sha(D/f'inputs/{ds}/{a}.pt')==tr['input_sha256'];assert hashlib.sha256(target.astype(np.float64).tobytes()).hexdigest()==z['shared_ReferDINO_evidence_hash'];assert np.array_equal(target,array(inp['teacher_native']['boxes']).astype(float))
  encoded=np.concatenate((src[:,:2],np.log(src[:,2:])),axis=-1);ref=np.concatenate((target[:,:2],np.log(target[:,2:])),axis=-1);d=ref-encoded[pos];scale=np.asarray(cfg['E_calibration']['scale']);cut=cfg['E_calibration']['truncation'];proposals=[]
  for assign in itertools.product([-1,0],repeat=8):
   active=np.array(assign)==0;delta=d[active].mean(0)if active.any()else np.zeros(4);dist=np.square((d-delta)/scale).sum(-1);selected=tuple(0 if v<cut else-1 for v in dist);value=float(np.minimum(dist,cut).sum());proposals.append((value,selected,assign,delta))
  proposals.sort(key=lambda x:(x[0],x[1],x[2]));_,selected,_,delta=proposals[0];out=src.copy()if not delta.any()else np.concatenate((src[:,:2]+delta[:2],np.exp(np.log(src[:,2:])+delta[2:])),axis=-1)
  errs['initialization']=max(errs['initialization'],float(abs(out-z['E_Ref']).max()));assert errs['initialization']<1e-10;assert list(selected)==z['initialization']['selected_reference_indices'];supports.append(z['initialization']['support_frames'])
  assert np.array_equal(array(f['path'][0]['boxes']),z['E_Ref'])and not array(f['path'][0]['state']).any();assert f['actual_updates']==cfg['steps']and len(f['path'])==cfg['steps']+1;assert np.array_equal(array(f['final']),z['Live']);fits+=1
  for k,rr in enumerate(f['rounds']):
   steps+=1;r=rr['rollout'];mu=array(rr['current_policy_mean']);Dd=array(r['D']);ac=array(r['actions']);weights=array(r['weights']);eps=array(r['eps']);proposal=array(r['proposal_mean']);boxes=array(f['path'][k]['boxes'])[pos];chart=np.concatenate((boxes[:,:2],np.log(boxes[:,2:])),axis=-1);sig=np.repeat(np.array(cfg['scales']),16);std=Dd[:,None,:]*np.array(cfg['scales'])[None,:,None]
   assert np.array_equal(eps[:,:8],-eps[:,8:16])and np.array_equal(eps[:,16:24],-eps[:,24:])
   b=np.concatenate((ac[...,:2],np.exp(ac[...,2:])),axis=-1);cost=5*abs(b-target[:,None]).sum(-1)+2*(1-giou(b,target[:,None]));q=softmax(-cost/cfg['tau']);diff=(ac[:,:,None,:]-mu[:,None,None,:])/std[:,None,:,:];comp=-.5*(diff*diff).sum(-1)-np.log(std).sum(-1)[:,None,:]-2*math.log(2*math.pi);resp=softmax(comp);grad=(weights[:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-ac[:,:,None,:])/std[:,None,:,:]**2).sum((1,2))/len(mu)
   checks=dict(chart=float(abs(chart-mu).max()),proposal=float(abs(proposal-mu).max()),actions=float(abs(ac-(proposal[:,None]+Dd[:,None]*sig[None,:,None]*eps)).max()),cost=float(abs(cost-array(r['cost'])).max()),weights=float(abs(q-weights).max()),gradient=float(abs(grad-array(rr['policy_mean_gradient'])).max()),state_increment=float(abs(array(f['path'][k+1]['state'])-array(f['path'][k]['state'])-array(rr['state_step'])).max()))
   assert max(checks.values())<2e-8,(a,k,checks)
   for key,v in checks.items():errs[key]=max(errs[key],v)
  old=load(OLD/f'predictions/{ds}/{a}.pt');assert all(np.array_equal(old['posterior'][k],z['posterior'][k])and old['intervals'][k]==z['intervals'][k]for k in ['Native','Strong']);assert np.array_equal(src,old['Source'])
  if fits%64==0:print('INDEPENDENT_SINGLE',fits,steps,flush=True)
 assert fits==256 and steps==8960
 put(P/'INDEPENDENT_SPATIAL_AUDIT_PUBLIC.json',dict(status='pass',fits=fits,updates=steps,independent_eight_singleton_initializations=256,zero_state_ERef_checks=256,shared_evidence_hash_checks=256,Source_and_time_unchanged=True,max_errors=errs,support_frames_min=int(min(supports)),support_frames_mean=float(np.mean(supports)),same_expert_for_initialization_and_supervision=True,expert_count=1,GT_read=False,seconds=time.perf_counter()-beg))
 print('PRE_GT_AUDIT_PASS',fits,steps,flush=True)
def scored():
 r=read(P/'RESULTS_PUBLIC.json');errors=[];checks=0
 for ds,group in r['directions'].items():
  rows=group['rows'];assert len(rows)==128
  for row in rows:
   x=read(D/f"private_metric_receipts/{ds}/{row['alias']}.json");g,h=x['GT_span'];obs=set(x['observed_GT_ids'])
   for arm,v in row['values'].items():
    d=x['per_arm'][arm];a,b=d['interval'];ids=np.array(d['dense_GT_ids']);io=np.array(d['dense_GT_IoU']);mask=(ids>=max(a,g))&(ids<min(b,h));overlap=max(0,min(b,h)-max(a,g));union=b-a+h-g-overlap
    expected=dict(v=float(io[mask].sum()/max(max(b,h)-min(a,g),1)),t=overlap/union,s=float(io.mean()),fixed_actual_sIoU=float(np.mean(list(d['actual_IoU'].values()))),unobserved_actual_sIoU=float(np.mean([q for f,q in d['actual_IoU'].items()if int(f)not in obs])))
    if obs:expected['observed_actual_sIoU']=float(np.mean([q for f,q in d['actual_IoU'].items()if int(f)in obs]))
    for k,q in expected.items():errors.append(abs(q-v[k]));checks+=1
  for arm,values in group['means'].items():
   for k,v in values.items():q=[row['values'][arm][k]for row in rows if row['values'][arm][k]is not None];errors.append(abs(sum(q)/len(q)-v));checks+=1
  for comparison,m in group['paired'].items():
   if comparison in ['New_ERef_minus_old_E','New_Full_minus_old_Full40','New_Full_minus_old_Full_matched_steps']:
    aa='Without_spatial_adaptation'if comparison=='New_ERef_minus_old_E'else'Full';bb={'New_ERef_minus_old_E':'old_two_expert_E_Strong','New_Full_minus_old_Full40':'old_two_expert_Full40','New_Full_minus_old_Full_matched_steps':'old_two_expert_Full_matched_steps'}[comparison];get=lambda row:(row['values'][aa],row['historical'][bb])
   else:
    aa,bb=comparison.split('_minus_');get=lambda row:(row['values'][aa],row['values'][bb])
   rng=np.random.default_rng(20261020);idx=rng.integers(0,128,(10000,128));x=np.array([100*(get(row)[0]['v']-get(row)[1]['v'])for row in rows]);errors.extend(abs(np.quantile(x[idx].mean(1),[.025,.975])-m['v']['CI95']).tolist());checks+=1
   for k,s in m.items():
    if k=='tail':continue
    errors.append(abs(np.mean([100*(get(row)[0][k]-get(row)[1][k])for row in rows])-s['mean']));checks+=1
   t=m['tail'];assert [int((x>1e-10).sum()),int((x< -1e-10).sum()),int((abs(x)<=1e-10).sum()),int((x< -5).sum())]==[t[k]for k in ['positive','negative','unchanged','below_minus5']]
 assert max(errors)<1e-10,max(errors)
 put(P/'ROOT_AUDIT_PUBLIC.json',dict(status='pass',independent_checks=checks,max_abs=max(errors),all256_retained=True,saved_receipt_scoring_and_parent_bootstrap=True,new_inference=False));print('ROOT_AUDIT_PASS',checks,max(errors))
if __name__=='__main__':
 if sys.argv[1]=='pre-gt':pregt()
 elif sys.argv[1]=='scored':scored()

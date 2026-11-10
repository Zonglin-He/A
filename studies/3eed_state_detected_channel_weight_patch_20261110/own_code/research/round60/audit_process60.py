"""Independent descriptive process verification after sealed task scoring.

No producer import, pseudo-label fitting, raw GT or additional science arms.
Recompute native per-view logits to verify the saved stability/separation claims.
"""
import sys,torch,time
from common60 import *
H=ROOT/'research/round60';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');rows=read(ROOT/p['manifest']);P=ROOT/p['prediction_root'];D=ROOT/p['detection_root'];F=ROOT/p['feature_root'];A=ROOT/p['association_root'];initial=load(ROOT/p['source_parameters']);W=torch.tensor(initial['4.weight']);b=torch.tensor(initial['4.bias']);process=read(H/'ALL_PROCESS_SCALARS_PUBLIC_SAFE.json');analysis=read(H/'ANALYSIS_SUMMARY.json');torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter();maxerror=0.;checked=0
def recompute(f,g,gate,activation):
 z=[];confidence=[];factor=torch.tensor(1-gate,dtype=torch.float32)
 for v in 'ABC':
  x=torch.tensor(f[v+'_X']);y=torch.nn.functional.linear(x*factor[None] if activation else x,W if activation else W*factor[None],b);q=torch.nn.functional.normalize(y,dim=1);l=q@torch.tensor(f[v+'_tokens']).T/.07;mask=torch.tensor(f[v+'_mask']).bool();zz=(torch.logsumexp(l[:,mask],1)-torch.logsumexp(l[:,~mask],1)).double().numpy()
  if v=='A':a=zz;rho=np.ones(256)
  else:
   tr=g[v+'_W'];mass=tr.sum(1);a=np.zeros(256);valid=mass>0;a[valid]=(tr@zz)[valid]/mass[valid];rho=g[v+'_affinity'].max(1)*valid
  z.append(a);confidence.append(rho)
 support=np.minimum(confidence[1],confidence[2]);den=float(support.sum())
 if den<=1e-12:return 0.,0.
 weights=support/den;z=np.array(z);centered=z-(z*weights).sum(1)[:,None];mean=centered.mean(0);d=float((((centered-mean)**2).mean(0)*weights).sum());s=float(((mean-(mean*weights).sum())**2*weights).sum());return d,s
for i,row in enumerate(rows):
 rel='samples_PRIVATE/'+row['id']+'.npz';f=load(F/('targets_PRIVATE/'+row['id']+'.npz'),[v+'_'+k for v in 'ABC' for k in ['X','tokens','mask']]);g=load(A/rel);patch=load(P/('patches_PRIVATE/'+row['id']+'.npz'));case=read(P/('cases_PRIVATE/'+row['id']+'.json'));base=recompute(f,g,np.zeros(288),False)
 for arm in ARMS[6:]:
  got=recompute(f,g,patch[arm],arm.endswith('_activation'));record=process[i*10+ARMS[6:].index(arm)];assert record['row']==f'DEV336_{i+1:05d}' and record['arm']==arm
  values=[record['disagreement_before'],record['separation_before'],record['disagreement_after'],record['separation_after']];expected=[*base,*got];err=max(abs(x-y) for x,y in zip(values,expected));maxerror=max(maxerror,err);assert err<1e-10
  assert np.count_nonzero(patch[arm])==record['edited_columns'];assert abs(np.linalg.norm(W.numpy()*patch[arm])/np.linalg.norm(W.numpy())-record['relative_weight_edit_norm'])<1e-12
  assert record['weight_edited']==case['arms'][arm]['actual_weight_changed'];checked+=1
for rec in analysis['process']:
 rr=[r for r in process if r['arm']==rec['arm']];ratios=np.array([r['disagreement_after']/max(r['disagreement_before'],1e-20)-1 for r in rr]);assert float(np.median(ratios)*100)==rec['median_matched_logodds_disagreement_change_percent'];assert int((ratios<0).sum())==rec['disagreement_decreased_n']
out=H/'independent_process_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R60_INDEPENDENT_NATIVE_STABILITY_AND_PROCESS_SCALARS_AUDIT','n':336,'process_rows':checked,'max_logodds_process_error':maxerror,'no_new_task_scoring_or_method_change':True,'program_sha256':sha(__file__),'prediction_seal_sha256':sha(P/'SEALED.json'),'analysis_sha256':sha(H/'ANALYSIS_SUMMARY.json'),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)

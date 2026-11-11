"""Independent NumPy head/objective/readout verification of every stored state."""
from common75 import *
from scipy.special import logsumexp,softmax
import time

def qhead(h,w):
 x=np.maximum(h@w['0.weight'].T+w['0.bias'],0);x=np.maximum(x@w['2.weight'].T+w['2.bias'],0);x=x@w['4.weight'].T+w['4.bias'];return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def lp(s,axis):return s-logsumexp(s,axis=axis,keepdims=True)
def prob(s,mask):return softmax(logsumexp(s[:,mask],axis=1)-logsumexp(s[:,~mask],axis=1))
def kd(a,b,axis):return np.sum(np.exp(a)*(a-b),axis=axis)
def main():
 p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/CAPTURE_CODE_LOCK.json'));pin(read(H/'experiment/FIT_CODE_LOCK.json'));pin(read(H/'experiment/PREDICT_CODE_LOCK.json'));src=ROOT/p['prefusion_source_root'];masked=ROOT/'crossplatform/round75/masked336_attempt001';held=ROOT/'crossplatform/round75/held336_attempt001';fit=ROOT/'crossplatform/round75/fit336_attempt001';pred=ROOT/'crossplatform/round75/predicted336_attempt001';seals={x:read(x/'SEALED.json') for x in [src,masked,held,fit,pred]};initial=load(ROOT/'crossplatform/round50/episodic336_capture_attempt001/SOURCE_MATCHING_MLP_PRIVATE.npz');maxloss=maxprob=maxref=0.;accepted=0;steps=[];start=time.perf_counter()
 for i,r in enumerate(rows()):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz'
  for root in [src,masked,held,pred]:assert sha(root/rel)==seals[root]['files'][rel]
  z=original(r);a=load(masked/rel);b=load(held/rel);f=load(fit/'reference_PRIVATE'/(sid+'.npz'));d={v:val for v,val in [('A',z['h']),('M1',a['M1_h']),('M2',a['M2_h']),('M3',b['M3_h'])]};u=z['tokens'].astype(float);mask=z['token_mask'];per=f['permutation'];assert np.array_equal(np.sort(per),np.arange(256)) and np.all(per!=np.arange(256));q0=qhead(d['A'],initial);S0=q0.astype(float)@u.T/.07;R0=lp(S0,1);C0=lp(S0,0);p0=prob(S0,mask);maxref=max(maxref,float(abs(f['S0']-S0).max()));assert maxref<3e-5
  for view,data in [('M1',a),('M2',a),('M3',b)]:
   mm=data[view+'_mask'];assert np.array_equal(mm,mask_for(r,z,view)) and mm.sum()==153 and not mm[np.asarray(z['query_points_sample_inds'],int)].any()
  out=load(pred/rel);pp=dict(zip(out['arm_names'],out['scores']));qs={v:qhead(h,initial) for v,h in d.items()};S={v:q.astype(float)@u.T/.07 for v,q in qs.items()};refprobs={'source':z['scores']/z['scores'].astype(float).sum(),'frozen_mask_probability':sum(prob(S[v],mask) for v in ['A','M1','M2'])/3,'frozen_mask_token':prob(sum(S[v] for v in ['A','M1','M2'])/3,mask),'held_mask_source':prob(S['M3'],mask)};meanq=sum(qs[v] for v in ['A','M1','M2'])/3;meanq=meanq/np.maximum(np.linalg.norm(meanq,axis=-1,keepdims=True),1e-12);refprobs['frozen_mask_representation']=prob(meanq.astype(float)@u.T/.07,mask)
  for kind in ADAPT:
   wf=fit/'weights_PRIVATE'/(sid+'_'+kind+'.npz');rf=fit/'cases_PRIVATE'/(sid+'_'+kind+'.json');assert sha(wf)==seals[fit]['files'][wf.relative_to(fit).as_posix()] and sha(rf)==seals[fit]['files'][rf.relative_to(fit).as_posix()];w=load(wf);rec=read(rf);accepted+=rec['accepted'];assert rec['reference_text_fixed'] and not rec['M3_used'] and not rec['GT_or_expert_used']
   def loss(ww):
    keep=np.mean(np.sum((qhead(d['A'],ww)-q0).astype(float)**2,axis=1))/2;vals=[]
    for v in ['M1','M2']:
     ss=qhead(d[v],ww).astype(float)@u.T/.07
     if kind=='output':pv=prob(ss,mask);vals.append(np.sum(p0*(np.log(np.maximum(p0,1e-300))-np.log(np.maximum(pv,1e-300)))))
     else:
      rr=R0[per] if kind=='shuffled' else R0;cc=C0[per] if kind=='shuffled' else C0;vals.append(kd(rr,lp(ss,1),1).mean()+kd(cc,lp(ss,0),0).mean())
    return float(np.mean(vals)+keep)
   old=loss(initial);new=loss(w);maxloss=max(maxloss,abs(old-rec['loss_before']),abs(new-rec['loss_after']));assert maxloss<1e-4
   if rec['accepted']:assert rec['loss_after']<=rec['loss_before']-max(1e-12,1e-6*abs(rec['loss_before'])) and any(np.any(w[k]!=initial[k]) for k in w)
   else:assert all(np.array_equal(w[k],initial[k]) for k in w)
   ss={v:qhead(h,w).astype(float)@u.T/.07 for v,h in d.items()};refprobs[kind+'_full']=prob(ss['A'],mask);refprobs[kind+'_ensemble']=sum(prob(ss[v],mask) for v in ['A','M1','M2'])/3;refprobs['held_mask_'+kind]=prob(ss['M3'],mask)
   steps.append({'row':f'R{i+1:05d}','platform':r['platform'],'method':kind,'accepted':rec['accepted'],'scale':rec['scale'],'loss_before':rec['loss_before'],'loss_after':rec['loss_after'],'auxiliary_before':rec['auxiliary_before'],'auxiliary_after':rec['auxiliary_after'],'keep_after':rec['keep_after'],'gradient_norm':rec['gradient_norm'],'parameter_step_L2':rec['parameter_step_L2'],'trials':len(rec['trials'])})
  for k,val in refprobs.items():maxprob=max(maxprob,float(abs(pp[k]-val).max()))
  assert np.array_equal(out['source_boxes'],z['boxes']) and np.array_equal(out['selected_original_boxes'],z['boxes'][out['queries']]);assert list(out['arm_names'])==ARMS;assert maxprob<2e-5
  if(i+1)%64==0:print('AUDIT75',i+1,{'maxprob':maxprob,'maxloss':maxloss},flush=True)
 out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R75_ALL336_NATIVE_HEAD_DENSE_OUTPUT_SHUFFLE_OBJECTIVES_FIXED_U_READOUTS_AND_MASK_IDENTITY','n':336,'head_states':1008,'accepted':accepted,'accepted_loss_increases':0,'max_independent_probability_error':maxprob,'max_independent_objective_error':maxloss,'max_independent_reference_matrix_error':maxref,'source_A256_unchanged':True,'M3_not_in_fitting':True,'task_GT_or_expert_read':False,'prediction_seal_sha256':sha(pred/'SEALED.json'),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);dump(H/'ALL_OPTIMIZER_STEP_ROWS.json',steps);print(rec)
if __name__=='__main__':main()

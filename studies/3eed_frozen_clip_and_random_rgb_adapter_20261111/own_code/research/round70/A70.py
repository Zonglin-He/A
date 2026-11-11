"""Frozen CLIP direct scoring and independent zero-output score-adapter probes."""
from common70 import *
from models70 import *
import time
from scipy.special import softmax
p=read(H/'experiment/A_PLAN_LOCK.json');pin(p);out=ROOT/p['A_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();torch.set_num_threads(4);start=time.perf_counter();steps=0
for i,r in enumerate(rows()):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';z=prefix(r);a=load(ROOT/p['prepared_root']/rel);c=load(ROOT/p['clip_root']/rel);base=np.log(np.maximum(z['scores'].astype(float),1e-300));scores={'source':z['scores']};
 for name,kw in [('A_subject_tight',{'joint':False}),('A_subject_joint',{}),('A_full_joint',{'kind':'full'}),('A_projectonly_joint',{'reliable':False}),('A_mismatch_joint',{'shuffle':True})]:scores[name]=softmax(base+evidence(a,c,**kw))
 torch.manual_seed(0);m=ScoreAdapter();initial=snapshot(m);saved={};opt=torch.optim.Adam(m.parameters(),lr=.005);x=torch.tensor(A_input(z,a,c));xt=torch.tensor(A_input(z,a,c,context=True));b=torch.tensor(base,dtype=torch.float64);teacher=torch.tensor(scores['A_subject_tight']);unknown=torch.tensor(a['reliability'][a['query_group']]==0);before=None;log=[]
 assert torch.count_nonzero(m(x))==0
 for step in range(1,17):
  opt.zero_grad(set_to_none=True);delta=m(x);lp=torch.log_softmax(b+delta.double(),dim=0);loss=torch.nn.functional.kl_div(lp,teacher,reduction='sum')+.1*(delta[unknown]**2).mean() if unknown.any() else torch.nn.functional.kl_div(lp,teacher,reduction='sum');before=float(loss.detach()) if before is None else before;assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();steps+=1
  if step in [3,16]:
   saved.update({'step'+str(step)+'_'+k:v for k,v in snapshot(m).items()})
   with torch.no_grad():scores['A_adapter'+str(step)]=torch.softmax(b+m(xt).double(),0).numpy();log.append({'step':step,'KL_training_before_step':float(loss.detach()),'gradient_norm':gn,'parameter_L2':float(sum(np.sum((v-initial[k])**2) for k,v in snapshot(m).items())**.5)})
 rec={'parameter_count':sum(v.numel() for v in m.parameters()),'zero_output_init_exact':True,'initial_loss':before,'updates':log,'CLIP_and_3EED_never_trainable':True,'context_crop_not_used_in_teacher_target':True};np.savez_compressed(out/rel,arm_names=np.array(list(scores)),scores=np.stack(list(scores.values())),**saved);dump(out/'cases_PRIVATE'/(sid+'.json'),rec)
 if(i+1)%32==0:print('A70',i+1,336,flush=True)
dump(out/'COUNTS.json',{'n':336,'adapter_steps':steps,'parameters':sum(v.numel() for v in m.parameters()),'seconds':time.perf_counter()-start,'external_RGB_semantic_teacher':'frozen_CLIP_not_teacher_free','adaptation_claim_requires_beating_joint_direct_and_mismatch':True});seal(out,'R70_A_CLIP_DIRECT_AND_RESIDUAL_ADAPTER_ALL_PREDICTIONS_BEFORE_LABELS',n=336)

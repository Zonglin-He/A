"""R77 source-label-controlled direction/radius/capacity interventions, not TTA claims."""
from pathlib import Path
import sys,time,copy
ROOT=Path(__file__).resolve().parents[2];H77=ROOT/'research/round77';OUT=ROOT/'crossplatform/round77/source128_attempt001'
sys.path.insert(0,str(ROOT/'research/round76'))
from common76 import *
from model76 import *
from target76 import objective,step,phash
sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,fingerprint
ARMS77=['source','bidirectional_frozen','aux1','aux3','aux_raw1','aux_raw10','ground_same_radius1','ground_same_radius10','ground_fit10','ground_match10','wrong_binding_aux3']

def vector(ps):return torch.cat([p.detach().flatten() for p in ps])
def setvector(ps,x):
 with torch.no_grad():
  off=0
  for p in ps:n=p.numel();p.copy_(x[off:off+n].reshape_as(p));off+=n
 assert off==len(x)
def gradient(L,ps):
 g=torch.autograd.grad(L,ps,allow_unused=True)
 return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(ps,g)])
def cosine(a,b):
 n=float(a.norm()*b.norm());return float(a.dot(b))/n if n>0 else None
def groundloss(p,y):return -(y*p.clamp_min(1e-300).log()).sum()
def enable(m,match=False):
 m.requires_grad_(False);nn=list(m.match.named_parameters()) if match else m.tta_parameters()
 for _,p in nn:p.requires_grad_(True)
 return [p for _,p in nn]
def adam_direction(ps,L):
 old=vector(ps);o=torch.optim.Adam(ps,lr=.0001,weight_decay=0);o.zero_grad(set_to_none=True);L.backward();gn=float(torch.nn.utils.clip_grad_norm_(ps,1));o.step();v=vector(ps)-old;setvector(ps,old);o.zero_grad(set_to_none=True);return v,gn

def main():
 plan=read(H77/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H77/'experiment/CODE_LOCK.json'));assert not OUT.exists();OUT.mkdir()
 for sub in ['samples_PRIVATE','states_PRIVATE','gradients_PRIVATE','records_PRIVATE']:(OUT/sub).mkdir()
 base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);basefp=fingerprint(base)
 m=Fusion(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/plan['source_prepared_module']);restore(m,S);initialhash=phash(m);ps=enable(m);names=[n for n,_ in m.tta_parameters()];theta=vector(ps).clone();assert len(theta)==102498
 rr=source_rows('VAL128');assert len(rr)==128;oldval=load(ROOT/'crossplatform/round76/trained_source_attempt002/val_PRIVATE/bidirectional_e06.npz')
 calls={'main':0,'auxiliary':0};native_main=m.main;native_aux=m.auxiliary
 def countmain(*a,**kw):calls['main']+=1;return native_main(*a,**kw)
 def countaux(*a,**kw):calls['auxiliary']+=1;return native_aux(*a,**kw)
 m.main=countmain;m.auxiliary=countaux
 labelsroot=ROOT/'research/round25/experiment/source_labels_attempt002';labseal=read(labelsroot/'SEALED.json');start=time.perf_counter();allrecords=[]
 for idx,r in enumerate(rr):
  restore(m,S);ps=enable(m);z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);yp=labelsroot/'samples_PRIVATE'/(r['id']+'.npz');assert sha(yp)==labseal['files'][yp.relative_to(labelsroot).as_posix()];ious=load(yp)['ious'];eligible=bool(ious.max()>0);y=torch.softmax(torch.tensor(ious,device='cuda',dtype=torch.float64)/.1,0);pm=[];im=[]
  for rep in [76001,76002]:a,b=masks(r,d,rep);pm.append(a);im.append(b)
  with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');refp=refp.detach();refq=refq.detach();assert np.max(abs(refp.cpu().numpy()-oldval['scores'][idx]))<1e-10
  records={'label_eligible':eligible,'source_labels_are_controlled_diagnostic_only':True};scores={'source':z0['scores']/z0['scores'].sum(),'bidirectional_frozen':refp.cpu().numpy()};armstates={};endpoint={}
  def loss():
   p,_,_=m.main(base,z,d,'bidirectional');return groundloss(p,y) if eligible else p.sum()*0
  # Both raw gradients and the actual first-Adam direction matter: Adam changes metric.
  L=loss();gg=gradient(L,ps);total,aa,kk=objective(m,base,z,d,refp,refq,pm,im);ga=gradient(total,ps)
  delta,gn=adam_direction(ps,objective(m,base,z,d,refp,refq,pm,im)[0]);gd,_=adam_direction(ps,loss());radius=float(delta.norm());gdn=float(gd.norm());gd=gd*(radius/gdn) if gdn>0 else gd
  records.update(aux_gradient_L2=float(ga.norm()),ground_gradient_L2=float(gg.norm()),raw_gradient_cosine=cosine(ga,gg),ground_gradient_dot_aux_Adam_delta=float(gg.dot(delta)),ground_gradient_dot_ground_Adam_delta=float(gg.dot(gd)),ground_aux_Adam_cosine=cosine(gg,delta),first_Adam_radius=radius)
  module=[];off=0
  for name,p in m.tta_parameters():
   n=p.numel();a=ga[off:off+n];g=gg[off:off+n];v=delta[off:off+n];module.append({'parameter':name,'n':n,'aux_gradient_L2':float(a.norm()),'ground_gradient_L2':float(g.norm()),'gradient_cosine':cosine(a,g),'ground_gradient_dot_aux_Adam_delta':float(g.dot(v))});off+=n
  records['parameter_gradient_rows']=module
  np.savez_compressed(OUT/'gradients_PRIVATE'/(r['id']+'.npz'),aux_gradient=ga.cpu().numpy(),ground_gradient=gg.cpu().numpy(),aux_Adam_delta=delta.cpu().numpy(),matched_ground_Adam_delta=gd.cpu().numpy())
  def save(arm,wrong=False,match=False):
   with torch.no_grad():
    pp,q,_=m.main(base,z,d,'bidirectional');tot,aux,keep=objective(m,base,z,d,refp,refq,pm,im,wrong);gl=groundloss(pp,y) if eligible else pp.sum()*0;scores[arm]=pp.cpu().numpy();endpoint[arm]={'ground_loss':float(gl),'auxiliary':float(aux),'total_aux_keep':float(tot),'keep':float(keep),'query_representation_delta_L2':float((q-refq).norm()),'score_delta_L2':float((pp-refp).norm()),'parameter_step_L2':float((vector([p for _,p in m.tta_parameters()])-theta).norm()),'query_changed':int(pp.argmax())!=int(refp.argmax())}
   nn=list(m.match.named_parameters()) if match else m.tta_parameters();patch={n:p.detach().cpu().numpy().copy() for n,p in nn};np.savez_compressed(OUT/'states_PRIVATE'/(r['id']+'_'+arm+'.npz'),**patch);armstates[arm]={'scope':'match' if match else 'tta','sha256':sha(OUT/'states_PRIVATE'/(r['id']+'_'+arm+'.npz'))}
  with torch.no_grad():g0=float(groundloss(refp,y)) if eligible else 0;a0=float(objective(m,base,z,d,refp,refq,pm,im)[1])
  endpoint['bidirectional_frozen']={'ground_loss':g0,'auxiliary':a0,'total_aux_keep':a0,'keep':0.,'parameter_step_L2':0.,'score_delta_L2':0.,'query_representation_delta_L2':0.,'query_changed':False}
  for arm,direction,mult in [('aux_raw1',delta,1),('aux_raw10',delta,10),('ground_same_radius1',gd,1),('ground_same_radius10',gd,10)]:
   setvector(ps,theta+mult*direction);save(arm);setvector(ps,theta)
  # Exact native R76 transactional auxiliary update, with same masks and keep terms.
  for wrong in [False,True]:
   restore(m,S);ps=enable(m);opt=torch.optim.Adam(ps,lr=.0001,weight_decay=0);updates=[]
   for t in range(3):
    updates.append(step(m,base,z,d,opt,refp,refq,pm,im,wrong))
    if not wrong and t==0:save('aux1')
   arm='wrong_binding_aux3' if wrong else 'aux3';save(arm,wrong);records[arm+'_steps']=updates
  # Source-supervised interface capacity controls; never enter target deployment.
  for match in [False,True]:
   restore(m,S);pparams=enable(m,match);opt=torch.optim.Adam(pparams,lr=.001,weight_decay=0);steps=[]
   for t in range(10):
    old=vector(pparams);os=copy.deepcopy(opt.state_dict());opt.zero_grad(set_to_none=True);L=loss();before=float(L.detach());L.backward();torch.nn.utils.clip_grad_norm_(pparams,1);opt.step();proposal=vector(pparams);accepted=False;trials=[]
    with torch.no_grad():
     for j in range(9):
      scale=2.**(-j);setvector(pparams,old+scale*(proposal-old));now=float(loss());trials.append({'scale':scale,'ground_loss':now})
      if eligible and now<=before-max(1e-12,1e-6*abs(before)):accepted=True;break
     if not accepted:setvector(pparams,old);opt.load_state_dict(os)
    steps.append({'accepted':accepted,'ground_before':before,'ground_after':float(loss().detach()),'parameter_step_L2':float((vector(pparams)-old).norm()),'trials':trials})
   arm='ground_match10' if match else 'ground_fit10';save(arm,match=match);records[arm+'_steps']=steps
  records['endpoints']=endpoint;records['states']=armstates;records['row']=f'S{idx+1:05d}';records['source_candidate_oracle_IoU']=float(ious.max());records['source_label_file_sha256']=sha(yp);dump(OUT/'records_PRIVATE'/(r['id']+'.json'),records);allrecords.append({'id':r['id'],'row':records['row']})
  ss=np.stack([scores[a] for a in ARMS77]);qq=ss.argmax(1);np.savez_compressed(OUT/'samples_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS77),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);restore(m,S);assert phash(m)==initialhash
  if(idx+1)%8==0:print('DIAG77',idx+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 assert fingerprint(base)==basefp;dump(OUT/'ROWS_PRIVATE.json',allrecords);dump(OUT/'COUNTS.json',{'n':128,'arms':len(ARMS77),'TTA_parameters':len(theta),'seconds':time.perf_counter()-start,'main_suffix_calls':calls['main'],'auxiliary_calls':calls['auxiliary'],'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'source_GT_read':True,'target_GT_read':False,'original_model_unchanged':True,'source_prepared_module_unchanged':phash(m)==initialhash,'FINAL_access':False});seal(OUT,'R77_SOURCE_LABEL_CONTROLLED_DIAGNOSTIC_READOUTS_SEALED',n=128,arm_names=ARMS77);print(read(OUT/'COUNTS.json'))
if __name__=='__main__':main()

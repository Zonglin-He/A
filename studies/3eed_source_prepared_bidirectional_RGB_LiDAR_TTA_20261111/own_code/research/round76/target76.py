"""Source-selected frozen architectures + unlabeled episodic physical TTA."""
from common76 import *
from model76 import *
import time,copy
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint

def phash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def ohash(o):
 h=hashlib.sha256();s=o.state_dict()
 for n,v in sorted(s['state'].items()):
  h.update(str(n).encode())
  for k,t in sorted(v.items()):h.update(k.encode());h.update(t.detach().cpu().numpy().tobytes())
 h.update(str(s['param_groups']).encode());return h.hexdigest()
def objective(m,base,z,d,refp,refq,pmasks,imasks,wrong=False):
 pp,q,_=m.main(base,z,d,'bidirectional');auxs=[];parts=[]
 for pm,im in zip(pmasks,imasks):
  a,b=m.auxiliary(z,d,'bidirectional',pm,im,wrong=wrong);auxs.append(a);parts.append(b)
 aux=torch.stack(auxs).mean();keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;return aux.double()+keep,aux,keep

def step(m,base,z,d,opt,refp,refq,pm,im,wrong):
 old={n:v.detach().clone() for n,v in m.tta_parameters()};stateopt=copy.deepcopy(opt.state_dict());hbefore=phash(m);obefore=ohash(opt);opt.zero_grad(set_to_none=True);L,aux,keep=objective(m,base,z,d,refp,refq,pm,im,wrong);before=float(L.detach());ab=float(aux.detach());L.backward();params=[v for _,v in m.tta_parameters()];gn=float(torch.nn.utils.clip_grad_norm_(params,1));assert np.isfinite(before) and np.isfinite(gn);opt.step();prop={n:v.detach().clone() for n,v in m.tta_parameters()};trials=[];accepted=False;scale=0
 with torch.no_grad():
  for j in range(9):
   alpha=2.**(-j)
   for n,v in m.tta_parameters():v.copy_(old[n]+alpha*(prop[n]-old[n]))
   l,a,k=objective(m,base,z,d,refp,refq,pm,im,wrong);val=float(l);trials.append({'scale':alpha,'loss':val,'auxiliary':float(a),'keep':float(k)})
   if np.isfinite(val) and val<=before-max(1e-12,1e-6*abs(before)):accepted=True;scale=alpha;break
  if not accepted:
   for n,v in m.tta_parameters():v.copy_(old[n])
   opt.load_state_dict(stateopt)
  l,a,k=objective(m,base,z,d,refp,refq,pm,im,wrong);delta=float(torch.sqrt(sum((v-old[n]).square().sum() for n,v in m.tta_parameters())))
 assert (float(l)<=before-max(1e-12,1e-6*abs(before)) and delta>0) if accepted else (phash(m)==hbefore and ohash(opt)==obefore)
 opt.zero_grad(set_to_none=True);return {'accepted':accepted,'scale':scale,'loss_before':before,'loss_after':float(l),'aux_before':ab,'aux_after':float(a),'keep_after':float(k),'gradient_norm':gn,'parameter_step_L2':delta,'trials':trials,'before_sha256':hbefore,'after_sha256':phash(m),'optimizer_before_sha256':obefore,'optimizer_after_sha256':ohash(opt),'target_GT_or_expert_used':False,'RGB_depth_input_is_prefusion_image_only':True}

def main():
 p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/TARGET_CODE_LOCK.json'));sr=BASE/'trained_source_attempt002';ss=read(sr/'SEALED.json');selected=read(sr/'SELECTED_SOURCE_ONLY.json');base=load_source(read(ROOT/p['source_plan']));fp=fingerprint(base);net={};states={};counts={'new_grounding_suffix_forwards':0};rr=target_rows();out=BASE/'target336_attempt001';assert not out.exists();out.mkdir()
 for f in ['samples_PRIVATE','steps_PRIVATE','adapted_PRIVATE','features_PRIVATE']:(out/f).mkdir()
 for mode in MODES:
  sel=selected[mode];sf=ROOT/sel['state_path'];assert sha(sf)==sel['state_sha256']==ss['files'][sf.relative_to(sr).as_posix()];m=Fusion(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);states[mode]=load(sf);restore(m,states[mode]);net[mode]=m
 initial_hash={k:phash(v) for k,v in net.items()};start=time.perf_counter();status=[];other={pl:[r for r in rr if r['platform']==pl] for pl in ['drone','quad']};wrong_images={r['id']:other[r['platform']][(i+1)%len(other[r['platform']])] for pl in other for i,r in enumerate(other[pl])}
 for i,r in enumerate(rr):
  sid=r['id'];z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);probs={};features={};cases={};pm=[];im=[]
  for repeat in [76001,76002]:
   aa,bb=masks(r,d,repeat);pm.append(aa);im.append(bb)
  for mode in MODES:
   m=net[mode]
   with torch.no_grad():pp,qq,_=m.main(base,z,d,mode);counts['new_grounding_suffix_forwards']+=1
   probs[mode+'_frozen']=pp.cpu().numpy();features[mode+'_frozen_q']=qq.cpu().numpy()
  bi=net['bidirectional'];refp=torch.tensor(probs['bidirectional_frozen'],device='cuda');refq=torch.tensor(features['bidirectional_frozen_q'],device='cuda')
  with torch.no_grad():
   pp,_,_=bi.main(base,z,d,'bidirectional',reverse_off=True);probs['bidirectional_reverse_off']=pp.cpu().numpy();counts['new_grounding_suffix_forwards']+=1;wrong_g=load(casepath(wrong_images[sid]));wi=tensor_data(wrong_g,wrong_images[sid])['image'];pp,_,_=bi.main(base,z,d,'bidirectional',image=wi);probs['bidirectional_wrong_RGB']=pp.cpu().numpy();counts['new_grounding_suffix_forwards']+=1;views=[refp]
   for a,b in zip(pm,im):pp,_,_=bi.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);counts['new_grounding_suffix_forwards']+=1;views.append(pp)
   probs['bidirectional_frozen_budget3']=torch.stack(views).mean(0).cpu().numpy();aux_before=float(torch.stack([bi.auxiliary(z,d,'bidirectional',a,b)[0] for a,b in zip(pm,im)]).mean())
  # Two independently reset adaptation runs, correct and interaction-binding control.
  for wrong in [False,True]:
   restore(bi,states['bidirectional']);bi.requires_grad_(False)
   for _,v in bi.tta_parameters():v.requires_grad_(True)
   opt=torch.optim.Adam([v for _,v in bi.tta_parameters()],lr=.0001,weight_decay=0);records=[]
   for t in range(1,4):
    rec=step(bi,base,z,d,opt,refp,refq,pm,im,wrong);records.append(rec)
    if (not wrong and t in [1,3]) or (wrong and t==3):
     with torch.no_grad():pp,q,_=bi.main(base,z,d,'bidirectional');counts['new_grounding_suffix_forwards']+=1
     arm=('bidirectional_TTA3_wrong_correspondence' if wrong else 'bidirectional_TTA'+str(t));probs[arm]=pp.cpu().numpy();features[arm+'_q']=q.cpu().numpy();np.savez_compressed(out/'adapted_PRIVATE'/(sid+'_'+arm+'.npz'),**{n:v.detach().cpu().numpy() for n,v in bi.tta_parameters()})
   cases['wrong_interaction_binding' if wrong else 'correct']=records
   del opt
  restore(bi,states['bidirectional']);bi.requires_grad_(False);assert phash(bi)==initial_hash['bidirectional']
  source=z0['scores'].astype(float);source/=source.sum();probs['source']=source;hist=load(ROOT/'crossplatform/round66/predicted336_attempt001/samples_PRIVATE'/(sid+'.npz'),['scores','arm_names','source_boxes']);assert np.array_equal(z0['boxes'],hist['source_boxes']);hd=dict(zip(hist['arm_names'],hist['scores']));probs['frozen_ABC']=hd['frozen_ABC'];probs['R59']=hd['R59'];scores=np.stack([probs[k] for k in ARMS]);q=scores.argmax(1);np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][q]);np.savez_compressed(out/'features_PRIVATE'/(sid+'.npz'),**features);dump(out/'steps_PRIVATE'/(sid+'.json'),{'updates':cases,'unlabeled_auxiliary_before':aux_before,'visible_point_tokens':int(d['visible'].sum()),'source_selected_epochs':{k:selected[k]['epoch'] for k in selected},'no_target_GT_read':True});st={}
  for k in ARMS:
   n=1 if k=='bidirectional_TTA1' else 3;rec=cases['wrong_interaction_binding' if k.endswith('wrong_correspondence') else 'correct'][:n] if 'TTA' in k else [];active=any(x['accepted'] for x in rec);st[k]={'unknown':bool(not d['visible'].any()) and k not in ['source','frozen_ABC','R59'],'fallback':bool(rec and not active),'no_parameter_update':not active}
  status.append({'id':sid,'arm_statuses':st})
  if(i+1)%16==0:print('TARGET76',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 assert fingerprint(base)==fp and all(phash(net[k])==initial_hash[k] for k in net);dump(out/'READOUT_STATUS_PRIVATE.json',status);dump(out/'COUNTS.json',{'n':336,'arms':len(ARMS),'original3EED_unchanged':True,'source_selected_before_target':True,'target_GT_read':False,'TTA_parameters':sum(v.numel() for _,v in bi.tta_parameters()),'TTA_optimizer_proposals':336*6,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),**counts});seal(out,'R76_ALL_SOURCE_PREPARED_AND_TARGET_TTA_PREDICTIONS_SEALED_BEFORE_TARGET_GT',n=336,arm_names=ARMS);print(read(out/'COUNTS.json'))
if __name__=='__main__':main()

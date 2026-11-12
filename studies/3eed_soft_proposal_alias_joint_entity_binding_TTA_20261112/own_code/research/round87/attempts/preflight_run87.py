"""All component direct/TTA pairs, same prepared branch and fixed observations."""
from core87 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();plan=read(H87/'experiment/DESIGN_LOCK.json');pin(plan);start=time.perf_counter();base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);m=Fusion85(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/plan['init_state']);restore(m,S);rows=read(H87/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p and read(B87/'evidence336_attempt001/records_PRIVATE'/(r['id']+'.json'))['full']['active']) for p in ['drone','quad']]
O=B87/('engineering_attempt001' if args.engineering else 'prediction336_attempt001');assert not O.exists()
for d0 in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE']:(O/d0).mkdir(parents=True,exist_ok=False)
calls={'grounding':0,'objective':0,'accepted_updates':0,'trial_objectives':0};statuses=[];legacy={r['id']:r['arm_statuses'] for r in read(B86/'prediction366_attempt001/READOUT_STATUS_PRIVATE.json')}
def obj(*a):calls['objective']+=1;calls['grounding']+=3;return objective86(*a)
def mainread(*a,**k):calls['grounding']+=1;return m.main(*a,**k)
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d=item80(r);old=load(B86/'prediction366_attempt001/samples_PRIVATE'/(sid+'.npz'));od=dict(zip(old['arm_names'],old['scores']));ev=load(B87/'evidence336_attempt001/samples_PRIVATE'/(sid+'.npz'));meta=read(B87/'evidence336_attempt001/records_PRIVATE'/(sid+'.json'));reg=load(B86/'evidence366_attempt002/samples_PRIVATE'/(sid+'.npz'));region={'point':reg['region_point'],'image':reg['region_image']};ab=[mask86(r,d,region,k) for k in [86001,86002]];pm=[p for p,i in ab];im=[i for p,i in ab];ws=local_weights85(d,pm,im,region);restore(m,S);m.requires_grad_(False)
 with torch.no_grad():
  frozen=mainread(base,z,d,'bidirectional')[0];assert float(abs(frozen-torch.tensor(od['frozen_bi'],device='cuda')).max())==0;baselines=[frozen]+[mainread(base,z,d,'bidirectional',point_mask=p,image_mask=i)[0] for p,i in ab];reverse=mainread(base,z,d,'bidirectional',reverse_off=True)[0].cpu().numpy();gray=mainread(base,z,d,'bidirectional',image=torch.full_like(d['image'],.5))[0].cpu().numpy()
 result={a:od[b].copy() for a,b in [('source','source'),('frozen_bi','frozen_bi'),('frozen_ABC','frozen_ABC'),('R59','R59'),('R86_direct','F_direct_guard'),('R86_TTA','F_joint1_guard'),('R86_raw','F_joint1_raw')]};result.update(frozen_no_reverse=reverse,frozen_gray_RGB=gray);st={a:dict(legacy[sid][b]) for a,b in [('source','source'),('frozen_bi','frozen_bi'),('frozen_ABC','frozen_ABC'),('R59','R59'),('R86_direct','F_direct_guard'),('R86_TTA','F_joint1_guard'),('R86_raw','F_joint1_raw')]}
 for a in ['frozen_no_reverse','frozen_gray_RGB']:st[a]={'unknown':False,'fallback':False,'no_parameter_update':True}
 record={'variants':{},'targets':meta};saved={'source_boxes':z0['boxes'],'mask_point':torch.stack(pm).cpu().numpy(),'mask_image':torch.stack(im).cpu().numpy(),'s0':frozen.log().cpu().numpy()};s0=frozen.log()
 for v in MODES87+['wrong']:
  result[v+'_direct']=ev[v+'_pi'];saved[v+'_pi']=ev[v+'_pi'];st[v+'_direct']={'unknown':not meta[v]['active'],'fallback':False,'no_parameter_update':True}
 for variant in MODES87+['wrong','candidate_only']:
  restore(m,S);enable79(m);targetname='full' if variant=='candidate_only' else variant;target=torch.tensor(ev[targetname+'_pi'],device='cuda');active=meta[targetname]['active'];trajectory=[];nsteps=3 if variant in ['full','wrong'] else 1
  for step in range(1,nsteps+1):
   accepted=False;scale0=0.;trials=[];ps=[p for _,p in m.tta_parameters()];gc=torch.zeros(sum(p.numel() for p in ps),device='cuda');gr=gc.clone();dC=dT=0.
   if active:
    L,_,C,R=obj(m,base,z,d,s0,target,baselines,pm,im,ws);gc=flat86(torch.autograd.grad(C,ps,retain_graph=True,allow_unused=True),ps);gr=flat86(torch.autograd.grad(R,ps,allow_unused=True),ps);g=gc if variant=='candidate_only' else priority86(gc,gr)[0] if float(gc.norm())>1e-12 else gc*0;g=g*min(1.,1/(float(g.norm())+1e-6));dC=float(gc@g);dT=float((gc+gr)@g);initial=[p.detach().clone() for p in ps]
    if float(g.norm())>1e-12:
     for scale in [1/(2**i) for i in range(9)]:
      apply86(m,initial,-.1*scale*g)
      with torch.no_grad():
       t,_,ct,rt=obj(m,base,z,d,s0,target,baselines,pm,im,ws);calls['trial_objectives']+=1;cb=float(C)-1e-4*scale*.1*max(0,dC);tb=float(L)-1e-4*scale*.1*max(0,dT);ok=float(ct)<=cb+1e-10 and (variant=='candidate_only' or float(t)<=tb+1e-10);trials.append({'scale':scale,'classification':float(ct),'total':float(t),'candidate_bound':cb,'total_bound':tb,'accepted':bool(ok)})
      if ok:accepted=True;scale0=scale;calls['accepted_updates']+=1;break
    if not accepted:apply86(m,initial,torch.zeros_like(gc))
    with torch.no_grad():af,pafter,ca,ra=obj(m,base,z,d,s0,target,baselines,pm,im,ws)
    before,cbefore,rbefore,after,cafter,rafter=map(float,[L,C,R,af,ca,ra]);assert not accepted or cafter<=cbefore+1e-10;assert variant=='candidate_only' or not accepted or after<=before+1e-10
   else:before=cbefore=rbefore=after=cafter=rafter=None;pafter=frozen
   delta=float(torch.sqrt(sum((p-torch.tensor(S[n],device='cuda')).square().sum() for n,p in m.tta_parameters())));trajectory.append({'step':step,'accepted':accepted,'scale':scale0,'trials':trials,'trial_count':len(trials),'total_before':before,'total_after':after,'classification_before':cbefore,'classification_after':cafter,'recovery_before':rbefore,'recovery_after':rafter,'classification_gradient_norm':float(gc.norm()),'recovery_gradient_norm':float(gr.norm()),'gradient_cosine':float(gc@gr/(gc.norm()*gr.norm())) if float(gc.norm()*gr.norm())>1e-20 else None,'classification_direction_dot':dC,'total_direction_dot':dT,'parameter_L2':delta})
   if step in [1,3]:
    name='candidate_only1' if variant=='candidate_only' else variant+'_tta'+str(step);result[name]=pafter.detach().cpu().numpy();np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+name+'.npz'),**patch85(m));st[name]={'unknown':not active,'fallback':not active or delta==0,'no_parameter_update':delta==0}
  record['variants'][variant]=trajectory
 restore(m,S);m.requires_grad_(False);scores=np.stack([result[a] for a in ARMS87]);assert np.isfinite(scores).all();q=scores.argmax(-1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS87),scores=scores,queries=q,selected_original_boxes=z0['boxes'][q],**saved);dump(O/'records_PRIVATE'/(sid+'.json'),record);statuses.append({'id':sid,'arm_statuses':st});print('RUN87',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp and all(np.array_equal(p.detach().cpu().numpy(),S[n]) for n,p in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',{'n':len(rows),'arms':len(ARMS87),**calls,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'TTA_parameters':sum(p.numel() for _,p in m.tta_parameters()),'GT_read':False});seal(O,'R87_ALL_REGISTERED_DIRECT_TTA_AND_COMPONENT_PREDICTIONS_SEALED',n=len(rows),arm_names=ARMS87);print(read(O/'COUNTS.json'))

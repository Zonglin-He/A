"""Full R86 prototype: two bases, matched direct/raw/qualified controls."""
from core86 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();plan=read(H86/'experiment/DESIGN_LOCK.json');pin(plan);start=time.perf_counter();base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);m=Fusion85(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/plan['init_state']);restore(m,S);rows=read(H86/'experiment/ALL366_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p and read(B86/'evidence366_attempt002/cases_PRIVATE'/(r['id']+'.json'))['factor_kinds']) for p in ['drone','quad']]
O=B86/('engineering_attempt001' if args.engineering else 'prediction366_attempt001');assert not O.exists()
for name in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE']:(O/name).mkdir(parents=True,exist_ok=False)
statuses=[];calls={'grounding':0,'objective':0,'accepted_updates':0,'trial_objectives':0};legacy={x['id']:x['arm_statuses'] for x in read(B82/'prediction2640_attempt001/READOUT_STATUS_PRIVATE.json')}
def obj(*a):
 calls['objective']+=1;calls['grounding']+=3;return objective86(*a)
def mainread(*a,**kw):calls['grounding']+=1;return m.main(*a,**kw)
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d=item80(r);old=load(B85/'prediction366_attempt002/samples_PRIVATE'/(sid+'.npz'));od=dict(zip(old['arm_names'],old['scores']));ev=load(B86/'evidence366_attempt002/samples_PRIVATE'/(sid+'.npz'));meta=read(B86/'evidence366_attempt002/cases_PRIVATE'/(sid+'.json'));factors=[{'kind':k,'support':s,'counter':c} for k,s,c in zip(meta['factor_kinds'],ev['support'],ev['counter'])];region={'point':ev['region_point'],'image':ev['region_image']};ab=[mask86(r,d,region,k) for k in [86001,86002]];pm=[a for a,b in ab];im=[b for a,b in ab];ws=local_weights85(d,pm,im,region);restore(m,S);m.requires_grad_(False)
 with torch.no_grad():
  frozen=mainread(base,z,d,'bidirectional')[0];assert float(abs(frozen-torch.tensor(od['frozen_bi'],device='cuda')).max())<1e-10;baselines=[frozen.detach()]+[mainread(base,z,d,'bidirectional',point_mask=a,image_mask=b)[0].detach() for a,b in ab]
 result={a:od[v].copy() for a,v in [('source','source'),('frozen_bi','frozen_bi'),('R82_BT','R82_BT'),('frozen_ABC','frozen_ABC'),('R59','R59'),('R85_full1','full1')]};st={a:dict(legacy[sid][v]) for a,v in [('source','source'),('frozen_bi','frozen_bi'),('R82_BT','bt_correct'),('frozen_ABC','frozen_ABC'),('R59','R59')]};st['R85_full1']=read(B85/'prediction366_attempt002/READOUT_STATUS_PRIVATE.json')[nr]['arm_statuses']['full1'] if not args.engineering else {'unknown':False,'fallback':False,'no_parameter_update':False};saved={'source_boxes':z0['boxes'],'mask_point':torch.stack(pm).cpu().numpy(),'mask_image':torch.stack(im).cpu().numpy()};record={'bases':{},'A256_fixed':True}
 for bname in ['S','F']:
  s0=torch.tensor(old['s0'],device='cuda') if bname=='S' else frozen.log();p0=s0.softmax(-1).detach();pnp=p0.cpu().numpy();pi,eligible,flog,known=target86(pnp,factors,ev['labels'],sid);wrong,wrongel,wronglog,_=target86(pnp,factors,ev['labels'],sid,True);active=bool(any(x['active'] for x in flog));saved[bname+'_s0']=s0.cpu().numpy();saved[bname+'_pi']=pi;saved[bname+'_eligible']=eligible;saved[bname+'_mismatch_pi']=wrong;saved[bname+'_mismatch_eligible']=wrongel;saved[bname+'_known']=known
  for k,prob,el in [('direct',pi,eligible),('mismatch_direct',wrong,wrongel)]:
   for rd,score in [('raw',prob),('guard',guarded86(prob,el))]:name=bname+'_'+k+'_'+rd;result[name]=score;st[name]={'unknown':not active,'fallback':False,'no_parameter_update':True}
  recbase={'active':active,'factor_logs':flog,'mismatch_factor_logs':wronglog,'known_queries':int(known.sum()),'eligible_queries':int(eligible.sum()),'direct_query_changes_base':int(pi.argmax())!=int(pnp.argmax()),'direct_guard_changes_base':int(guarded86(pi,eligible).argmax())!=int(pnp.argmax()),'unknown_probability_error':float(abs(pi[~known]-pnp[~known]).max()) if (~known).any() else 0.,'variants':{}}
  assert recbase['unknown_probability_error']==0
  for variant in ['candidate','joint','mismatch','sum']:
   restore(m,S);enable79(m);target=torch.tensor(wrong if variant=='mismatch' else pi,device='cuda');el=wrongel if variant=='mismatch' else eligible;trajectory=[];nsteps=1 if variant=='sum' else 3
   for step in range(1,nsteps+1):
    accepted=False;beta=0.;trials=[];gc=torch.zeros(sum(p.numel() for _,p in m.tta_parameters()),device='cuda');gr=torch.zeros_like(gc);gp=gr;dC=dT=0.
    if active:
     L,pp,C,R=obj(m,base,z,d,s0,target,baselines,pm,im,ws);ps=[p for _,p in m.tta_parameters()];gc=flat86(torch.autograd.grad(C,ps,retain_graph=True,allow_unused=True),ps);gr=flat86(torch.autograd.grad(R,ps,allow_unused=True),ps)
     if variant=='candidate':g=gc;gp=gr*0
     elif variant=='sum':g=gc+gr;gp=gr
     elif float(gc.norm())>1e-12:g,gp=priority86(gc,gr)
     else:g=gc*0;gp=gr*0
     clipped=g*min(1.,1/(float(g.norm())+1e-6));dC=float(gc@clipped);dT=float((gc+gr)@clipped);oldparams=[p.detach().clone() for p in ps]
     if float(g.norm())>1e-12:
      for scale in [1/(2**i) for i in range(9)]:
       apply86(m,oldparams,-.1*scale*clipped)
       with torch.no_grad():
        test,_,tc,tr=obj(m,base,z,d,s0,target,baselines,pm,im,ws);calls['trial_objectives']+=1;cb=float(C)-1e-4*scale*.1*max(0.,dC);tb=float(L)-1e-4*scale*.1*max(0.,dT);cok=float(tc)<=cb+1e-10;tok=float(test)<=tb+1e-10;ok=cok if variant=='candidate' else tok if variant=='sum' else cok and tok;trials.append({'scale':scale,'classification':float(tc),'total':float(test),'candidate_bound':cb,'total_bound':tb,'candidate_ok':cok,'total_ok':tok,'accepted':bool(ok)})
       if ok:accepted=True;beta=scale;calls['accepted_updates']+=1;break
     if not accepted:apply86(m,oldparams,torch.zeros_like(gc))
     with torch.no_grad():after,pafter,ca,ra=obj(m,base,z,d,s0,target,baselines,pm,im,ws)
     before=float(L);cbefore=float(C);rbefore=float(R);af=float(after);caf=float(ca);raf=float(ra)
    else:before=cbefore=rbefore=af=caf=raf=None;pafter=p0
    delta=float(torch.sqrt(sum((p-torch.tensor(S[n],device='cuda')).square().sum() for n,p in m.tta_parameters())));cn=float(gc.norm());rn=float(gr.norm());cos=float(gc@gr/(gc.norm()*gr.norm())) if cn*rn>1e-20 else None
    trajectory.append({'step':step,'accepted':accepted,'scale':beta,'trial_count':len(trials),'trials':trials,'total_before':before,'total_after':af,'classification_before':cbefore,'classification_after':caf,'recovery_before':rbefore,'recovery_after':raf,'classification_gradient_norm':cn,'recovery_gradient_norm':rn,'projected_recovery_norm':float(gp.norm()),'gradient_cosine':cos,'classification_dot_projected_recovery':float(gc@gp),'classification_direction_dot':dC,'total_direction_dot':dT,'parameter_L2':delta});assert af is None or np.isfinite(af)
    if variant in ['joint','mismatch'] and accepted:assert caf<=cbefore+1e-10 and af<=before+1e-10
    if step in [1,3]:
     stem=bname+'_'+variant+str(step);prob=pafter.detach().cpu().numpy();saved[stem+'_raw']=prob
     for rd,score in [('raw',prob),('guard',guarded86(prob,el))]:name=stem+'_'+rd;result[name]=score;st[name]={'unknown':not active,'fallback':not active or delta==0,'no_parameter_update':delta==0}
     np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+stem+'.npz'),**patch85(m))
   recbase['variants'][variant]=trajectory
  record['bases'][bname]=recbase
 restore(m,S);m.requires_grad_(False);assert set(result)==set(ARMS86);scores=np.stack([result[a] for a in ARMS86]);assert np.isfinite(scores).all();queries=scores.argmax(-1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS86),scores=scores,queries=queries,selected_original_boxes=z0['boxes'][queries],**saved);dump(O/'records_PRIVATE'/(sid+'.json'),record);statuses.append({'id':sid,'arm_statuses':st});print('RUN86',nr+1,len(rows),'factors',len(factors),'active',active,flush=True)
assert fingerprint(base)==fp and all(np.array_equal(p.detach().cpu().numpy(),S[n]) for n,p in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',{'n':len(rows),'arms':len(ARMS86),**calls,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'TTA_parameters':sum(p.numel() for _,p in m.tta_parameters()),'GT_read':False});seal(O,'R86_TWO_BASES_TRISTATE_AND_DISCRIMINATION_PRIORITY_ALL_PREDICTIONS_SEALED',n=len(rows),arm_names=ARMS86);print(read(O/'COUNTS.json'))

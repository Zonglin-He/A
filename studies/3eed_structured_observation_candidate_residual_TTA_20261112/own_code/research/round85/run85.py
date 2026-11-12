"""Complete prototype and fixed ablations; all output sealed before task scoring."""
from core85 import *
import time,argparse
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();plan=read(H85/'experiment/DESIGN_LOCK.json');pin(plan);start=time.perf_counter();base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);m=Fusion85(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/plan['init_state']);restore(m,S);rows=read(H85/'experiment/ALL366_PRIVATE.json')
if a.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
O=B85/('engineering_attempt002' if a.engineering else 'prediction366_attempt002');assert not O.exists();(O/'samples_PRIVATE').mkdir(parents=True);(O/'states_PRIVATE').mkdir();(O/'records_PRIVATE').mkdir();statuses=[];steps=0;readers=0;trial_calls=0;legacy_status={v['id']:v['arm_statuses'] for v in read(B82/'prediction2640_attempt001/READOUT_STATUS_PRIVATE.json')}
def mainread(*args,**kwargs):
 global readers;readers+=1;return m.main(*args,**kwargs)
def leakcheck(z,d,pm,im):
 # Reverse keys cannot expose the masked feature. Cross-modal context remains allowed.
 a=pm[0];idx=d['i2p'].clamp_min(0);rw=d['i2p_weight']*(~a[idx])*(d['i2p']>=0);assert not rw[a[idx]].any();assert all(not b[d['cell'][aa]].any() for aa,b in zip(pm,im));assert torch.equal(z['query_points_sample_inds'].long(),d['query_anchor_indices'].long())
for ni,r in enumerate(rows):
 restore(m,S);m.requires_grad_(False);sid=r['id'];z0,z,d=item80(r);ev=load(B85/'observations366_attempt001/samples_PRIVATE'/(sid+'.npz'));meta=read(B85/'observations366_attempt001/cases_PRIVATE'/(sid+'.json'));graph=meta['graph'];assert meta['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest()
 old=load(B82/'prediction2640_attempt001/samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']))
 with torch.no_grad():
  h,u=original84(base,z);s0=logits85(base,z,h,u);p0=s0.softmax(-1);assert int(p0.argmax())==int(oldp['source'].argmax());entity=torch.stack([phrase85(base,z,e['phrase']) for e in graph['entities']]) if graph['entities'] else s0[None];entity_np=entity.cpu().numpy();measure=(ev['means'],ev['counts'],ev['coverage'],ev['interval'],ev['reliability']);cost,details=evidence85(graph,entity_np,measure,ev['labels'],ev['reps']);shcost,_=evidence85(graph,entity_np,measure,ev['labels'],ev['reps'],'mismatch',sid);region=region85(ev,entity_np,s0.cpu().numpy(),{'cell':d['cell'].cpu().numpy(),'visible':d['visible'].cpu().numpy()});pi=(s0-torch.tensor(cost,device='cuda')).softmax(-1).detach();pish=(s0-torch.tensor(shcost,device='cuda')).softmax(-1).detach();active=bool(np.ptp(cost)>1e-8);frozen,_,_=mainread(base,z,d,'bidirectional');assert np.max(abs(frozen.cpu().numpy()-oldp['frozen_bi']))<1e-10
 result={k:oldp[v].copy() for k,v in [('source','source'),('frozen_bi','frozen_bi'),('R82_BT','bt_correct'),('frozen_ABC','frozen_ABC'),('R59','R59')]};result.update(direct_constraints=pi.cpu().numpy(),mismatch_direct=pish.cpu().numpy());st={k:{'unknown':not active,'fallback':False,'no_parameter_update':True} for k in result};st.update({k:legacy_status[sid][v] for k,v in [('source','source'),('frozen_bi','frozen_bi'),('R82_BT','bt_correct'),('frozen_ABC','frozen_ABC'),('R59','R59')]});record={'active_evidence':active,'cost_range':float(np.ptp(cost)),'cost_at_source_top1':float(cost[int(p0.argmax())]),'relations':details,'variants':{},'A256_boxes_unchanged':True,'source_zero_step_exact':True};saved={'s0':s0.cpu().numpy(),'pi':pi.cpu().numpy(),'cost':cost,'mismatch_cost':shcost,'entity_scores':entity_np,'region_point':region['point'],'region_image':region['image']};zeroerr=0.
 for variant in ['full','no_constraints','protected','mismatch']:
  restore(m,S);m.requires_grad_(False);ab=[mask85(r,d,region,k,variant=='protected') for k in [85001,85002]];pm=[x[0] for x in ab];im=[x[1] for x in ab];ws=local_weights85(d,pm,im,region);leakcheck(z,d,pm,im);target=p0.detach() if variant=='no_constraints' else pish if variant=='mismatch' else pi
  with torch.no_grad():
   baselines=[frozen.detach()]+[mainread(base,z,d,'bidirectional',point_mask=x,image_mask=y)[0].detach() for x,y in ab];zero=anchored85(frozen,baselines[0],s0);zeroerr=max(zeroerr,float(abs(zero-p0).max()));assert zeroerr<1e-14
   if variant=='full':
    votes=torch.stack([p0]+[anchored85(b,baselines[0],s0) for b in baselines[1:]]).mean(0);result['direct_multiview_constraints']=(votes.log()-torch.tensor(cost,device='cuda')).softmax(-1).cpu().numpy();st['direct_multiview_constraints']={'unknown':not active,'fallback':False,'no_parameter_update':True}
  trajectory=[];enable79(m)
  for step in range(1,4):
   L,p,klass,rec=objective85(m,base,z,d,s0,target,baselines,pm,im,ws);ps=[v for _,v in m.tta_parameters()]
   if active:
    cg=torch.autograd.grad(klass,ps,retain_graph=True,allow_unused=True) if step==1 else None
    gg=torch.autograd.grad(L,ps,allow_unused=True);gn=torch.sqrt(sum(v.square().sum() for v in gg if v is not None));scale=min(1.,1/(float(gn)+1e-6));
    before_params=[v.detach().clone() for v in ps];accepted=False;beta=0.;trials=[]
    if float(gn)>1e-12:
     for b in [1/(2**k) for k in range(9)]:
      with torch.no_grad():
       for par,oldval,grad in zip(ps,before_params,gg):par.copy_(oldval if grad is None else oldval-b*.1*scale*grad)
       test,_,_,_=objective85(m,base,z,d,s0,target,baselines,pm,im,ws);trial_calls+=1;bound=float(L)-1e-4*b*.1*scale*float(gn.square());ok=np.isfinite(float(test)) and float(test)<=bound+1e-10;trials.append({'scale':b,'objective':float(test),'bound':bound,'accepted':bool(ok)})
      if ok:accepted=True;beta=b;steps+=1;break
    if not accepted:
     with torch.no_grad():
      for par,oldval in zip(ps,before_params):par.copy_(oldval)
   else:gn=L.new_zeros(());cg=None;gg=None;accepted=False;beta=0.;trials=[]
   cn=rn=cos=None
   if cg is not None:
    cv=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(ps,cg)]);gv=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(ps,gg)]);rv=gv-cv;cn=float(cv.norm());rn=float(rv.norm());cos=float(cv@rv/(cv.norm()*rv.norm())) if cn*rn>1e-20 else None
   with torch.no_grad():
    after,pp,kafter,rafter=objective85(m,base,z,d,s0,target,baselines,pm,im,ws);delta=float(torch.sqrt(sum((v-torch.as_tensor(S[n],device='cuda')).square().sum() for n,v in m.tta_parameters())));trajectory.append({'step':step,'accepted':accepted,'step_scale':beta,'trial_count':len(trials),'trials':trials,'objective_before':float(L),'objective_after':float(after),'classification_before':float(klass),'classification_after':float(kafter),'recovery_before':float(rec),'recovery_after':float(rafter),'gradient_norm':float(gn),'classification_gradient_norm':cn,'recovery_gradient_norm':rn,'gradient_cosine':cos,'parameter_L2':delta,'point_targets':int((pm[0]|pm[1]).sum()),'masked_anchor_seed_targets':int((pm[0]|pm[1])[d['query_anchor_indices'].long().unique()].sum()),'local_point_targets':int(((pm[0]|pm[1])&torch.tensor(region['point'],device='cuda')).sum())});assert np.isfinite(float(after))
    if step in [1,3]:
     arm=variant+str(step);result[arm]=pp.cpu().numpy() if active else p0.cpu().numpy();st[arm]={'unknown':not active,'fallback':not active or delta==0,'no_parameter_update':delta==0};np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch85(m))
  record['variants'][variant]=trajectory
 record['zero_step_max_error']=zeroerr;assert set(result)==set(ARMS85);scores=np.stack([result[k] for k in ARMS85]);queries=scores.argmax(-1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS85),scores=scores,queries=queries,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][queries],**saved);dump(O/'records_PRIVATE'/(sid+'.json'),record);statuses.append({'id':sid,'arm_statuses':st});print('RUN85',ni+1,len(rows),'active',active,flush=True)
 restore(m,S);m.requires_grad_(False)
assert fingerprint(base)==fp and all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',{'n':len(rows),'actual_optimizer_steps':steps,'explicit_reader_calls':readers,'objective_calls':len(rows)*4*3*2+trial_calls,'line_search_objective_calls':trial_calls,'grounding_calls_per_objective':3,'auxiliary_fusion_calls_per_objective':2,'entity_phrase_suffix_calls':sum(len(read(B85/'observations366_attempt001/cases_PRIVATE'/(r['id']+'.json'))['graph']['entities']) or 0 for r in rows),'original_fulltext_suffix_calls':len(rows),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'TTA_parameters':sum(v.numel() for _,v in m.tta_parameters()),'task_GT_read':False});seal(O,'R85_COMPLETE_PROTOTYPE_AND_ABLATIONS_PREDICTIONS_SEALED_BEFORE_TASK_JOIN',n=len(rows),arm_names=ARMS85);print(read(O/'COUNTS.json'))

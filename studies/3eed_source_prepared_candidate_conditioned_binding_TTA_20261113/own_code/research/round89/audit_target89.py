"""Independent native target route, source state, fixed-box and actual-loss replay."""
from target_core89 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
start=time.perf_counter();P=B89/'prediction336_attempt001';seal0=read(P/'SEALED.json');E=B89/'target_evidence336_attempt001';es=read(E/'SEALED.json');choice=read(B89/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');states={b:load(ROOT/choice['models'][b]['state_path']) for b in BRANCHES89};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.manual_seed(0);m=Fusion89(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);load_R76(m);initial=state(m);fp=fingerprint(base);worst=routeerr=losserr=0.;endpoints=losschecks=accepted=0;heldout=[]
def err(p,s):return float(abs(p.detach().cpu().numpy()-s).max())
for nr,r in enumerate(read(H89/'experiment/DEV336_PRIVATE.json')):
 sid=r['id'];f=P/'samples_PRIVATE'/(sid+'.npz');assert sha(f)==seal0['files'][f.relative_to(P).as_posix()];a=load(f);rec=read(P/'records_PRIVATE'/(sid+'.json'));ef=E/'samples_PRIVATE'/(sid+'.npz');assert sha(ef)==es['files'][ef.relative_to(E).as_posix()];z0,z,d,ec,pm,im,ws=item89(r);scores=dict(zip(a['arm_names'],a['scores']));assert np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(-1)) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);assert np.array_equal(a['mask_point'],torch.stack(pm).cpu().numpy()) and np.array_equal(a['mask_image'],torch.stack(im).cpu().numpy());restore(m,initial);m.requires_grad_(False);m.route=None;m.use_interface=False;native=[];hk=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:native.append(o[1].detach().cpu().numpy()[0]))
 with torch.no_grad():p0,_,_=m.main(base,z,d,'bidirectional')
 hk.remove();assert np.array_equal(native[0],ec['native_role_attention']);assert np.array_equal(p0.cpu().numpy(),scores['frozen_bi']);graph=read(E/'records_PRIVATE'/(sid+'.json'))['graph'];doc=read(ROOT/'crossplatform/round48/parse336_attempt001/dependencies_PRIVATE'/(sid+'.json'));roles,offsets=token_roles88(base,r,z0,graph,doc);rebuilt,_=route89(r,{**z0,'frozen_probability':p0.cpu().numpy()},load(casepath(r)),graph,roles,native[0],offsets,doc)
 for k in rebuilt:
  if rebuilt[k].size:routeerr=max(routeerr,float(np.max(abs(rebuilt[k].astype(float)-ec[k].astype(float)))))
 assert routeerr==0;m.route=tensor_route89(ec)
 specs=[(b,b,'full' if b!='query' else 'query_only','full') for b in BRANCHES89]+[(v,'candidate',v,'full') for v in VARIANTS89]+[('candidate_physical','candidate','full','physical'),('candidate_no_correspondence','candidate','full','no_correspondence')]
 for name,branch,variant,kind in specs:
  restore(m,states[branch]);m.requires_grad_(False);m.use_interface=branch!='continue';m.variant=variant
  with torch.no_grad():refp,refq,ex=m.main(base,z,d,'bidirectional');refs=ex['local_reads']
  if name in BRANCHES89+VARIANTS89:worst=max(worst,err(refp,scores[name+'_frozen']));endpoints+=1
  if name in BRANCHES89:
   with torch.no_grad():mean=torch.stack([refp]+[m.main(base,z,d,'bidirectional',point_mask=x,image_mask=y)[0] for x,y in zip(pm,im)]).mean(0)
   worst=max(worst,err(mean,scores[name+'_mean3']));endpoints+=1
  for j,t in enumerate(rec[name]):
   for ti,x in enumerate(t['trials']):
    assert x['scale']==1/2**ti;bound=t['before']-1e-4*.1*x['scale']*t['directional_derivative'];assert abs(bound-x['bound'])<1e-10 and x['accepted']==(x['actual_total']<=bound+1e-10)
   accepted+=int(t['accepted']);assert not t['accepted'] or t['parameter_L2']>0 and t['after']<=t['before']+1e-10
   if j==1:
    # Step2 has no prediction endpoint; reconstruct its accepted parameter move from step1.
    enable79(m);lv,_,_=objective_target89(m,base,z,d,refp,refq,refs,pm,im,ws,kind);ps=[p for _,p in m.tta_parameters()];raw=flat89(torch.autograd.grad(lv,ps,allow_unused=True),ps);g=raw*min(1.,1/(float(raw.norm())+1e-6));old=[p.detach().clone() for p in ps];apply89(m,old,-.1*t['scale']*g if t['accepted'] else torch.zeros_like(g));m.requires_grad_(False)
   else:
    with torch.no_grad():lv,_,parts=objective_target89(m,base,z,d,refp,refq,refs,pm,im,ws,kind)
   losserr=max(losserr,abs(float(lv)-t['before']));losschecks+=1
   step=j+1
   if step in [1,3]:
    arm=name+str(step) if name in ['candidate_physical','candidate_no_correspondence'] else name+'_tta'+str(step);sf=P/'states_PRIVATE'/(sid+'_'+arm+'.npz');assert sha(sf)==seal0['files'][sf.relative_to(P).as_posix()];patch=load(sf);assert set(patch)=={n for n,_ in m.tta_parameters()}
    sd=state(m);sd.update(patch);restore(m,sd);m.requires_grad_(False)
   with torch.no_grad():lv,pv,parts=objective_target89(m,base,z,d,refp,refq,refs,pm,im,ws,kind)
   losserr=max(losserr,abs(float(lv)-t['after']));losschecks+=1
   for k,v in parts.items():losserr=max(losserr,abs(float(v)-t['parts_after'][k]));losschecks+=1
   if step in [1,3]:worst=max(worst,err(pv,scores[arm]));endpoints+=1
  assert all(np.array_equal(v.detach().cpu().numpy(),states[branch][k]) for k,v in m.state_dict().items() if k not in {n for n,_ in m.tta_parameters()})
 # One unused third missingness, postseal descriptive representation check.
 restore(m,states['candidate']);m.use_interface=True;m.variant='full';m.requires_grad_(False);region={'point':ec['region_point'],'image':ec['region_image']};hp,hi=mask89(r,d,region,88003)
 with torch.no_grad():_,_,full=m.main(base,z,d,'bidirectional');target=full['local_reads'];_,_,before=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi);ss=state(m);ss.update(load(P/'states_PRIVATE'/(sid+'_candidate_tta1.npz')));restore(m,ss);_,_,after=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi)
 heldout.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'third_mask_representation_error_before':float((before['local_reads']-target).square().sum(-1).mean()) if target.numel() else None,'third_mask_representation_error_after':float((after['local_reads']-target).square().sum(-1).mean()) if target.numel() else None});assert worst<1e-10 and losserr<1e-10
 if (nr+1)%32==0:print('AUDIT_TARGET89',nr+1,flush=True)
assert fingerprint(base)==fp and accepted==read(P/'COUNTS.json')['accepted_updates'];O=H89/'independent_native_audit_attempt001';O.mkdir();out={'state':'PASS_R89_ALL336_NATIVE_PREPARED_HEAD_ROUTES_ENDPOINTS_AND_ACTUAL_LOSSES','n':336,'arm_names':ARMS89,'saved_endpoint_replays':endpoints,'actual_loss_scalar_checks':losschecks,'accepted_updates':accepted,'accepted_total_increases':0,'max_probability_error':worst,'max_route_error':routeerr,'max_actual_loss_error':losserr,'canonical_source_256_boxes_and_readout_verified':True,'prepared_heads_not_updated_at_test':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start,'new_task_scores':0};dump(O/'AUDIT_RESULT.json',out);dump(H89/'HELDOUT_OBSERVATION_ROWS.json',heldout);print(out)

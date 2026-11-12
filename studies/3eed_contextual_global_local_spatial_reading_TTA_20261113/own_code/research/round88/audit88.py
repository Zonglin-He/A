"""Independent sealed native read, role/space reconstruction and actual-loss audit."""
from core88 import *
from routing88 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
start=time.perf_counter();P=B88/'prediction336_attempt001';seal0=read(P/'SEALED.json');E=B88/'evidence336_attempt001';es=read(E/'SEALED.json');base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));m=Fusion88(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');worst=0.;routeerr=0.;losserr=0.;endpoints=losschecks=accepted=0;heldout=[]
with torch.no_grad():
 for nr,r in enumerate(read(H88/'experiment/DEV336_PRIVATE.json')):
  sid=r['id'];f=P/'samples_PRIVATE'/(sid+'.npz');assert sha(f)==seal0['files'][f.relative_to(P).as_posix()];a=load(f);rec=read(P/'records_PRIVATE'/(sid+'.json'));ef=E/'samples_PRIVATE'/(sid+'.npz');assert sha(ef)==es['files'][ef.relative_to(E).as_posix()];ec=load(ef);z0,z,d,_,pm,im,ws=data88(r);assert np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(-1)) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);assert np.array_equal(a['mask_point'],torch.stack(pm).cpu().numpy()) and np.array_equal(a['mask_image'],torch.stack(im).cpu().numpy());graph=read(E/'records_PRIVATE'/(sid+'.json'))['graph'];doc=read(ROOT/'crossplatform/round48/parse336_attempt001/dependencies_PRIVATE'/(sid+'.json'));roles,offsets=token_roles88(base,r,z0,graph,doc);restore(m,S);m.route=None;m.route_mode='off';native=[];hk=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:native.append(o[1].cpu().numpy()[0]));fp,_,_=m.main(base,z,d,'bidirectional');hk.remove();assert np.array_equal(native[0],ec['native_role_attention']);assert np.array_equal(fp.cpu().numpy(),a['scores'][ARMS88.index('frozen_bi')]);rebuilt,_=build88(r,{**z0,'frozen_probability':fp.cpu().numpy()},load(casepath(r)),graph,roles,native[0]);
  for key in rebuilt:routeerr=max(routeerr,float(np.max(abs(rebuilt[key].astype(float)-ec[key].astype(float)))) if rebuilt[key].size else 0.)
  assert routeerr<1e-12;m.route=tensor_route88(ec)
  for variant in MODES88+['physical_only','no_rank','no_correspondence']:
   mode=variant if variant in MODES88 else 'full';kind='full' if variant in MODES88 else variant;m.route_mode=mode;restore(m,S);refp,refq,ex=m.main(base,z,d,'bidirectional');refs=ex['local_reads']
   if variant in MODES88:worst=max(worst,float(abs(refp.cpu().numpy()-a['scores'][ARMS88.index(variant+'_direct')]).max()));endpoints+=1
   tr=rec[variant]
   for j,t in enumerate(tr):
    for ti,x in enumerate(t['trials']):
     assert x['scale']==1/2**ti;bound=t['before']-1e-4*.1*x['scale']*t['directional_derivative'];assert abs(bound-x['bound'])<1e-10 and x['accepted']==(x['actual_total']<=bound+1e-10)
    if t['accepted']:assert t['parameter_L2']>0 and t['after']<=t['before']+1e-10;accepted+=1
    step=j+1
    if step not in [1,3]:continue
    if step==1:
     restore(m,S);lv,_,parts=objective88(m,base,z,d,refp,refq,refs,pm,im,ws,kind);losserr=max(losserr,abs(float(lv)-t['before']));losschecks+=1
     for k,v in parts.items():losserr=max(losserr,abs(float(v)-t['parts_before'][k]));losschecks+=1
    arm=variant+'_tta'+str(step) if variant in MODES88 else variant+'1';sf=P/'states_PRIVATE'/(sid+'_'+arm+'.npz');assert sha(sf)==seal0['files'][sf.relative_to(P).as_posix()];put_patch(m,load(sf));lv,pv,parts=objective88(m,base,z,d,refp,refq,refs,pm,im,ws,kind);worst=max(worst,float(abs(pv.cpu().numpy()-a['scores'][ARMS88.index(arm)]).max()));losserr=max(losserr,abs(float(lv)-t['after']));losschecks+=1;endpoints+=1
    for k,v in parts.items():losserr=max(losserr,abs(float(v)-t['parts_after'][k]));losschecks+=1
   restore(m,S)
  # Held-out third missingness only descriptive: never participates in update/acceptance.
  m.route_mode='full';region={'point':ec['region_point'],'image':ec['region_image']};hp,hi=mask86(r,d,region,88003);restore(m,S);_,_,full=m.main(base,z,d,'bidirectional');target=full['local_reads'];_,_,before=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi);put_patch(m,load(P/'states_PRIVATE'/(sid+'_full_tta1.npz')));_,_,after=m.main(base,z,d,'bidirectional',point_mask=hp,image_mask=hi);b=float((before['local_reads']-target).square().sum(-1).mean()) if target.numel() else None;af=float((after['local_reads']-target).square().sum(-1).mean()) if target.numel() else None;heldout.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'third_mask_representation_error_before':b,'third_mask_representation_error_after':af});restore(m,S)
  assert worst<1e-10 and losserr<1e-10
  if (nr+1)%64==0:print('AUDIT88',nr+1,flush=True)
assert accepted==read(P/'COUNTS.json')['accepted_updates'];O=H88/'independent_native_audit_attempt001';O.mkdir();out={'state':'PASS_R88_ALL336_NATIVE_CONTEXT_ROUTES_ENDPOINTS_AND_ACTUAL_LOSSES','n':336,'arm_names':ARMS88,'saved_endpoint_replays':endpoints,'actual_loss_scalar_checks':losschecks,'accepted_updates':accepted,'accepted_total_increases':0,'max_probability_or_score_error':worst,'max_route_error':routeerr,'max_actual_loss_error':losserr,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start,'new_task_scores':0};dump(O/'AUDIT_RESULT.json',out);dump(H88/'HELDOUT_OBSERVATION_ROWS.json',heldout);print(out)

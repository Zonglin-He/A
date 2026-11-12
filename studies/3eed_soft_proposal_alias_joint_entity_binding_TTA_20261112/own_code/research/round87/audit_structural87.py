"""Independent pre-GT soft graph/cache mapping and native state/loss replay."""
from structural_core87 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
start=time.perf_counter();P=B87/'structural_prediction336_attempt001';seal0=read(P/'SEALED.json');E=B87/'structural_evidence336_attempt002';eseal=read(E/'SEALED.json');base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.set_num_threads(4);m=Fusion85(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');worst=0.;targeterror=0.;losserror=0.;endpoints=0;losschecks=0;accepted=0;ups=0
with torch.no_grad():
 for nr,r in enumerate(read(H87/'experiment/DEV336_PRIVATE.json')):
  sid=r['id'];file=P/'samples_PRIVATE'/(sid+'.npz');assert sha(file)==seal0['files'][file.relative_to(P).as_posix()];a=load(file);record=read(P/'records_PRIVATE'/(sid+'.json'));z0,z,d=item80(r);ev,graph,es=prepared87(r);g=load(casepath(r));A=alias87(z0['boxes'],ev['support'],z0['vis0'][0],g['cell'],g['visible']);cachefile=E/'samples_PRIVATE'/(sid+'.npz');assert sha(cachefile)==eseal['files'][cachefile.relative_to(E).as_posix()];cache=load(cachefile);assert np.max(abs(A-cache['alias']))<1e-13;assert np.array_equal(a['queries'],a['scores'].argmax(-1)) and np.array_equal(a['source_boxes'],z0['boxes']) and np.array_equal(a['selected_original_boxes'],z0['boxes'][a['queries']]);restore(m,S);frozen=m.main(base,z,d,'bidirectional')[0];s0=torch.tensor(a['s0'],device='cuda');assert float(abs(anchored85(frozen,frozen,s0)-frozen).max())<1e-15
  region0=load(B86/'evidence366_attempt002/samples_PRIVATE'/(sid+'.npz'));region={'point':region0['region_point'],'image':region0['region_image']};ab=[mask86(r,d,region,k) for k in [86001,86002]];pm=[p for p,i in ab];im=[i for p,i in ab];assert np.array_equal(torch.stack(pm).cpu().numpy(),a['mask_point']) and np.array_equal(torch.stack(im).cpu().numpy(),a['mask_image']);ws=local_weights85(d,pm,im,region);baselines=[frozen]+[m.main(base,z,d,'bidirectional',point_mask=p,image_mask=i)[0] for p,i in ab]
  for arm,kw in [('frozen_no_reverse',{'reverse_off':True}),('frozen_gray_RGB',{'image':torch.full_like(d['image'],.5)})]:worst=max(worst,float(abs(m.main(base,z,d,'bidirectional',**kw)[0].cpu().numpy()-a['scores'][ARMS87.index(arm)]).max()))
  for v in MODES87+['wrong']:
   p,info,belief=target87(frozen.cpu().numpy(),es,graph,ev,A,mode='full' if v=='wrong' else v,seed=sid,mismatch=v=='wrong');targeterror=max(targeterror,float(abs(p-a[v+'_pi']).max()));assert targeterror<1e-13 and np.max(abs(p-a['scores'][ARMS87.index(v+'_direct')]))<1e-13;assert np.allclose(belief,cache[v+'_beliefs'],rtol=0,atol=1e-13)
  for v in MODES87+['wrong']:
   assert np.max(abs(.5*(frozen.cpu().numpy()+a[v+'_pi'])-a['scores'][ARMS87.index(v+'_half')]))<1e-13
  for variant in MODES87+['wrong','candidate_only']:
   targetname='full' if variant=='candidate_only' else variant;pi=torch.tensor(a[targetname+'_pi'],device='cuda');trajectory=record['variants'][variant];active=record['targets'][targetname]['active']
   for j,st in enumerate(trajectory):
    for ti,tr in enumerate(st['trials']):
     assert tr['scale']==1/(2**ti);cb=st['classification_before']-1e-4*tr['scale']*.1*max(0,st['classification_direction_dot']);tb=st['total_before']-1e-4*tr['scale']*.1*max(0,st['total_direction_dot']);assert abs(cb-tr['candidate_bound'])<1e-10 and abs(tb-tr['total_bound'])<1e-10;assert tr['accepted']==(tr['classification']<=cb+1e-10 and (variant=='candidate_only' or tr['total']<=tb+1e-10))
    if st['accepted']:accepted+=1;ups+=int(st['classification_after']>st['classification_before']+1e-10);assert variant=='candidate_only' or st['total_after']<=st['total_before']+1e-10
    if not active:assert st['classification_before'] is None
    k=j+1
    if k not in [1,3]:continue
    if k==1 and active:
     restore(m,S);bv=objective86(m,base,z,d,s0,pi,baselines,pm,im,ws)
     for actual,key in [(bv[0],'total_before'),(bv[2],'classification_before'),(bv[3],'recovery_before')]:losserror=max(losserror,abs(float(actual)-st[key]));losschecks+=1
    arm='candidate_only1' if variant=='candidate_only' else variant+'_tta'+str(k);statefile=P/'states_PRIVATE'/(sid+'_'+arm+'.npz');assert sha(statefile)==seal0['files'][statefile.relative_to(P).as_posix()];put_patch(m,load(statefile));pcur=m.main(base,z,d,'bidirectional')[0];pred=anchored85(pcur,frozen,s0).cpu().numpy();worst=max(worst,float(abs(pred-a['scores'][ARMS87.index(arm)]).max()));endpoints+=1
    if active:
     vals=objective86(m,base,z,d,s0,pi,baselines,pm,im,ws)
     for actual,key in [(vals[0],'total_after'),(vals[2],'classification_after'),(vals[3],'recovery_after')]:losserror=max(losserror,abs(float(actual)-st[key]));losschecks+=1
    restore(m,S)
  assert worst<1e-10 and losserror<1e-10
  if (nr+1)%64==0:print('AUDIT87',nr+1,flush=True)
assert ups==0 and accepted==read(P/'COUNTS.json')['accepted_updates'];O=H87/'independent_structural_native_audit_attempt001';O.mkdir();result={'state':'PASS_R87_ALL336_SOFT_ALIAS_JOINT_TARGET_NATIVE_READOUT_LOSS_AND_PRIORITY_AUDIT','n':336,'arm_names':ARMS87,'saved_endpoint_replays':endpoints,'actual_loss_scalar_checks':losschecks,'accepted_updates':accepted,'accepted_candidate_increases':ups,'max_probability_or_score_error':worst,'max_target_error':targeterror,'max_actual_loss_error':losserror,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start,'new_task_scores':0};dump(O/'AUDIT_RESULT.json',result);print(result)

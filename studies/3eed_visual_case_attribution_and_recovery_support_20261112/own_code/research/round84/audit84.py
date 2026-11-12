"""Independent selected-state, original-feature bridge and map-scalar replay."""
from core84 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H84/'experiment/DESIGN_LOCK.json');pin(p);P=B84/'capture33_attempt001';seal0=read(P/'SEALED.json')
for f,h in seal0['files'].items():assert sha(P/f)==h
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(m,S);start=time.perf_counter();maxmap=maxnative=maxh=0.;checks=0
for c in read(H84/'experiment/CASES33_PRIVATE.json'):
 r=c['row'];sid=r['id'];z0,z,d=item80(r);a=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/(c['case']+'_PRIVATE.json'));old=load(B82/'prediction2640_attempt001/samples_PRIVATE'/(sid+'.npz'));refs=dict(zip(old['arm_names'],old['scores']));oldrec=read(B82/'prediction2640_attempt001/records_PRIVATE'/(sid+'.json'))
 with torch.no_grad():
  h,u=original84(base,z);f=prefixpath(r) if sid in DEV_IDS else ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz')
  with np.load(f,allow_pickle=False) as data:expected=data['A_h'].copy()
  maxh=max(maxh,float(abs(h.cpu().numpy()-expected).max()));assert maxh<1e-6;combos={'original_features_original_head':score84(base.contrastive_align_projection_image,h,u,z['token_mask'].bool()).cpu().numpy(),'original_features_prepared_head':score84(m.match,h,u,z['token_mask'].bool()).cpu().numpy()}
  pm,im=maskpair79(r,d);hp,hi,hw=holdout81(r,d);views=list(zip(pm,im))+list(zip(hp,hi));ws=uniform_weights80(d,pm,im)+hw
  for state in STATES84:
   if state.startswith('raw'):put_patch(m,rawpatch82(r,state=='raw_wrong'))
   elif state.startswith('BT'):put_patch(m,load(B82/'prediction2640_attempt001/states_PRIVATE'/(sid+('_bt_wrong.npz' if state=='BT_wrong' else '_bt_correct.npz'))))
   pp,q,e=m.main(base,z,d,'bidirectional');maxnative=max(maxnative,float(abs(pp.cpu().numpy()-refs[REF84[state]]).max()));assert maxnative==0.
   if state=='frozen':combos.update(prepared_features_original_head=score84(base.contrastive_align_projection_image,e['query_hidden'],e['projected_text'],z['token_mask'].bool(),e['late_RGB_delta']).cpu().numpy(),prepared_features_prepared_head=pp.cpu().numpy())
   for vi,((pm0,im0),w) in enumerate(zip(views,ws)):
    lp,li,ld=m.local_losses(z,d,pm0,im0,False)
    for key,v in [('P',.1*w['P']*lp),('RGB',.1*w['I']*li),('D',.01*w['D']*ld)]:maxmap=max(maxmap,float(abs(v.cpu().numpy()-a[state+'_'+key+'_maps'][vi]).max()));assert maxmap==0.;checks+=1
   if state in ['frozen','raw','BT']:
    fit=float(np.mean(a[state+'_losses'][:2]));held=float(a[state+'_losses'][2]);ref=oldrec['bt_correct'];expectedfit=ref['objective_before'] if state=='frozen' else (ref['raw_record']['auxiliary_after'] if state=='raw' else ref['auxiliary_after']);expectedheld=ref['heldout_true_auxiliary_before'] if state=='frozen' else ref['raw_heldout_true_auxiliary_after'] if state=='raw' else ref['heldout_true_auxiliary_after'];assert abs(fit-expectedfit)<1e-7 and abs(held-expectedheld)<1e-7
   restore(m,S)
  for k,v in combos.items():assert int(v.argmax())==rec['hybrid_PRIVATE'][k]['query_PRIVATE'] and float(a['candidate_ious'][v.argmax()])==rec['hybrid_PRIVATE'][k]['IoU']
 assert np.array_equal(a['boxes'],old['source_boxes']) and rec['chosen_PRIVATE']['oracle']==int(a['candidate_ious'].argmax());assert all(rec['chosen_PRIVATE'][k]==int(refs[v].argmax()) for k,v in REF84.items());assert np.isfinite(a['source_attention']).all() and np.max(abs(a['source_attention'].sum(-1)-1))<1e-5
assert fingerprint(base)==fp and all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());O=H84/'independent_native_audit_attempt001';O.mkdir();result={'state':'PASS_R84_ALL_SELECTED_STATES_ORIGINAL_FEATURE_BRIDGE_LOCAL_LOSS_MAPS_AND_HYBRID_READOUTS','n':33,'weighted_map_checks':checks,'max_native_probability_error':maxnative,'max_map_error':maxmap,'max_original_hidden_bridge_error':maxh,'new_parameter_updates':0,'target_label_use_only_offline_diagnosis':True,'capture_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',result);print(result)

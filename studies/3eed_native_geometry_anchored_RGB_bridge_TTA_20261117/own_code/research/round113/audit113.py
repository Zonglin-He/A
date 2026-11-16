"""Independent CPU saved-state/readout/seal audit before any new GT score."""
from core113 import *
def main():
 torch.set_num_threads(4);matrix=jread(B113/'MATRIX_SEALED.json');assert matrix['n']==768 and matrix['primary']==PRIMARY113 and matrix['arms']==ARMS113 and jread(H113/'ENGINEERING.json')['n']==12;n=alias=ends=states=0;mx=0.;activity=[];ties=[]
 for src in ['waymo','drone','quad']:
  S=load(R113/('crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz' if src=='waymo' else 'crossplatform/round97/'+src+'/SELECTED_SOURCE_ONLY.npz'));head=load(B113/(src+'_SOURCE_READOUT_PRIVATE.npz'));assert all(np.array_equal(v,S[k]) for k,v in head.items());schema=jread(B113/(src+'_THETA_SCHEMA_PRIVATE.json'));names,sizes=schema['names'],schema['sizes'];assert sum(sizes)==102498 and schema['source_module_sha256']==hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(S.items()))).hexdigest();flat=np.concatenate([S[k].reshape(-1) for k in names]);anchor=flat.copy();offset=np.cumsum([0]+sizes);bridge=np.zeros(102498,bool);rgb=bridge.copy()
  for ix,k in enumerate(names):
   sl=slice(offset[ix],offset[ix+1]);bridge[sl]=k=='gate_I' or k.startswith('point_to_image.');rgb[sl]=bridge[sl]|k.startswith('image_adapter.')
   if k in ['point_adapter.up.weight','point_adapter.up.bias','gate_P']:anchor[sl]=0
  assert bridge.sum()==45313 and rgb.sum()==47441
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B113/name;seal=jread(D/'SEALED.json');assert sha(D/'SEALED.json')==matrix['directions'][name]
   for f,h in seal['files'].items():assert sha(D/f)==h,f
   for r in jread(R113/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id'];out=load(D/'samples_PRIVATE'/(sid+'.npz'));prior=load(B112/name/'samples_PRIVATE'/(sid+'.npz'));old=dict(zip(prior['arm_names'],prior['scores']));ps=dict(zip(out['arm_names'],out['scores']));rec=jread(D/'records_PRIVATE'/(sid+'.json'));n+=1;assert np.array_equal(out['source_boxes'],prior['source_boxes']) and np.array_equal(out['queries'],out['scores'].argmax(1));assert set(ps)==set(ARMS113) and np.isfinite(out['scores']).all() and np.allclose(out['scores'].sum(1),1,atol=1e-12)
    assert all(np.array_equal(ps[k],old[k]) for k in REFS113)
    if rec['protected_alias_fallback']:
     alias+=1;assert not rec['fits'] and not rec['TTA_run'] and all(np.array_equal(ps[k],old['prepared_frozen']) for k in NEW113);continue
    assert rec['C_neural_after_all_fits'] and rec['C_boundary_stamp']['before_B_C_arrays'] and rec['evaluation']['all_C_neural_after_all_fits'];assert rec['source_inputs']['B_C_arrays_not_read_before_all_fits'];assert sha(B111/name/'prefix_PRIVATE'/(sid+'.npz'))==rec['source_inputs']['R111_prefix_sha256']
    ff=load(D/'features_PRIVATE'/(sid+'.npz'));ta=load(D/'fits_PRIVATE'/(sid+'.npz'));computed={}
    for arm,desc in rec['cpu_readouts'].items():
     key=desc['feature_key'];tag=(key,desc['no_late'],desc['no_r'])
     if tag not in computed:
      a={k[len(key)+1:]:v for k,v in ff.items() if k.startswith(key+'_')};computed[tag],diag=cpu113(a,head,*tag[1:]);assert max(diag.values())<5e-6
     p=computed[tag][desc['readout']];err=float(abs(p-ps[arm]).max());mx=max(mx,err);assert err<5e-6,(sid,arm,err);ends+=1
     if int(p.argmax())!=int(ps[arm].argmax()):
      gap=float(ps[arm].max()-ps[arm][int(p.argmax())]);assert gap<=2*err+1e-12;ties.append({'direction':name,'id_PRIVATE':sid,'arm':arm,'CPU_probability_error':err,'native_top_gap_at_CPU_winner':gap,'native_choice_preserved':True,'status':'CPU_FP32_UNRESOLVED_NEAR_TIE'})
    assert set(rec['cpu_readouts'])==set(NEW113)
    for setting in SETTINGS113:
     meta=rec['fits'][setting];is_anchor=setting.startswith('anchor');start=anchor if is_anchor else flat;scope=np.ones(102498,bool) if meta['scope']=='all' else (rgb if meta['scope']=='rgb' else bridge);assert meta['scope_parameters']==int(scope.sum()) and meta['full_parent_gradient_computed_parameters']==102498 and meta['no_C_or_fused_features_read'];gg=ta['gradient_'+('anchor_wrong' if meta['wrong_fit_binding'] else ('anchor' if is_anchor else 'full'))];gn=np.linalg.norm(gg.astype(float));assert abs(gn-meta['full_gradient_L2'])<2e-6
     for step in ['raw','bt']:
      st=ta[setting+'_'+step+'_theta'];assert st.shape==start.shape and np.isfinite(st).all();assert np.array_equal(st[~scope],start[~scope]);delta=st.astype(float)-start.astype(float);l2=float(np.linalg.norm(delta));expected=meta['raw_proposal_parameter_L2']*(1 if step=='raw' else (meta['accepted_alpha']/.5 if meta['accepted'] else 0));assert abs(l2-expected)<3e-6;states+=1
      if meta['scope']!='all' and is_anchor:
       for view in ['A','C','fusedABC']:
        lab=setting+'_'+step+'_'+view;ref='frozen_anchor_'+view;assert all(np.array_equal(ff[lab+'_'+k],ff[ref+'_'+k]) for k in ['h','u','r','rs','rmean','inputmean']), (setting,view)
      activity.append({'source':src,'target':tgt,'id_PRIVATE':sid,'setting':setting,'step':step,'scope_parameters':int(scope.sum()),'changed_values':int(np.count_nonzero(delta)),'actual_parameter_L2':l2,'proposal_L2':meta['raw_proposal_parameter_L2'],'accepted':meta['accepted'],'alpha':.5 if step=='raw' else meta['accepted_alpha'],'auxiliary_before':meta['auxiliary_before'],'auxiliary_after':meta['trials'][0 if step=='raw' else -1]['physical'] if step=='raw' or meta['accepted'] else meta['auxiliary_before'],'holdout_mask_recovery':rec['evaluation']['holdout_mask76003_recovery'][setting+'_'+step],'C_available':rec['evaluation']['C_available']})
    assert rec['evaluation']['legacy_R97_raw_bt_query_and_alpha_exact'] and rec['evaluation']['legacy_R97_raw_bt_max_probability_error']<1e-6
   print('AUDIT113',name,n,ends,flush=True)
 jdump(H113/'UPDATE_ACTIVITY_PRIVATE.json',activity);jdump(H113/'CPU_NEAR_TIES_PRIVATE.json',ties);jdump(H113/'NATIVE_AUDIT.json',{'state':'PASS_ALL_SEALS_CPU_SAMEINFO_READOUTS_SCOPE_WRITES_NATIVE_INVARIANCE_LEGACY_REPLAY_C_BOUNDARY','n':n,'arms':89,'CPU_readouts':ends,'actual_parameter_states':states,'max_CPU_probability_error':mx,'CPU_FP32_unresolved_near_ties':len(ties),'near_tie_native_winner_preserved':True,'protected_alias_fallback':alias,'engineering_real_inputs':12,'GT_used':False,'FINAL_access':False});print('AUDIT113_PASS',n,ends,mx,flush=True)
if __name__=='__main__':main()

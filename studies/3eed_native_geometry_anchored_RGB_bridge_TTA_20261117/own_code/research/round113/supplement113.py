"""Missing matched wrong-binding controls for the positive unmasked natural step.
This does not change the failed original R113 primary or its predictions.
"""
from core113 import *
from datetime import datetime,timezone
SUP113=B113/'supplement_wrong_natural_attempt001'
OLD113S=['anchor_bridge_natural_raw_'+v for v in ['A','C','fusedABC']]
NEW113S=[k+'_'+v for k in ['wrong_natural_raw','wrong_natural_sameL2_raw'] for v in ['A','C','fusedABC']]
def fast_single113s(m,z,d,a):
 ids=z['query_points_sample_inds'].long();known=(d['p2i'][ids]>=0).any(-1);_,II,_=m.features(z['vis0'][0],d,'bidirectional');late=torch.tanh(m.gate_late)*m.late(m.point_image(II,d)[ids].float())*known[:,None];h=torch.as_tensor(a['h'][0],device='cuda');u=torch.as_tensor(a['u'][0],device='cuda');mask=torch.as_tensor(a['mask'],device='cuda').bool();r=torch.as_tensor(a['r'],device='cuda');q=F.normalize(m.match(h[None])[0]+late,dim=-1);p=(native_scores(q,u,mask).double()+r.double()).softmax(0);return p,late
def lock():
 p=H113/'experiment/SUPPLEMENT_LOCK.json';assert not p.exists();jdump(p,{'registered_utc':datetime.now(timezone.utc).isoformat(),'original_R113_primary_remains':PRIMARY113,'original_primary_failed':True,'reason':'secondary unmasked natural raw17.383951 beats strongest frozen17.265658 on exposedDEV; fill missing same-optimizer and same-L2 wrong-binding controls before deciding expansion, not introduce a new loss or rescue the originalprimary','new_controls':NEW113S,'old_positive_parity':OLD113S,'data':'same exposed DEV768, all44aliases frozenfallback','gradients':'replay ONLY existing sealed currentA gradient_anchor/anchor_wrong; project bridge45313; -.5clip1 subset gradient. SameL2 wrong direction scaled once to correct natural proposal norm; zero gradient remains unknown/no update. No fresh gradient optimization or GT rule.','scope':'existing originalR113 point_adapter.up andgateP hardzero; onlypoint_to_image/gateI edits','C_boundary':'all three rawstates computed from savedA-onlygradients before loadingR111C/oldR113featurevalues; no C optimization','no_new_source_preparation_or_experts_or_FINAL':True,'no_change_to_original_predictions_or_primary':True,'files':{str(x.relative_to(R113)):sha(x) for x in [H113/'supplement113.py',H113/'core113.py']},'original_matrix_sha256':sha(B113/'MATRIX_SEALED.json')});print('SUP113_LOCK',flush=True)
def main():
 plan=jread(H113/'experiment/SUPPLEMENT_LOCK.json')
 for p,h in plan['files'].items():assert sha(R113/p)==h
 assert sha(B113/'MATRIX_SEALED.json')==plan['original_matrix_sha256'];assert jread(H113/'SCALAR_AUDIT.json')['n']==768;SUP113.mkdir(exist_ok=False);seals={};n=0;mx=0.
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);anchor=initial113(m,S,True)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=SUP113/name
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for ix,r in enumerate(jread(R113/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')):
    sid=r['id'];prior=B113/name;meta=jread(prior/'records_PRIVATE'/(sid+'.json'));out={};features={};record={'protected_alias_fallback':meta['protected_alias_fallback'],'new_wrong_states':{},'old_natural_correct_max_error':0.,'no_new_GT_or_gradient_optimization':True};tick=time.perf_counter()
    if record['protected_alias_fallback']:
     old=load(prior/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));out={k:po['prepared_frozen'].copy() for k in NEW113S}
    else:
     # Fit-state archive contains only A-derived gradients and selected parameters.
     saved=load(prior/'fits_PRIVATE'/(sid+'.npz'));g=torch.as_tensor(saved['gradient_anchor'],device='cuda');gw=torch.as_tensor(saved['gradient_anchor_wrong'],device='cuda');gg=project113(m,g,'bridge');ww=project113(m,gw,'bridge');dc=-.5*torch.clamp(1/(gg.norm()+1e-6),max=1)*gg;dw=-.5*torch.clamp(1/(ww.norm()+1e-6),max=1)*ww;dm=dw*(dc.norm()/dw.norm()) if float(dw.norm())>0 else dw;directions={'correct_parity':dc,'wrong_natural_raw':dw,'wrong_natural_sameL2_raw':dm}
     # Only now read the current prefix and additional observations/hidden states.
     z0,geom,z,_=current113(src,r);d=tensor_data(geom,r);obs=load(B111/name/'observations_PRIVATE'/(sid+'.npz'));oo={v:{k:torch.as_tensor(obs['real_'+v+'_'+k],device='cuda') for k in ['vis','valid']} for v in ['B','C'] if 'real_'+v+'_vis' in obs};zz={'A':z,'C':prefix106(z,[oo['C']],False) if 'C' in oo else z,'fusedABC':prefix106(z,list(oo.values()),True)};old=load(prior/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));ff=load(prior/'features_PRIVATE'/(sid+'.npz'))
     with torch.no_grad():
      for key,delta in directions.items():
       restore(m,anchor);theta_step91(m,anchor,delta);freeze91(m);actual=theta_patch91(m);changed=[k for k,_ in m.tta_parameters() if not np.array_equal(actual[k],anchor[k])];assert all(scope113(k,'bridge') for k in changed)
       for view,vz in zz.items():
        prefix='frozen_anchor_'+view+'_';a={k[len(prefix):]:v for k,v in ff.items() if k.startswith(prefix)};p,late=fast_single113s(m,vz,d,a)
        if key=='correct_parity':
         err=float(abs(p.cpu().numpy()-po['anchor_bridge_natural_raw_'+view]).max());assert err<1e-7 and int(p.argmax())==int(po['anchor_bridge_natural_raw_'+view].argmax());record['old_natural_correct_max_error']=max(record['old_natural_correct_max_error'],err);mx=max(mx,err)
        else:
         arm=key+'_'+view;out[arm]=p.cpu().numpy();features[arm+'_late']=late.cpu().numpy()
       if key!='correct_parity':record['new_wrong_states'][key]={'proposal_parameter_L2':float(delta.norm()),'correct_natural_parameter_L2':float(dc.norm()),'changed_tensors':changed,'scope_parameters':45313,'gradient_status':'known_A_only_wrong_gradient' if float(ww.norm())>0 else 'UNKNOWN_zero_wrong_gradient','C_before_state_selection':False}
    for k in OLD113S:out[k]=po[k].copy()
    arr=np.stack([out[k] for k in OLD113S+NEW113S]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(OLD113S+NEW113S),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1));np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features);record['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),record);restore(m,S);n+=1
    if (ix+1)%32==0:print('SUP113',name,ix+1,flush=True)
   jdump(D/'SEALED.json',{'n':128,'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(SUP113/'MATRIX_SEALED.json',{'n':n,'arms':OLD113S+NEW113S,'new_arms':NEW113S,'directions':seals,'correct_positive_parity_max_error':mx,'GT_used':False,'original_primary_unchanged':True,'FINAL_access':False});print('SUP113_COMPLETE',n,mx,flush=True)
if __name__=='__main__':
 if '--lock' in sys.argv:lock()
 else:main()

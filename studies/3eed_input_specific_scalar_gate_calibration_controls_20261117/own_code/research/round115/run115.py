from core115 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H115/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R115/p)==h,p
 allproposals,summ=controls115();B115.mkdir(exist_ok=True);n=0;eng=[];seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);anchor=initial113(m,S,True)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R115/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');D=B115/('engineering' if a.engineering else 'formal')/name
   if a.engineering:rows=[r for r in rows if (B111/name/'prefix_PRIVATE'/(r['id']+'.npz')).exists()][:2]
   else:assert jread(H115/'ENGINEERING.json')['n']==12
   assert not D.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];meta0=jread(B113/name/'records_PRIVATE'/(sid+'.json'));meta={'protected_alias_fallback':meta0['protected_alias_fallback'],'all_control_proposals_fixed_before_C':True,'new_backward':False};new={};features={}
    if not meta['protected_alias_fallback']:
     # All source-mean/shuffle/dynamic gate deltas were computed before any C arrays.
     choices=allproposals[(name,sid)];z0,geom,z,_=current113(src,r);d=tensor_data(geom,r);zz,proof=heldout113(src,r,z);meta['evaluation']=proof;meta['proposals']=choices
     if a.engineering:restore(m,S);rt,_=role97(base,m,r,z0,geom,z,d);rt=tensor_route89(rt)
     with np.load(B113/name/'features_PRIVATE'/(sid+'.npz')) as ff,torch.no_grad():
      for control in ['dynamic','fixed_source_mean_gate','shuffled_gate']:
       restore(m,anchor);m.gate_I.copy_(torch.as_tensor(anchor['gate_I'],device='cuda')+choices[control]);freeze91(m)
       for view,vz in zz.items():
        aa={k:ff['frozen_anchor_'+view+'_'+k] for k in ['h','u','mask','r']};p,late=fast114(m,vz,d,aa)
        if control=='dynamic':
         with np.load(B114/'formal'/name/'samples_PRIVATE'/(sid+'.npz')) as old:po=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(p.cpu().numpy(),po['native_gate_only_'+view])
        else:
         label=control+'_'+view;new[label]=p.cpu().numpy();features[label+'_late']=late.cpu().numpy()
         if a.engineering and view=='A':
          f=feature91(m,base,vz,d,rt,anchor);pn,_,_=scorer91(m,f,rt,anchor,'R76','full_disc');assert torch.equal(pn,p) and np.array_equal(f['h'].cpu().numpy(),aa['h'][0]);eng.append({'source':src,'target':tgt,'control':control,'actual_native_main_probability_error':0.})
    with np.load(B114/'formal'/name/'samples_PRIVATE'/(sid+'.npz')) as old:po=dict(zip(old['arm_names'],old['scores']));boxes=old['source_boxes'].copy()
    if meta['protected_alias_fallback']:new={k:po['prepared_frozen'].copy() for k in NEW115}
    out={k:po[k].copy() for k in REF115};out.update(new);arr=np.stack([out[k] for k in ARMS115]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS115),scores=arr,queries=arr.argmax(1),source_boxes=boxes)
    if features:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features)
    meta['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),meta);restore(m,S);freeze91(m);n+=1
    if (ix+1)%32==0:print('RUN115',a.engineering,name,ix+1,flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H115/'ENGINEERING.json',{'state':'PASS_ALL_OLD_DYNAMIC_PROBABILITIES_BITEXACT_ACTUAL_NEW_GATE_EDITS','n':n,'new_actual_native_checks':eng,'GT_used':False})
 else:jdump(B115/'MATRIX_SEALED.json',{'n':n,'arms':ARMS115,'new_arms':NEW115,'directions':seals,'primary':PRIMARY115,'source_A_only_calibration':summ,'GT_used':False,'FINAL_access':False})
 print('RUN115_COMPLETE',a.engineering,n,flush=True)
if __name__=='__main__':main()

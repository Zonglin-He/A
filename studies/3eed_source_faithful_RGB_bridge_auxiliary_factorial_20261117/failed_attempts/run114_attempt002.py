from core114 import *
import argparse,datetime
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H114/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R114/p)==h,p
 B114.mkdir(exist_ok=True);n=0;eng=[];seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);anchor=initial113(m,S,True)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R114/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');D=B114/('engineering' if a.engineering else 'formal')/name
   if a.engineering:rows=[r for r in rows if (B111/name/'prefix_PRIVATE'/(r['id']+'.npz')).exists()][:2]
   elif not (H114/'ENGINEERING.json').exists():raise RuntimeError('Engineering required')
   assert not D.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];oldmeta=jread(B113/name/'records_PRIVATE'/(sid+'.json'));meta={'protected_alias_fallback':oldmeta['protected_alias_fallback'],'TTA':not oldmeta['protected_alias_fallback']};features={}
    if not meta['protected_alias_fallback']:
     z0,z,d,dw,pm,im,ws,proof=input114(src,r);dirs,ev=proposals114(m,S,anchor,src,r,z,d,dw,pm,im,ws,a.engineering);new,features,ev2=evaluate114(base,m,S,anchor,src,r,z0,z,d,pm,im,dirs,a.engineering);meta.update(inputs=proof,updates=ev,evaluation=ev2)
    old=load(B113/name/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));out={k:po[k] for k in REF114}
    if meta['protected_alias_fallback']:new={k:po['prepared_frozen'] for k in NEW114}
    out.update(new);arr=np.stack([out[k] for k in ARMS114]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS114),scores=arr,queries=arr.argmax(1),source_boxes=old['source_boxes'])
    if features:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features)
    meta['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),meta);restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items());n+=1
    if a.engineering:eng.append({'source':src,'target':tgt,'actual_native_main_max_error':meta['evaluation']['actual_native_main_max_error'],'source_RGB_keys_zero_late_max_error':meta['evaluation']['source_key_frozen_late_max_error'],'source_aux_gradient_within_1e12_and_bridge_proposal_bitexact_previous':True})
    if (ix+1)%16==0 or a.engineering:print('RUN114',a.engineering,name,ix+1,round(meta['seconds'],2),flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H114/'ENGINEERING.json',{'state':'PASS_ACTUAL_SOURCE_SEPARATED_NATIVE_MAIN_FAST_CPU_READY','n':n,'checks':eng,'GT_used':False})
 else:jdump(B114/'MATRIX_SEALED.json',{'n':n,'arms':ARMS114,'new_arms':NEW114,'primary':PRIMARY114,'directions':seals,'GT_used':False,'FINAL_access':False})
 print('COMPLETE114',a.engineering,n,flush=True)
if __name__=='__main__':main()

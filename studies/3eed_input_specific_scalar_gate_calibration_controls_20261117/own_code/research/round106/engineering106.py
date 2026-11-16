"""Real-input canonical-point interpolation and actual patched native verification."""
from core106 import *
from engineering105 import check_derivative105
def main():
 assert jread(H106/'experiment/DESIGN_LOCK.json')['fixed'];anchors={x['A']['metapath']:x for x in jread(R106/'research/round105/FULL_ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in jread(R106/'research/round105/FULL_PAIRS_PRIVATE.json')};n=0;checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   rows=jread(R106/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');rr=[]
   for r in rows:
    ar=anchors[str(Path(r['metapath']).resolve())];pid=ar['B_pair']
    if pid and jread(R106/'crossplatform/round105/registration_full_PRIVATE'/(pid+'.json'))['geometry_cycle_supported']:rr.append(r)
   for r in rr[:2]:
    restore(m,S)
    with torch.no_grad():
     z0,g,_=capture98(base,r,src);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);A=feature106(m,base,z,d,route,S,[]);old=load(B98/(src+'_to_'+tgt)/'samples_PRIVATE'/(r['id']+'.npz'));err=float(abs(A['p']-torch.as_tensor(dict(zip(old['arm_names'],old['scores']))['prepared_frozen'],device='cuda')).max());assert err<1e-7
     own,valid,pr=interpolation106(z0['xyz'][0],z0['xyz'][0],z['vis0'][0],z0['xyz'][0]);assert valid.all();selferr=float(abs(own-z['vis0'][0]).max());assert selferr==0.;selfv={'vis':own,'valid':valid};same=feature106(m,base,z,d,route,S,[selfv],True);sameerr=float(abs(same['p']-A['p']).max());assert sameerr<1e-7
     ar=anchors[str(Path(r['metapath']).resolve())];B,proof=extra106(base,ar['A'],pairs[ar['B_pair']],z0,'real','B');random,randomproof=extra106(base,ar['A'],pairs[ar['B_pair']],z0,'randomA','B');assert B is not None and random is not None;unknown=~B['valid'];assert torch.equal(B['vis'][unknown],z['vis0'][0,unknown]);fB=feature106(m,base,z,d,route,S,[B]);fAB=feature106(m,base,z,d,route,S,[B],True);cached_error=float(abs(read105(m,fB)-fB['p']).max());assert cached_error<1e-7;deriv=check_derivative105(m,fB);q=(A['p']+fB['p'])/2
    st,trials,info=lm105(m,[A,fB],q.log()[None].repeat(2,1),head)
    with torch.no_grad():
     put94(m,st);m.route=route;m.use_interface=True;p,_,e=m.main(base,prefix106(z,[B]),d,'bidirectional');nativeerr=float(abs(p-read105(m,fB)).max());assert nativeerr<1e-7;assert torch.equal(e['residual'].double(),fB['r']);restore(m,S)
    checks.append({'source':src,'target':tgt,'source_probability_error':err,'self_feature_error':selferr,'self_native_probability_error':sameerr,'native_cached_probability_error':cached_error,'patched_native_probability_error':nativeerr,'unknown_A_feature_unchanged':True,'real_support':{k:v for k,v in proof.items() if not k.endswith('sha256')},'randomA_support':{k:v for k,v in randomproof.items() if not k.endswith('sha256')},'fit':info,**deriv});n+=1;print('ENGINEERING106',src,tgt,n,flush=True)
  ss=state(m);assert fingerprint(base)==fp and all(np.array_equal(ss[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(H106/'ENGINEERING.json',{'state':'PASS_REAL_SEED_ALIGNMENT_SOURCE_AND_SELF_IDENTITY_UNKNOWN_FALLBACK_NATIVE_PATCHED_MODEL_AND_JACOBIAN','n':n,'checks':checks,'GT_read':False,'FINAL_access':False})
if __name__=='__main__':main()

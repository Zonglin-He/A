"""Actual frozen fusedAB extraction and actual updated C/fusedABC native parity."""
from core107 import *
from engineering105 import check_derivative105
def main():
 assert (H107/'experiment/DESIGN_LOCK.json').is_file();anchors={x['A']['metapath']:x for x in jread(R107/'research/round105/FULL_ANCHORS_PRIVATE.json')};pairs={x['pair_PRIVATE']:x for x in jread(R107/'research/round105/FULL_PAIRS_PRIVATE.json')};checks=[];n=0
 for src in ['waymo','drone','quad']:
  setsource107(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;root=B106/name;rows=jread(R107/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');eligible=[]
   for r in rows:
    rec=jread(root/'records_PRIVATE'/(r['id']+'.json'))
    if rec['families'].get('real',{}).get('updated_output_enabled'):eligible.append((r,rec))
   for r,rec in eligible[:2]:
    restore(m,S);ar=anchors[str(Path(r['metapath']).resolve())]
    with np.load(root/'features_PRIVATE'/(r['id']+'.npz')) as old:
     A=cached107(old,'real','A');B=cached107(old,'real','B')
     with torch.no_grad():
      ab,proof,ctx=obtain_AB107(base,m,S,r,ar,pairs,rec);z0,g,z,d,route,obs=ctx;oldout=load(root/'samples_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z0['boxes'],oldout['source_boxes']);liveA=feature106(m,base,z,d,route,S,[]);liveB=feature106(m,base,z,d,route,S,[obs['real']]);ea=max(float(abs(liveA[k]-A[k]).max()) for k in ['h','u','late','r','p']);eb=max(float(abs(liveB[k]-B[k]).max()) for k in ['h','u','late','r','p']);assert ea==0 and eb==0;der=check_derivative105(m,ab['real'])
     states={};infos={}
     for variant in FITS107:
      vv,tt=targets107(A,B,ab['real'],variant);st,tr,info=lm105(m,vv,tt,head);states[variant]=st;infos[variant]=info
     oldst=load(root/'states_PRIVATE'/(r['id']+'_real.npz'));normst,norminfo=normmatch107(head,oldst,states['mean_keep']);states['norm']=normst
     # Independent C neural/cache access is after ALL fits, never part of targets.
     with torch.no_grad():
      Cobs,Cproof=extra106(base,ar['A'],pairs[ar['C_pair']],z0,'real','C');assert Cobs is not None;C=cached107(old,'real','C');ABC=cached107(old,'real','fusedABC');liveC=feature106(m,base,z,d,route,S,[Cobs]);liveABC=feature106(m,base,z,d,route,S,[obs['real'],Cobs],True);assert all(torch.equal(liveC[k],C[k]) and torch.equal(liveABC[k],ABC[k]) for k in ['h','u','late','r','p']);err=0.
      for variant,st in states.items():
       for f,zz in [(C,prefix106(z,[Cobs])),(ABC,prefix106(z,[obs['real'],Cobs],True))]:
        put94(m,st);m.route=route;m.use_interface=True;p,_,e=m.main(base,zz,d,'bidirectional');diff=float(abs(p-read105(m,f)).max());assert diff<1e-7 and torch.equal(e['residual'].double(),f['r']);err=max(err,diff)
     checks.append({'source':src,'target':tgt,'exact_A_feature_error':ea,'exact_B_feature_error':eb,'B_sensor_and_sample_hashes_match':True,'C_and_fusedABC_source_cache_exact':True,'patched_native_C_and_fusedABC_max_error':err,'norm_control':norminfo,'fits':infos,**der});restore(m,S);n+=1;print('ENGINEERING107',src,tgt,n,flush=True)
  final=state(m);assert fingerprint(base)==fp and all(np.array_equal(final[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(H107/'ENGINEERING.json',{'state':'PASS_NATIVE_AB_EXTRACTION_EXACT_R106_A_B_C_FUSEDABC_AND_UPDATED_NATIVE_OUTPUTS_JACOBIAN_NORM_CONTROL','n':n,'checks':checks,'GT_used':False,'FINAL_access':False})
if __name__=='__main__':main()

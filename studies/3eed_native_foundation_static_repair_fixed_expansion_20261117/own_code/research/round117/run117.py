from core117 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H117/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R117/p)==h,p
 for p,h in plan['input_sha256'].items():assert sha(R117/p)==h,p
 anchors={x['A']['metapath']:x for x in jread(H109/'ANCHORS_PRIVATE.json')};pairs={x['pair_PRIVATE']:x for x in jread(H109/'PAIRS_PRIVATE.json')};calib=calibration117();B117.mkdir(exist_ok=True);n=0;seals={};checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(H109/f'input/TARGET_{tgt.upper()}256_PRIVATE.json');D=B117/('engineering' if a.engineering else 'formal')/name
   if a.engineering:
    eligible=[r for r in rows if anchors[str(Path(r['metapath']).resolve())]['A']['eligible_for_extra_media']];available=[r for r in eligible if jread(B109/name/'records_PRIVATE'/(r['id']+'.json'))['families']['real']['C_available']];rows=(available[:1]+[r for r in eligible if r not in available[:1]][:1]);assert len(rows)==2
   else:assert jread(H117/'ENGINEERING.json')['n']==12
   assert not D.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','prefix_PRIVATE','observations_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];prior=jread(B109/name/'records_PRIVATE'/(sid+'.json'));alias=prior['protected_alias_fallback'];meta={'protected_alias_fallback':alias,'TTA':False,'gradient_updates':0,'fixed_calibration_not_fit_on_expansion':True};ff={}
    if not alias:
     z0,z,d,zz,route,proof,obarrays,ev,role=extract117(base,m,S,src,r,anchors,pairs,a.engineering);out,ff,_=evaluate117(base,m,S,src,r,z0,d,zz,route,calib[src]);meta.update(inputs=ev,role_info=role,sensor_support=proof);np.savez_compressed(D/'prefix_PRIVATE'/(sid+'.npz'),**z0);np.savez_compressed(D/'observations_PRIVATE'/(sid+'.npz'),**obarrays)
     if a.engineering:checks.append({'source':src,'target':tgt,'old_prepared_A_F_hidden_bitexact':True,'source_A_native_hidden_text_bitexact':True,'C_available':ev['C_available'],'source_original_probability_error':ev['source_original_probability_error']})
    old=np.load(B109/name/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']))
    if alias:out={k:po['prepared_frozen'].copy() for k in NEW117}
    out={**{k:po[k] for k in REF117},**out};arr=np.stack([out[k] for k in ARMS117]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS117),scores=arr,queries=arr.argmax(1),source_boxes=old['source_boxes']);
    if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff)
    meta['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),meta);restore(m,S);freeze91(m);n+=1
    if a.engineering or (ix+1)%16==0:print('RUN117',a.engineering,name,ix+1,round(meta['seconds'],2),flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H117/'ENGINEERING.json',{'state':'PASS12_REAL_EXPANSION_SOURCE_NATIVE_PREPARED_AND_REGISTERED_SENSOR_PARITY','n':n,'checks':checks,'GT_used':False})
 else:jdump(B117/'MATRIX_SEALED.json',{'n':n,'arms':ARMS117,'new_arms':NEW117,'directions':seals,'primary':PRIMARY117,'TTA':False,'GT_used':False,'FINAL_access':False})
 print('RUN117_COMPLETE',a.engineering,n,flush=True)
if __name__=='__main__':main()

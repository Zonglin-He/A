from core118 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');a=p.parse_args()
def main():
 lock=rd(H/'experiment/DESIGN_LOCK.json')
 for f,h in lock['files'].items():assert sha(R/f)==h
 if not a.engineering:assert rd(H/'ENGINEERING.json')['n']==24
 total=aliases=0;seals={};proofs=[]
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   W=head(src)
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;D=B/('engineering' if a.engineering else 'formal')/co/name;assert not D.exists();rr=rows(co,tgt)
    if a.engineering:rr=[r for r in rr if not rd(olddir(co,name)/'records_PRIVATE'/(r['id']+'.json'))['protected_alias_fallback']][:2]
    assert len(rr)==(2 if a.engineering else (128 if co=='DEV' else 256))
    for ix,r in enumerate(rr):
     tick=time.perf_counter();sid=r['id'];oldmeta=rd(olddir(co,name)/'records_PRIVATE'/(sid+'.json'));alias=oldmeta['protected_alias_fallback'];aa=np.load(olddir(co,name)/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(aa['arm_names'],aa['scores']));refs={k:po[v] for k,v in REFMAP[co].items()};meta={'protected_alias_fallback':alias,'TTA':False,'parameter_updates':0,'cohort':co,'C_available':oldmeta.get('inputs',{}).get('C_available',oldmeta.get('evaluation',{}).get('C_available',False))}
     if alias:out={k:refs['prepared_A'].copy() for k in NEW};aliases+=1
     else:
      N,S,files=components(co,name,sid);out,ck,perm=predictions(co,name,sid,N,S,W);errs={k:float(abs(ck[k]-refs[{'prepared_A':'prepared_A','prepared_fusedABC':'prepared_F','native_A':'original_source','native_F':'native_F'}[k]]).max()) for k in ck};errs['repaired_F']=float(abs(out['F_lNF_rNF']-refs['repaired_F']).max());errs['lateOnly_F']=float(abs(out['F_lNF_rzero']-refs['lateOnly_F']).max());assert max(errs.values())<5e-6,(co,name,errs);meta.update(component_sha256=files,old_provider_probability_errors=errs,query_derangement_sha256=hashlib.sha256(perm.tobytes()).hexdigest());
      if a.engineering:proofs.append({'cohort':co,'source':src,'target':tgt,'errors':errs,'GT_used':False})
     out=refs|out;arr=np.stack([out[k] for k in ARMS]);assert arr.shape==(25,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);dst=D/'samples_PRIVATE'/(sid+'.npz');dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,arm_names=np.array(ARMS),scores=arr,queries=arr.argmax(1),source_boxes=aa['source_boxes']);meta['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),meta);total+=1
     if a.engineering or (ix+1)%32==0:print('RUN118',a.engineering,co,name,ix+1,round(time.perf_counter()-tick,3),flush=True)
    dump(D/'SEALED.json',{'n':len(rr),'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file()},'GT_used':False});seals[co+'/'+name]=sha(D/'SEALED.json')
 if a.engineering:dump(H/'ENGINEERING.json',{'state':'PASS24_REAL_PROVIDER_IDENTITY_ANCHORS_AND_NATIVE_READOUT','n':total,'checks':proofs,'GT_used':False})
 else:assert total==2304 and aliases==154;dump(B/'MATRIX_SEALED.json',{'n':total,'aliases':aliases,'arms':ARMS,'new_arms':NEW,'primary':PRIMARY,'directions':seals,'GT_used':False,'FINAL_access':False,'TTA':False})
 print('RUN118_COMPLETE',a.engineering,total,aliases,flush=True)
if __name__=='__main__':main()

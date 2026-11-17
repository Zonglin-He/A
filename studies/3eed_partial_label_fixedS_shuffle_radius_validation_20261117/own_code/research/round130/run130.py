from core130 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 reg=jread(R130/'research/round122/CACHE_REGISTRY_PRIVATE.json');rows=jread(H130/'CONTROL_ROWS_PRIVATE.json');by={(x['cohort'],x['source'],x['target'],x['id_PRIVATE']):x for x in rows};seals={};n=0;checks=[]
 for src in ['waymo','drone','quad']:
  base=base97(src);finger=fingerprint(base);m,S=model97(base,src)
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if tgt==src:continue
    name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt and e['cohort']==co]
    if args.engineering:
     active=[e for e in rr if by[(co,src,tgt,e['row']['id'])]['effective_shuffle']];off=[e for e in rr if not e['protected_alias_fallback'] and not by[(co,src,tgt,e['row']['id'])]['partial_eligible']];rr=[active[0],off[0]]
    D=B130/('engineering' if args.engineering else 'formal')/co/name
    for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
    for i,e in enumerate(rr):
     sid=e['row']['id'];meta=by[(co,src,tgt,sid)];tick=time.perf_counter();ff={};states={}
     if meta['partial_eligible']:new,ff,states,rec=process130(base,m,S,e,args.engineering)
     else:new={};rec={'no_partial_supervision_no_update':True,'traces':{},'C_used_for_fit_or_gate':False}
     # No C output is decoded before this input's updates are fixed.
     old=np.load(R130/'crossplatform/round129/formal'/co/name/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));refs={k:po[k].copy() for k in REF130 if not k.startswith('R129_')}
     for v in ['F','C']:refs['R129_guard_'+v]=po['partial_guard_'+v].copy()
     if not meta['partial_eligible']:
      for k in NEW130:new[k]=po['zero_'+k.rsplit('_',1)[1]].copy()
     assert set(new)==set(NEW130);p=refs|new;a=np.stack([p[k] for k in ARMS130]);assert a.shape==(23,256) and np.isfinite(a).all() and np.allclose(a.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS130),scores=a,queries=a.argmax(1),source_boxes=old['source_boxes']);old.close()
     if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff)
     if states:np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states)
     rec.update(control=meta,seconds=time.perf_counter()-tick);jdump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
     if args.engineering:checks.append({'cohort':co,'source':src,'target':tgt,'active':meta['partial_eligible'],'checks':rec.get('engineering_checks',[]),'seconds':rec['seconds']})
     if args.engineering or i%64==0:print('RUN130',args.engineering,co,name,i+1,round(rec['seconds'],3),flush=True)
    jdump(D/'SEALED.json',{'n':len(rr),'new_GT_used':False,'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='SEALED.json'}});seals[co+'/'+name]=sha(D/'SEALED.json')
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if args.engineering:jdump(H130/'ENGINEERING.json',{'status':'PASS_REAL_FIXEDS_PARTIAL_SHUFFLE_STEPS_ORIGINAL_REPLAY','n':n,'checks':checks})
 else:assert n==2304;jdump(B130/'MATRIX_SEALED.json',{'n':n,'arms':ARMS130,'primary':'partial_F','new_arms':NEW130,'scales':[.5,1.,2.],'seals':seals,'FINAL_access':False,'new_GT_used':False})
 print('COMPLETE130',args.engineering,n,flush=True)
if __name__=='__main__':B130.mkdir(exist_ok=True);main()

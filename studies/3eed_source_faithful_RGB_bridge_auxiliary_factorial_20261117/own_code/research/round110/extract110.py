"""Only missing original-source query states; no old method refits or GT."""
from core110 import *
def main():
 lock=jread(H110/'experiment/STAGE0_CODE_LOCK.json')
 for p,h in lock['files'].items():assert sha(R110/p)==h
 O=B110/'original_PRIVATE';O.mkdir(exist_ok=False,parents=True);n=aliases=0;seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=O/name;D.mkdir();rows=jread(R110/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')
   for r in rows:
    sid=r['id'];record=jread(B106/name/'records_PRIVATE'/(sid+'.json'))
    if record['protected_alias_fallback']:aliases+=1;continue
    tick=time.perf_counter();z,proof=captureO110(base,r,src);proof['seconds']=time.perf_counter()-tick;np.savez_compressed(D/(sid+'.npz'),**z);jdump(D/(sid+'.json'),proof);n+=1
   jdump(D/'SEALED.json',{'n':len(list(D.glob('*.npz'))),'files':{p.name:sha(p) for p in sorted(D.glob('*')) if p.is_file()},'source':src,'target':tgt,'no_GT':True});seals[name]=sha(D/'SEALED.json');print('EXTRACT110',name,n,aliases,flush=True)
  assert fingerprint(base)==fp;del base;torch.cuda.empty_cache()
 jdump(O/'SEALED.json',{'n':n,'protected_alias_fallback':aliases,'directions':seals,'missing_state_extraction_only':True,'no_GT':True,'FINAL_access':False});print('EXTRACT110_COMPLETE',n,aliases)
if __name__=='__main__':main()

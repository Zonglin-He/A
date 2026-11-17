"""Frozen source-own A/B prediction collection, source labels never opened here."""
from core132 import *
def main():
 basecheck();reg=rd(H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json');pairs={p['pair_PRIVATE']:p for p in rd(H132/'SOURCE_PAIRS_PRIVATE.json')};assert not (B132/'source_calibration_PRIVATE').exists();seals={};n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);D=B132/'source_calibration_PRIVATE'/src;D.mkdir(parents=True)
  for i,e in enumerate([e for e in reg if e['entry']['source']==src]):
   for key in ['prefix','geometry']:a=e['entry']['files'][key];assert sha(R132/a['path'])==a['sha256']
   tick=time.perf_counter();ff,rec=source132(base,m,S,e,pairs);sid=e['entry']['row']['id'];np.savez_compressed(D/(sid+'.npz'),**ff);rec['seconds']=time.perf_counter()-tick;dump(D/(sid+'.json'),rec);n+=1
   if i%16==0:print('SOURCE132',src,i+1,round(rec['seconds'],2),flush=True)
  assert fingerprint(base)==fp;dump(D/'SEALED.json',{'files':{p.name:sha(p) for p in D.glob('*') if p.is_file()},'task_GT_read':False});seals[src]=sha(D/'SEALED.json');del base,m;torch.cuda.empty_cache()
 assert n==len(reg);dump(B132/'SOURCE_PREDICTIONS_SEALED.json',{'n':n,'seals':seals,'source_labels_read':False,'no_source_training':True,'FINAL_access':False});print('SOURCE132_COMPLETE',n,flush=True)
if __name__=='__main__':main()

"""Additional strong same-observation direct readouts, registered before any task scoring.
Canonical spatial anchors are shared, not independently predicted query IDs.
This is projected-feature averaging, NOT the historical R59 soft-transport algorithm.
"""
from pathlib import Path
import sys,json,numpy as np,torch,torch.nn.functional as F,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round122';B=R/'crossplatform/round122';sys.path.insert(0,str(R/'research/round118'));import core118 as C
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);O=B/'headmean_controls';assert not O.exists();O.mkdir();seals={};n=0
 for src in ['waymo','drone','quad']:
  W=C.head(src)
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if tgt==src:continue
    name=src+'_to_'+tgt;D=O/co/name;D.mkdir(parents=True)
    for e in [x for x in rd(H/'CACHE_REGISTRY_PRIVATE.json') if x['cohort']==co and x['source']==src and x['target']==tgt]:
     sid=e['row']['id']
     if e['protected_alias_fallback']:
      a=np.load(R/f'crossplatform/round118/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');p=dict(zip(a['arm_names'],a['scores']))['prepared_A'];arr=np.stack([p,p])
     else:
      N,S,_=C.components(co,name,sid);raw=torch.stack([C.project(N[v]['h'],W) for v in ['A','B','C']]).mean(0);U=F.normalize(torch.stack([torch.from_numpy(N[v]['u']) for v in ['A','B','C']]).mean(0),dim=-1).numpy();mask=N['A']['mask'];arr=np.stack([C.read(raw,U,mask,np.zeros((256,64),np.float32),np.zeros(256,np.float32)),C.read(raw,U,mask,S['A']['late'],S['fusedABC']['r'])])
     np.savez_compressed(D/(sid+'.npz'),arm_names=np.array(['headmean_native','headmean_context']),scores=arr,queries=arr.argmax(1));n+=1
    (D/'SEALED.json').write_text(json.dumps({'n':sum(1 for _ in D.glob('*.npz')),'GT_used':False,'files':{f.name:sha(f) for f in D.glob('*.npz')}},indent=2)+'\n');seals[co+'/'+name]=sha(D/'SEALED.json');print('HEADMEAN122',co,name,flush=True)
 assert n==2304;(O/'MATRIX_SEALED.json').write_text(json.dumps({'n':2304,'arms':['headmean_native','headmean_context'],'directions':seals,'GT_used':False,'FINAL_access':False},indent=2)+'\n')
if __name__=='__main__':main()

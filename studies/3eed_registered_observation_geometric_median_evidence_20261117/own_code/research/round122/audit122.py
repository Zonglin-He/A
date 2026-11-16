"""CPU convex-optimality, correspondence/weight controls and native-head audit."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round122';B=R/'crossplatform/round122';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round118'));import core118 as C
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
VARS=['robust','weight_shuffle','weight_mean','wrong_robust','wrong_mean'];REF={'original_source':'original_source','prepared_A':'prepared_A','prepared_F':'prepared_F','native_F':'native_F','repaired_F':'repaired_F','lateOnly_F':'lateOnly_F','mean_context':'F_lSA_rSF','native_ABC':'ABC_native','context_ABC':'ABC_lSA_rSA','context_noR_ABC':'ABC_lSA_rzero'}
def kkt(X,w):
 z=(w[:,:,None]*X).sum(0);dist=np.linalg.norm(z[None]-X,axis=-1);zero=dist<=1e-9;g=((z[None]-X)/np.maximum(dist[:,:,None],1e-30)*(~zero)[:,:,None]).sum(0);res=np.maximum(0,np.linalg.norm(g,axis=-1)-zero.sum(0));old=np.linalg.norm(X-X.mean(0)[None],axis=-1).sum(0);new=dist.sum(0);assert np.max(new-old)<1e-8;assert res.max()<1e-5,float(res.max());return float(res.max()),len(z)
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==2304 and rd(H/'ENGINEERING.json')['n']==24;plan=rd(H/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R/p)==h,p
 n=aliases=heads=seeds=triple_inputs=ties=0;maxp=maxkkt=0.
 for key,ss in mat['directions'].items():
  co,name=key.split('/');src,_,tgt=name.split('_');D=B/'formal'/co/name;assert sha(D/'SEALED.json')==ss
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  head=dict(np.load(B/(src+'_MATCH_PRIVATE.npz')))
  for e in [x for x in rd(H/'CACHE_REGISTRY_PRIVATE.json') if x['cohort']==co and x['source']==src and x['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));old=np.load(R/f'crossplatform/round118/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');po=dict(zip(a['arm_names'],a['scores']));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   for k,v in REF.items():assert np.array_equal(po[k],op[v])
   if e['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[v+'_'+rdout],op['prepared_A']) for v in VARS for rdout in ['native','context']);continue
   rec=rd(D/'records_PRIVATE'/(sid+'.json'));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));z=np.load(R/e['prefix']);ob=np.load(R/e['observations']);X=[z['vis0'][0]];valid=[np.ones(1024,bool)]
   for view in ['B','C']:
    pre=('real_' if co=='DEV' else '')+view+'_'
    if pre+'vis' in ob:X.append(ob[pre+'vis']);valid.append(ob[pre+'valid'])
   X=np.stack(X).astype(float);valid=np.stack(valid);tri=valid.sum(0)==3;assert np.array_equal(tri,ff['triple']);assert int(tri.sum())==rec['three_valid_seeds'];w=ff['robust_weights'];assert np.allclose(w.sum(0),1,atol=1e-12) and np.all(w>=0);uniform=valid/valid.sum(0)[None];assert np.array_equal(w[:,~tri],uniform[:,~tri])
   if tri.any():
    er,nn=kkt(X[:,tri],w[:,tri]);maxkkt=max(maxkkt,er);seeds+=nn;triple_inputs+=1;ids=np.flatnonzero(tri);perm=ff['weight_permutation'];assert np.array_equal(np.sort(perm),np.arange(len(ids)));assert (perm!=np.arange(len(ids))).all() or len(ids)==1;assert np.array_equal(ff['weight_shuffle'][:,ids],w[:,ids[perm]]);assert np.allclose(ff['weight_mean'][:,ids],w[:,ids].mean(1)[:,None],atol=1e-15);WX=X.copy()
    for j in range(1,3):
     p=ff['wrong_permutation_'+str(j-1)];assert np.array_equal(np.sort(p),np.arange(len(ids)));assert (p!=np.arange(len(ids))).all() or len(ids)==1;WX[j,ids]=X[j,ids[p]]
    er,nn=kkt(WX[:,tri],ff['wrong_robust_weights'][:,tri]);maxkkt=max(maxkkt,er);seeds+=nn
   N,S,hashes=C.components(co,name,sid);assert hashes==rec['provider_sha256'];assert np.array_equal(ff['fixed_late'],S['A']['late']) and np.array_equal(ff['fixed_residual'],S['fusedABC']['r']);assert np.array_equal(ff['mask'],N['fusedABC']['mask'])
   for v in VARS:
    for read in ['native','context']:
     pp=cpu112(ff[v+'_h'],ff[v+'_u'],ff['mask'],ff['fixed_late'] if read=='context' else np.zeros((256,64),np.float32),ff['fixed_residual'] if read=='context' else np.zeros(256,np.float32),head);want=po[v+'_'+read];er=float(abs(pp-want).max());maxp=max(maxp,er);assert er<5e-6
     if pp.argmax()!=want.argmax():assert want.max()-want[pp.argmax()]<=2*er+1e-12;ties+=1
     if not tri.any():assert np.array_equal(ff[v+'_h'],N['fusedABC']['h']) and np.array_equal(ff[v+'_u'],N['fusedABC']['u'])
     heads+=1
   if n%64==0:print('AUDIT122',co,name,n,flush=True)
 assert n==2304 and aliases==154 and heads==21500;out={'state':'PASS_ALL_FROZEN_HEADS_EXACT_MEDIAN_CONVEX_KKT_WEIGHT_AND_CORRESPONDENCE_CONTROLS','n':n,'aliases':aliases,'CPU_heads':heads,'triple_supported_inputs':triple_inputs,'point_triplet_KKT_checks':seeds,'max_KKT_violation':maxkkt,'max_probability_error':maxp,'CPU_near_ties':ties,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

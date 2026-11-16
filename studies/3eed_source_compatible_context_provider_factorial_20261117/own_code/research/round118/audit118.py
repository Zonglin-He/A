"""Independent NumPy MLP, normalization, logmass and aggregation replay.
Uses cache loader/schema only; never imports the production prediction function.
"""
from core118 import R,H,B,rd,sha,dump,components,REFMAP,rows
import numpy as np

def soft(x):
 e=np.exp(x-x.max());return e/e.sum()
def lse(x):
 m=x.max(-1);return m+np.log(np.exp(x-m[:,None]).sum(-1))
def read(z,late,res,S):
 x=z['h']
 for k in ['0','2']:x=np.maximum(x@S[k+'.weight'].T+S[k+'.bias'],0)
 x=x@S['4.weight'].T+S['4.bias']+late;x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12);score=(x@z['u'].T)/.07;m=z['mask'].astype(bool);score=lse(score[:,m])-lse(score[:,~m]);return soft(score.astype(float)+res.astype(float))
def main():
 mat=rd(B/'MATRIX_SEALED.json');plan=rd(H/'experiment/DESIGN_LOCK.json');assert mat['n']==2304 and len(mat['new_arms'])==19
 for p,h in plan['files'].items():assert sha(R/p)==h
 for p,h in plan['input_seals'].items():assert sha(R/p)==h
 n=aliases=heads=ties=0;mx=0.;reg=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules']
 for combo,se in mat['directions'].items():
  co,name=combo.split('/');src,_,tgt=name.split('_');D=B/'formal'/co/name;assert sha(D/'SEALED.json')==se
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  with np.load(R/reg[src]['state_path']) as zz:W={k[6:]:zz[k] for k in zz.files if k.startswith('match.')}
  for ix,r in enumerate(rows(co,tgt)):
   sid=r['id'];meta=rd(D/'records_PRIVATE'/(sid+'.json'));a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(a['arm_names'],a['scores']));old=np.load((R/('crossplatform/round113' if co=='DEV' else 'crossplatform/round117/formal')/name/'samples_PRIVATE'/(sid+'.npz')));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   for k,v in REFMAP[co].items():assert np.array_equal(po[k],op[v])
   if meta['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[k],po['prepared_A']) for k in mat['new_arms']);continue
   N,S,files=components(co,name,sid);assert files==meta['component_sha256'];assert meta['parameter_updates']==0 and not meta['TTA'];F=N['fusedABC'];L={'NF':F['late'],'SF':S['fusedABC']['late'],'SA':S['A']['late']};V={'zero':np.zeros(256,np.float32),'NF':F['r'],'SF':S['fusedABC']['r'],'SA':S['A']['r']};seed=int(__import__('hashlib').sha256(('R118|'+co+'|'+name+'|'+sid).encode()).hexdigest()[:16],16);order=np.roll(np.random.default_rng(seed).permutation(256),1);perm=np.empty(256,int);perm[order]=np.roll(order,1);assert np.all(perm!=np.arange(256));L['SAshuffled']=L['SA'][perm];V['SAshuffled']=V['SA'][perm]
   for arm in mat['new_arms']:
    if arm=='ABC_native':p=np.mean([read(N[v],L['SA']*0,V['zero'],W) for v in ['A','B','C']],axis=0)
    else:
     view,lt,rs=arm.split('_');lt=lt[1:];rs=rs[1:]
     if view=='ABC':p=np.mean([read(N[v],L[lt],V[rs],W) for v in ['A','B','C']],axis=0)
     else:p=read(N['fusedABC' if view=='F' else view],L[lt],V[rs],W)
    er=float(abs(p-po[arm]).max());mx=max(mx,er);assert er<5e-6,(combo,sid,arm,er)
    if p.argmax()!=po[arm].argmax():assert po[arm].max()-po[arm][p.argmax()]<=2*er+1e-12;ties+=1
    heads+=1
   if (ix+1)%64==0:print('AUDIT118',combo,ix+1,flush=True)
 assert n==2304 and aliases==154 and heads==(2304-154)*19
 dump(H/'NATIVE_AUDIT.json',{'state':'PASS_INDEPENDENT_NUMPY_ALL19_NEW_READOUTS_AND_ANCHOR_SEALS','n':n,'aliases':aliases,'new_CPU_endpoints':heads,'max_probability_error':mx,'CPU_near_ties':ties,'GT_used':False,'FINAL_access':False});print('AUDIT118_COMPLETE',n,heads,mx,flush=True)
if __name__=='__main__':main()

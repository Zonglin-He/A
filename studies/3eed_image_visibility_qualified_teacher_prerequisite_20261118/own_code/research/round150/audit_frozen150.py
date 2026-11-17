"""Independent group support, routed teacher and frozen-control reconstruction; no GT."""
from common150 import *
import numpy as np

def main():
 check()
 for f,h in rd(H/'experiment/FROZEN_AUDIT_CODE_LOCK.json')['files'].items():assert sha(R/f)==h
 root=B/'frozen';seal=rd(root/'SEALED.json');es=rd(B/'VISIBILITY_SEALED.json');reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');n=0;nb=0;mx=0.
 for f,h in seal['files'].items():assert sha(root/f)==h
 for f,h in es['files'].items():assert sha(B/f)==h
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(root/name/(sid+'.npz'));ev=np.load(B/'visibility_PRIVATE'/name/(sid+'.npz'));old=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));ps=dict(zip(a['arm_names'],a['scores']));orig=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1)) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
  for k in ['assignment','S','C','pA','pB','eligible']:assert np.array_equal(ev[k],orig[k])
  ass=ev['assignment'];gs=[np.where(ass==g)[0] for g in range(int(ass.max()+1))]
  if bool(ev['eligible']):
   nb+=1;z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));im=np.load(R/'crossplatform/round145/image'/name/'banks_PRIVATE'/(sid+'.npz'));xyz=z['xyz'][0];_,inv=np.unique(xyz,axis=0,return_inverse=True);nA=[];nB=[];nn=[]
   for group in gs:
    inside=np.logical_or.reduce([(np.abs(xyz-z['boxes'][q,:3])<=np.maximum(z['boxes'][q,3:6],1e-6)/2).all(-1) for q in group]);nn.append(len(set(inv[inside].tolist())));nA.append(len(set(inv[inside&im['norm_Aimage_visible']].tolist())));nB.append(len(set(inv[inside&im['norm_Bimage_visible']].tolist())))
   nA,nB=np.array(nA),np.array(nB);assert np.array_equal(nA,ev['nA']) and np.array_equal(nB,ev['nB']) and np.array_equal(nn,ev['n_total']);Aok=nA>=4;Bok=nB>=4;scope=ev['S']&(Aok|Bok);assert np.array_equal(scope,ev['S_rgb']);ae=np.ones(len(gs));ac=np.ones(len(gs));has=Aok|Bok;ae[has]=Aok[has]/(Aok[has].astype(float)+Bok[has]);numsA=np.where(Aok,nA,0);numsB=np.where(Bok,nB,0);ac[has]=numsA[has]/(numsA[has]+numsB[has]);assert np.array_equal(ae,ev['alpha_equal']) and np.array_equal(ac,ev['alpha_count']);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R150_visibility_route').encode()).hexdigest()[:16],16));perm=np.arange(len(gs))
   for size in sorted({len(gs[g]) for g in np.flatnonzero(scope)}):
    ids=np.array([g for g in np.flatnonzero(scope) if len(gs[g])==size],int);order=rng.permutation(ids)
    if len(order)>1:perm[order]=np.roll(order,1)
   assert np.array_equal(perm,ev['permutation']);assert np.array_equal(scope[perm],scope);ga=ae[scope].mean() if scope.any() else 1.;gc=numsA[scope].sum()/max((numsA[scope]+numsB[scope]).sum(),1)
   for suf,alp in [('equal',ae),('count',ac),('wrong_equal',ae[perm]),('wrong_count',ac[perm]),('global_equal',np.full(len(gs),ga)),('global_count',np.full(len(gs),gc))]:
    qscope=scope[ass];weights=alp[ass];p=orig['pA'].copy();p[qscope]=(weights*orig['pA']+(1-weights)*orig['pB'])[qscope];p/=p.sum();err=float(abs(p-ps['vis_'+suf]).max());mx=max(mx,err);assert err<1e-15
   for suf in ['equal','wrong_equal','global_equal','count']:
    p=ps['vis_'+suf];qc=ev['C'][ass];q=p*qc;q/=q.sum();assert np.array_equal(q,ps['vis_'+suf+'_Cteacher']);tar=np.array([p[g].max() for g in gs])[ev['C']];tar/=tar.sum();q0=op['norm_joint_zero'];cids=np.flatnonzero(ev['C']);groupmean=np.array([q0[g].mean() for g in gs]);weight=tar*np.array([q0[gs[g]].sum()/max(q0[gs[g]].max(),1e-300) for g in cids]);weight/=weight.sum();out=q0.copy();inS=ev['S'][ass];mass=out[inS].sum();out[inS]=0
    # Match R136 direct: preserve within-group relative query scores and set weighted group mass.
    for g,w in zip(cids,weight):out[gs[g]]=q0[gs[g]]/max(q0[gs[g]].sum(),1e-300)*mass*w
    out/=out.sum();err=float(abs(out-ps['vis_'+suf+'_oldC_direct']).max());mx=max(mx,err);assert err<1e-14
   qc=ev['C'][ass];p=orig['pB']*qc;p/=p.sum();assert np.array_equal(p,ps['old_Cteacher'])
  else:
   for arm in ps:assert np.array_equal(ps[arm],op['Frozen_A'])
  n+=1
 assert n==1798;dump(H/'FROZEN_AUDIT.json',dict(status='PASS',n=n,eligible=nb,max_routing_or_direct_error=mx,cached_native_reader_reference='R145/IMAGE_AUDIT.json',new_network_forwards=0,GT_read=False,FINAL_access=False,unknown_groups_keep_A_not_negative=True));print('FROZEN150_AUDIT',n,nb,mx,flush=True)
if __name__=='__main__':main()

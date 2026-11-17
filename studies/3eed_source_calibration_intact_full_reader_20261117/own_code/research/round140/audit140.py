"""Independent native CPU readout, exact no-B/reuse and source-calibrated scope."""
from common140 import *
import argparse,numpy as np,torch
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
sys.path.insert(0,str(R/'research/round129'));import support129 as U
LEVELS=['oldreader_ownsource','fullreader_pooled','fullreader_ownsource']
def main():
 check();torch.set_num_threads(4);root=B/('engineering' if args.engineering else 'formal');seal=rd(root/'SEALED.json');W={s:tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};n=0;fits=0;reuse=0;mxp=0.;mxd=0.;mxo=0.;process=[];kappas=rd(H/'SOURCE_KAPPAS_LOCK.json')
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  for D in sorted(root.glob(src+'_to_*')):
   for path in sorted((D/'samples_PRIVATE').glob('*.npz')):
    sid=path.stem;aa=np.load(path);ps=dict(zip(aa['arm_names'],aa['scores']));ev=np.load(B/'evidence_PRIVATE'/D.name/(sid+'.npz'));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));ss=np.load(D/'states_PRIVATE'/(sid+'.npz'));records=rd(D/'records_PRIVATE'/(sid+'.json'))['fits'];old=np.load(R/'crossplatform/round136/method'/D.name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));hasB=rd(R/'crossplatform/round135/formal'/D.name/'records_PRIVATE'/(sid+'.json'))['has_B'];ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];S=ev['S'];rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/D.name/'route_PRIVATE'/(sid+'.npz'))));assert np.array_equal(aa['queries'],aa['scores'].argmax(1)) and np.array_equal(aa['source_boxes'],old['source_boxes']);assert np.isfinite(aa['scores']).all() and np.allclose(aa['scores'].sum(1),1,atol=1e-9)
    for k in ['Frozen_A','Frozen_F','AB_mean','R135_A','R135_F','C_B_A','wrong_C_B_A','direct_C_B_A']:assert np.array_equal(ps[k],op[k])
    for level in LEVELS:
     C=np.zeros_like(S)
     if hasB and S.sum()>=2:
      for prob in [ev['pA'],ev['pB']]:
       x=np.log(np.maximum([prob[g].mean() for g in groups],1e-300))[S];C[S]|=x.max()-x<=kappas[level][src]
     assert np.array_equal(C,ev[level+'_C']);eligible=bool(hasB and S.sum()>=2 and C.any());assert eligible==bool(ev[level+'_eligible']);target=np.exp(np.array([np.log(max(float(ev['pB'][g].max()),1e-300)) for g in groups])[C]-np.max([np.log(max(float(ev['pB'][g].max()),1e-300)) for g in groups if C[ass[g[0]]]])) if eligible else np.zeros(int(C.sum()))
     if eligible:target/=target.sum()
     assert np.allclose(target,ev[level+'_target'],rtol=0,atol=1e-15);perm=ev[level+'_permutation'];assert np.array_equal(np.sort(perm),np.arange(C.sum())) and np.array_equal(ev[level+'_wrong_target'],ev[level+'_target'][perm]);cids=np.flatnonzero(C);assert all(len(groups[cids[j]])==len(groups[cids[perm[j]]]) for j in range(len(cids)))
     # Rebuild direct projection with group mass preservation and group-MAX ratios.
     for kind in ['correct','wrong']:
      t=ev[level+('_target' if kind=='correct' else '_wrong_target')];p=ps['Frozen_A'].copy();ids=np.concatenate([groups[g] for g in np.flatnonzero(S)]) if S.any() else np.array([],int)
      if eligible:
       cc=np.concatenate([groups[g] for g in cids]);mass=p[ids].sum()
       if (S&~C).any():p[ids]=0;p[cc]=ps['Frozen_A'][cc]*mass/ps['Frozen_A'][cc].sum()
       cmass=p[cc].sum();gm=np.array([np.log(max(float(p[g].max()),1e-300)) for g in groups])[C]
       for g,d in zip(cids,np.log(np.maximum(t,1e-300))-gm):p[groups[g]]*=np.exp(d)
       p[cc]*=cmass/p[cc].sum()
      tag=level+'_'+kind;directtag=level+('_direct' if kind=='correct' else '_wrong_direct');assert np.allclose(p,ps[directtag],rtol=0,atol=1e-12)
      rec=records[tag]
      if 'reused' in rec:
       source=rec['reused'];expected=ps['Frozen_A'] if source=='completeFrozenA_no_evidence' else op['C_B_A' if source=='R136_correct' else 'wrong_C_B_A'] if source.startswith('R136_') else ps[source];assert np.array_equal(ps[tag],expected);reuse+=1;continue
      fits+=1;delta=np.zeros(4896,np.float32)
      for j,row in enumerate(rec['trace'],1):
       g=ss[tag+'_gradient'+str(j)]
       if row['accepted']:delta=(delta-np.float32(rec['radius'])*g/np.float32(row['gradient_L2'])*np.float32(row['fraction'])).astype(np.float32)
       assert row['after']<=row['before']+1e-8
       for trial in row['trials']:assert trial['accepted']==(trial['objective']<=trial['bound'])
      err=float(abs(delta-ss[tag+'_delta']).max());mxd=max(mxd,err);assert err<1e-7;feat=tensors({k:ff[tag+'_final_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=read135(feat,rt,W[src]);err=float(abs(prob-ps[tag]).max());mxp=max(mxp,err);assert err<5e-6
      lp=torch.from_numpy(np.maximum(ps[tag],1e-300)).log();pt=float(U.taskloss(lp,groups,torch.from_numpy(S),torch.from_numpy(C),None,'partial'));gm=torch.stack([lp[g].max() for g in groups]);tt=torch.from_numpy(t);rank=float((tt*(tt.clamp_min(1e-300).log()-gm[C].log_softmax(0))).sum());X=torch.from_numpy(np.load(R/'crossplatform/round131/formal'/D.name/'prefix_PRIVATE'/(sid+'.npz'))['vis0'][0].copy());hidden=F.gelu(F.linear(X,W[src]['point_adapter.down.weight'],W[src]['point_adapter.down.bias']));orig=X+F.linear(hidden,W[src]['point_adapter.up.weight'],W[src]['point_adapter.up.bias']);adapt=X+F.linear(hidden,W[src]['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W[src]['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));assert abs(adapt.numpy()-ff[tag+'_adapted_A']).max()<2e-5;keep=float((F.normalize(adapt,dim=-1)-F.normalize(orig,dim=-1)).square().sum(-1).mean()/2);last=rec['trace'][-1];err=max(abs(pt-last['parts_after']['partial']),abs(rank-last['parts_after']['conditional_B_KL']),abs(keep-last['keep_after']),abs(pt+rank+keep-last['after']));mxo=max(mxo,err);assert err<2e-6;process.append({'source':src,'target':D.name.split('_to_')[1],'id_PRIVATE':sid,'arm':tag,'accepted_steps':sum(x['accepted'] for x in rec['trace']),'trials':sum(len(x['trials']) for x in rec['trace']),'parameter_L2':float(np.linalg.norm(delta.astype(float))),'initial_task':rec['trace'][0]['before'],'final_task':last['after']})
    if not hasB:
     for level in LEVELS:
      for k in ['direct','wrong_direct','correct','wrong']:assert np.array_equal(ps[level+'_'+k],ps['Frozen_A'])
    n+=1
    if n%128==0:print('AUDIT140',n,fits,reuse,flush=True)
 assert n==(6 if args.engineering else 1798);out={'status':'PASS','n':n,'new_fits':fits,'compatible_reuses_or_exact_fallbacks':reuse,'max_CPU_native_probability_error':mxp,'max_parameter_delta_error':mxd,'max_objective_error':mxo,'source_kappas_and_geometry_exact':True,'full_transformer_gradients_not_CPU_regenerated':True,'FINAL_access':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),out)
 if not args.engineering:dump(H/'PROCESS_ROWS_PRIVATE.json',process)
 print(out,flush=True)
if __name__=='__main__':main()

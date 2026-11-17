"""Independent evidence, native reader, delta and objective audit; no target GT."""
from common145 import *
import numpy as np,torch,argparse
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct
from common145 import R,H,B,rd,sha,dump,check,utc

def main():
 check();torch.set_num_threads(4);root=B/('engineering_method' if args.engineering else 'method');seal=rd(root/'SEALED.json');ks=rd(H/'SOURCE_KAPPAS_LOCK.json');reg=rd(H/('TTA_ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));mxp=0.;mxs=0.;mxf=0.;mxo=0.;n=0;fits=0;process=[]
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  W=tensors(dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))));radius=.01*float(torch.sqrt(sum(v.double().square().sum() for k,v in W.items() if k.startswith('point_adapter.'))))
  for e in [e for e in reg if e['source']==src]:
   tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name;p=D/'samples_PRIVATE'/(sid+'.npz');a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));old=np.load(B/'image'/name/'samples_PRIVATE'/p.name);op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1)) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
   for arm in op:assert np.array_equal(ps[arm],op[arm])
   ev=np.load(B/'evidence_PRIVATE'/name/p.name);original=np.load(R/'crossplatform/round135/formal'/name/'evidence_PRIVATE'/p.name);assert np.array_equal(ev['S'],original['S']) and np.array_equal(ev['assignment'],original['assignment']);ass=ev['assignment'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];scope=ev['S'];C=np.zeros_like(scope)
   if e['has_B'] and scope.sum()>=2:
    for prob in [op['norm_Aimage'],op['norm_Bimage']]:
     lp=np.log(np.maximum([prob[g].mean() for g in gs],1e-300))[scope];C[scope]|=lp.max()-lp<=ks['kappa'][src]
   assert np.array_equal(C,ev['C']);eligible=bool(e['has_B'] and scope.sum()>=2 and C.any());assert eligible==bool(ev['eligible']);target=np.array([op['norm_Bimage'][g].max() for g in gs])[C];target=target/target.sum() if eligible else target;assert np.allclose(target,ev['target'],atol=1e-15,rtol=1e-13)
   perm=np.arange(len(target));cids=np.flatnonzero(C);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142_Cconditional').encode()).hexdigest()[:16],16))
   for sz in sorted({len(gs[g]) for g in cids}):
    jj=np.array([i for i,g in enumerate(cids) if len(gs[g])==sz],int);order=rng.permutation(jj)
    if len(order)>1:perm[order]=np.roll(order,1)
   assert np.array_equal(perm,ev['permutation']) and np.array_equal(ev['wrong_target'],ev['target'][perm]);expected=direct(op['norm_joint_zero'],gs,scope,C,ev['target']) if eligible else op['norm_joint_zero'];wrong=direct(op['norm_joint_zero'],gs,scope,C,ev['wrong_target']) if eligible else op['norm_joint_zero'];assert np.array_equal(ps['norm_direct'],expected) and np.array_equal(ps['norm_wrong_direct'],wrong);rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert rec['eligible']==eligible and not rec['GT_read'];ff=np.load(D/'features_PRIVATE'/p.name);ss=np.load(D/'states_PRIVATE'/p.name);rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/p.name)));z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/p.name);b=np.load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/p.name);P=np.where(b['valid_B'][:,None],(z['vis0'][0]+b['B_vis'])/2,z['vis0'][0]);X=torch.from_numpy(P.copy())
   for arm in ['norm_TTA','norm_wrong_TTA']:
    fit=rec['fits'][arm]
    if fit.get('reused'):
     ctrl='norm_TTA' if ('wrong' in arm) and eligible else 'norm_joint_zero';assert np.array_equal(ps[arm],ps[ctrl]);continue
    fits+=1;delta=np.zeros(4896,np.float32);assert abs(fit['radius']-radius)<1e-10 and len(fit['trace'])==3
    for j,t in enumerate(fit['trace'],1):
     g=ss[arm+'_gradient'+str(j)]
     if t['accepted']:delta=(delta-np.float32(radius)*g/np.float32(t['gradient_L2'])*np.float32(t['fraction'])).astype(np.float32)
     assert t['after']<=t['before']+1e-8
     for tr in t['trials']:assert tr['accepted']==(tr['objective']<=tr['bound'])
    err=float(abs(delta-ss[arm+'_delta']).max());mxs=max(mxs,err);assert err<1e-7;feat=tensors({k:ff[arm+'_final_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});pr=read135(feat,rt,W);err=float(abs(pr-ps[arm]).max());mxp=max(mxp,err);assert err<5e-6;hidden=F.gelu(F.linear(X,W['point_adapter.down.weight'],W['point_adapter.down.bias']));base=X+F.linear(hidden,W['point_adapter.up.weight'],W['point_adapter.up.bias']);adapt=X+F.linear(hidden,W['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));err=float(abs(adapt.numpy()-ff[arm+'_adapted_A']).max());mxf=max(mxf,err);assert err<2e-5;keep=float((F.normalize(adapt,dim=-1)-F.normalize(base,dim=-1)).square().sum(-1).mean()/2);lp=torch.from_numpy(np.log(np.maximum(ps[arm],1e-300)));lm=torch.stack([lp[g].logsumexp(0)-np.log(len(g)) for g in gs]);partial=float(lm[scope].logsumexp(0)-lm[C].logsumexp(0)) if (scope&~C).any() else 0.;gm=torch.stack([lp[g].max() for g in gs]);tar=torch.from_numpy(ev['wrong_target'] if ('wrong' in arm) else ev['target']);rank=float((tar*(tar.clamp_min(1e-300).log()-gm[C].log_softmax(0))).sum());task=partial+rank;t=fit['trace'][-1];err=max(abs(task-t['task_after']),abs(keep-t['keep_after']),abs(task+keep-t['after']));mxo=max(mxo,err);assert err<2e-6
    process.append({'source':src,'target':tgt,'id_PRIVATE':sid,'arm':arm,'nonzero':bool(delta.any()),'parameter_L2':float(np.linalg.norm(delta.astype(float))),'accepted_steps':sum(t['accepted'] for t in fit['trace']),'trials':sum(len(t['trials']) for t in fit['trace']),'task_after':task,'keep_after':keep,'finite_difference':fit['finite_difference'],'final_F_RGB_AB':True})
   if not e['has_B']:
    for arm in ['norm_TTA','norm_wrong_TTA','norm_joint_zero']:assert np.array_equal(ps[arm],ps['Frozen_A'])
   n+=1
   if n%128==0:print('AUDIT_METHOD145',n,fits,flush=True)
 assert n==len(reg) and n==(len(rd(H/'TTA_ENGINEERING_INPUTS_PRIVATE.json')) if args.engineering else 1798);out={'status':'PASS','n':n,'fit_count':fits,'max_CPU_probability_error':mxp,'max_state_error':mxs,'max_Ffeature_error':mxf,'max_objective_error':mxo,'full_encoder_gradients_not_CPU_regenerated':True,'evidence_C_target_and_direct_reconstructed':True,'original_rho0_3steps_Up4896_and_source_only_initialization':True,'FINAL_access':False};dump(H/('TTA_ENGINEERING_AUDIT.json' if args.engineering else 'TTA_AUDIT.json'),out);dump(H/('TTA_ENGINEERING_PROCESS_ROWS_PRIVATE.json' if args.engineering else 'PROCESS_ROWS_PRIVATE.json'),process);print(out,flush=True)
if __name__=='__main__':main()

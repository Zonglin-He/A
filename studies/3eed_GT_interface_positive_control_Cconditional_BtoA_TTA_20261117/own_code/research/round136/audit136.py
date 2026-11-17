from common136 import *
import numpy as np,torch,sys,argparse
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['oracle','method'],required=True);ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R136/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
sys.path.insert(0,str(R136/'research/round129'));import support129 as U
def main():
 check();torch.set_num_threads(4);root=B136/(('engineering_' if args.engineering else '')+args.stage);seal=rd(root/'SEALED.json');n=0;fits=0;mxp=0.;mxs=0.;mxf=0.;mxo=0.;process=[]
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  W=tensors(dict(np.load(R136/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))))
  for D in sorted(root.glob(src+'_to_*')):
   for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
    a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));sid=p.stem;ev=np.load(B136/'evidence_PRIVATE'/D.name/p.name);old=np.load(R136/'crossplatform/round135/formal'/D.name/'samples_PRIVATE'/p.name);op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']);assert np.array_equal(ps['Frozen_A'],op['intact_live_A']);assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9) and np.array_equal(a['queries'],a['scores'].argmax(1));rec=rd(D/'records_PRIVATE'/(sid+'.json'));ff=np.load(D/'features_PRIVATE'/p.name);ss=np.load(D/'states_PRIVATE'/p.name);rt=tensors(dict(np.load(R136/'crossplatform/round135/formal'/D.name/'route_PRIVATE'/p.name)));z=np.load(R136/'crossplatform/round131/formal'/D.name/'prefix_PRIVATE'/p.name);X=torch.from_numpy(z['vis0'][0].copy());ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];n+=1
    if args.stage=='oracle':
     kinds=['GT3_diagnostic'];records={'GT3_diagnostic':rec.get('fit',{'reused':True})}
     if rec['eligible']:
      y=np.load(R136/'research/round131/scores_attempt001'/(D.name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];positive=y>=y.max()-1e-12;assert np.array_equal(positive,ff['positive']);direct=np.where(positive,ps['Frozen_A'],0);direct/=direct.sum();assert np.array_equal(direct,ps['GT_direct_diagnostic'])
    else:
     kinds=['C_B_A','wrong_C_B_A'];records=rec['fits'];assert np.array_equal(ps['direct_C_B_A'],ev['direct']) and np.array_equal(ps['direct_wrong_C_A'],ev['wrong_direct']);C=ev['C'];ids=np.flatnonzero(C);assert np.array_equal(np.sort(ev['permutation']),np.arange(len(ids)))
     assert np.array_equal(ev['wrong_target'],ev['target'][ev['permutation']]);assert all(len(groups[ids[i]])==len(groups[ids[ev['permutation'][i]]]) for i in range(len(ids)))
    for kind in kinds:
     fit=records[kind]
     if fit.get('reused'):
      compare='C_B_A' if kind=='wrong_C_B_A' and rec['eligible'] else 'Frozen_A';assert np.array_equal(ps[kind],ps[compare]);continue
     fits+=1;prefix='' if args.stage=='oracle' else kind+'_';delta=np.zeros(4896,np.float32)
     for j,t in enumerate(fit['trace'],1):
      g=ss[prefix+'gradient'+str(j)]
      if t['accepted']:delta=(delta-np.float32(fit['radius'])*g/np.float32(t['gradient_L2'])*np.float32(t['fraction'])).astype(np.float32)
      assert t['after']<=t['before']+1e-8
      for q in t['trials']:assert q['accepted']==(q['objective']<=q['bound'])
     err=float(abs(delta-ss[prefix+'delta']).max());mxs=max(mxs,err);assert err<1e-7
     feat=tensors({k:ff[prefix+'final_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=read135(feat,rt,W);err=float(abs(prob-ps[kind]).max());mxp=max(mxp,err);assert err<5e-6
     hidden=F.gelu(F.linear(X,W['point_adapter.down.weight'],W['point_adapter.down.bias']));base=X+F.linear(hidden,W['point_adapter.up.weight'],W['point_adapter.up.bias']);adapt=X+F.linear(hidden,W['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));err=float(abs(adapt.numpy()-ff[prefix+'adapted_A']).max());mxf=max(mxf,err);assert err<2e-5;keep=float((F.normalize(adapt,dim=-1)-F.normalize(base,dim=-1)).square().sum(-1).mean()/2);lp=torch.from_numpy(np.log(np.maximum(ps[kind],1e-300)))
     if args.stage=='oracle':task=float(-lp[positive].logsumexp(0))
     else:
      Sg=torch.from_numpy(ev['S']);Cg=torch.from_numpy(ev['C']);partial=float(U.taskloss(lp,groups,Sg,Cg,None,'partial'));gm=torch.stack([lp[g].max() for g in groups]);target=torch.from_numpy(ev['wrong_target'] if kind.startswith('wrong') else ev['target']);rank=float((target*(target.clamp_min(1e-300).log()-gm[Cg].log_softmax(0))).sum());task=partial+rank
     last=fit['trace'][-1];err=max(abs(task-last['task_after']),abs(keep-last['keep_after']),abs(task+keep-last['after']));mxo=max(mxo,err);assert err<2e-6;process.append({'source':src,'target':D.name.split('_')[-1],'id_PRIVATE':sid,'kind':kind,'nonzero':bool(delta.any()),'parameter_L2':float(np.linalg.norm(delta.astype(float))),'accepted_steps':sum(t['accepted'] for t in fit['trace']),'trials':sum(len(t['trials']) for t in fit['trace']),'task_after':task,'keep_after':keep,'finite_difference':fit['finite_difference']})
    if n%256==0:print('AUDIT136',args.stage,args.engineering,n,flush=True)
 assert n==(12 if args.engineering else 1798);key=('ENGINEERING_' if args.engineering else '')+args.stage.upper();out={'status':'PASS','n':n,'fit_count':fits,'max_CPU_probability_error':mxp,'max_state_error':mxs,'max_feature_error':mxf,'max_objective_error':mxo,'GT_module_only_diagnostic':args.stage=='oracle','FINAL_access':False,'full_encoder_gradients_not_CPU_regenerated':True};dump(H136/(key+'_AUDIT.json'),out);dump(H136/(key+'_PROCESS_ROWS_PRIVATE.json'),process);print(out,flush=True)
if __name__=='__main__':main()

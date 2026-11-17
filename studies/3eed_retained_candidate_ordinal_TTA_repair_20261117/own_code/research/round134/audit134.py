"""Independent post-seal CPU evidence, direct convex-readout, state and head replay."""
from common134 import *
import numpy as np,torch,sys
from scipy.optimize import nnls
from torch.nn import functional as F
sys.path.insert(0,str(R134/'research/round129'));from audit129 import read
def maximum(prob,groups):return np.array([np.log(max(float(prob[g].max()),1e-300)) for g in groups])
def qpcheck(p0,p,groups,w,l,m):
 if not len(w):assert np.array_equal(p0,p);return 0.,0.
 nodes=np.unique(np.r_[w,l]);ix={g:j for j,g in enumerate(nodes)};D=np.zeros((len(w),len(nodes)))
 for i,(a,b) in enumerate(zip(w,l)):D[i,ix[a]]=1;D[i,ix[b]]=-1
 a,b=maximum(p0,groups),maximum(p,groups);delta=b[nodes]-a[nodes];delta-=delta.mean();res=(D@b[nodes])-m;assert res.min()>-1e-7
 active=res<1e-6
 if active.any():mult,r=nnls(D[active].T,delta,maxiter=10000)
 else:r=float(np.linalg.norm(delta))
 assert r<1e-5,(r,len(w))
 # Entire query scores must be a groupwise common offset, no hidden identity edit.
 lp0=np.log(np.maximum(p0,1e-300));lp=np.log(np.maximum(p,1e-300));offset=lp-lp0
 for g in groups:
  live=p0[g]>1e-290
  if live.any():assert np.ptp(offset[g][live])<1e-8
 return float(max(0.,-res.min())),r
def main():
 basecheck();torch.set_num_threads(4);mat=rd(B134/'MATRIX_SEALED.json');reg=rd(H134/'INPUT_REGISTRY_PRIVATE.json');index={(e['source']+'_to_'+e['target'],e['row']['id']):e for e in reg};kappa=rd(R134/'research/round132/SOURCE_LOG_CALIBRATION.json')['kappa'];n=0;active=0;maxp=0.;maxstate=0.;maxobj=0.;maxfeature=0.;maxkkt=0.;maxqp=0.;process=[]
 for lock in ['MODEL_CODE_LOCK','FORMAL_LOCK','EVALUATION_CODE_LOCK']:
  for p,h in rd(H134/'experiment'/(lock+'.json'))['files'].items():assert sha(R134/p)==h,p
 assert len(reg)==1798
 for name,h in mat['seals'].items():
  D=B134/'formal'/name;assert sha(D/'SEALED.json')==h
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h,p
  src=name.split('_')[0]
  with np.load(R134/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as f:W={k[6:]:torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('match.')};down=torch.from_numpy(f['point_adapter.down.weight'].copy());db=torch.from_numpy(f['point_adapter.down.bias'].copy())
  for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
   e=index[(name,p.stem)];a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));assert a['scores'].shape==(9,256) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9);assert np.array_equal(a['queries'],a['scores'].argmax(1));old=np.load(R134/e['original_prediction_path']);op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']);n+=1
   for k,o in [('zero_F','zero_F'),('partial3_F','partial3_F'),('partial_direct_F','direct_F')]:assert np.array_equal(ps[k],op[o])
   assert sha(R134/e['evidence_path'])==e['evidence_sha256'];ev=np.load(R134/e['evidence_path']);ass=ev['assignment'];groups=[np.flatnonzero(ass==j) for j in range(int(ass.max()+1))];S=ev['S'];C=ev['C'];aa,bb,zz=[maximum(v,groups) for v in [ev['pA'],ev['pB'],ps['zero_F']]];gaps=np.minimum(aa[:,None]-aa[None,:],bb[:,None]-bb[None,:]);ww,ll=np.where((gaps>1e-10)&C[:,None]&C[None,:]&(zz[:,None]<zz[None,:]));assert np.array_equal(ww,ev['win']) and np.array_equal(ll,ev['lose']);assert np.allclose(ev['margin'],np.minimum(gaps[ww,ll],kappa),atol=1e-13)
   perm=ev['wrong_perm'];assert np.array_equal(np.sort(perm),np.arange(len(groups))) and np.array_equal(C[perm],C);assert np.array_equal(ev['wrong_win'],perm[ww]) and np.array_equal(ev['wrong_lose'],perm[ll]);assert all(len(groups[i])==len(groups[perm[i]]) for i in range(len(groups)))
   for k,c,w,l in [('pair_direct_F','partial_direct_F',ww,ll),('partial3_pair_direct_F','partial3_F',ww,ll),('wrongpair_direct_F','partial_direct_F',ev['wrong_win'],ev['wrong_lose'])]:
    assert np.array_equal(ps[k],ev[k]);viol,kkt=qpcheck(ps[c],ps[k],groups,w,l,ev['margin']);maxqp=max(maxqp,viol);maxkkt=max(maxkkt,kkt)
   assert bool(len(ww))==e['pair_active']==rec['pair_active']
   if not e['pair_active']:
    assert np.array_equal(ps['pair_only_F'],ps['zero_F']) and np.array_equal(ps['partial_pair_F'],ps['partial3_F']) and np.array_equal(ps['partial_wrongpair_F'],ps['partial3_F']);continue
   active+=1;ff=np.load(D/'features_PRIVATE'/(p.stem+'.npz'));ss=np.load(D/'states_PRIVATE'/(p.stem+'.npz'));X=torch.from_numpy(ff['FAB'].copy());hidden=F.gelu(F.linear(X,down,db));norm=F.normalize(X,dim=-1);zs=torch.from_numpy(ev['S']);zc=torch.from_numpy(ev['C'])
   for kind in ['pair_only','partial_pair','partial_wrongpair']:
    expected=read(ff[kind+'_h'],ff[kind+'_u'],ff['mask'],ff['fixed_late'],ff['fixed_r'],W);err=float(abs(expected-ps[kind+'_F']).max());maxp=max(maxp,err);assert err<5e-6
    delta=np.zeros(4896,np.float32);path=0.;trials=0;accepted=0;trace=rec['traces'][kind]
    for step,t in enumerate(trace,1):
     g=ss[kind+'_step'+str(step)+'_gradient'];before=delta.copy()
     if t['accepted']:delta=(delta-np.float32(rec['radius'])*g/np.float32(t['gradient_L2'])*np.float32(t['fraction'])).astype(np.float32);accepted+=1
     path+=float(np.linalg.norm((delta-before).astype(float)));trials+=len(t['trials']);assert t['after']<=t['before']+1e-8
     for trial in t['trials']:assert trial['accepted']==(trial['objective']<=trial['bound'])
    err=float(abs(delta-ss[kind]).max());maxstate=max(maxstate,err);assert err<1e-7
    adapt=X+F.linear(hidden,torch.from_numpy(delta[:4608].reshape(288,16).copy()),torch.from_numpy(delta[4608:].copy()));err=float(abs(adapt.numpy()-ff[kind+'_adapted_FAB']).max());maxfeature=max(maxfeature,err);assert err<2e-5
    preserve=float((F.normalize(torch.from_numpy(ff[kind+'_adapted_FAB']),dim=-1)-norm).square().sum(-1).mean()/2);lp=np.log(np.maximum(ps[kind+'_F'],1e-300));ls=torch.from_numpy(lp);gs=torch.stack([ls[g].logsumexp(0)-np.log(len(g)) for g in groups]);pt=float(gs[zs].logsumexp(0)-gs[zc].logsumexp(0)) if kind!='pair_only' else 0.;gm=maximum(ps[kind+'_F'],groups);w=ev['wrong_win'] if kind=='partial_wrongpair' else ww;l=ev['wrong_lose'] if kind=='partial_wrongpair' else ll;ordinal=float(np.maximum(ev['margin']-gm[w]+gm[l],0).mean());err=max(abs(pt-trace[-1]['partial_after']),abs(ordinal-trace[-1]['ordinal_after']),abs(preserve-trace[-1]['keep_after']),abs(pt+ordinal+preserve-trace[-1]['after']));maxobj=max(maxobj,err);assert err<2e-6
    process.append({'source':src,'target':e['target'],'id_PRIVATE':p.stem,'kind':kind,'parameter_L2':float(np.linalg.norm(delta.astype(float))),'path_length':path,'accepted_steps':accepted,'line_search_trials':trials,'backward_evaluations':3,'reader_evaluations':6+trials+1,'partial_after':pt,'ordinal_after':ordinal,'feature_keep_after':preserve,'initial_ordinal':rec['initial_ordinal'][kind]['ordinal_loss'],'initial_violated_pairs':rec['initial_ordinal'][kind]['violated_pairs']})
 policy=rd(B134/'POLICY_SEALED.json');assert policy['n']==1798 and policy['eligible']==12 and policy['MATRIX_SEALED_sha256']==sha(B134/'MATRIX_SEALED.json');pn=0;pg=0
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];path=B134/'policy_PRIVATE'/name/(sid+'.npz');assert sha(path)==policy['files'][str(path.relative_to(R134))];a=np.load(B134/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));ev=np.load(R134/e['evidence_path']);g0=ev['assignment'][int(ps['zero_F'].argmax())];gate=bool((ev['lose']==g0).any());b=np.load(path);bp=dict(zip(b['arm_names'],b['scores']));assert bool(b['eligible'])==gate
  for k,a0,b0 in [('tasklinked_pair_F','partial_pair_F','partial3_F'),('tasklinked_wrongpair_F','partial_wrongpair_F','partial3_F'),('tasklinked_direct_F','pair_direct_F','partial_direct_F'),('tasklinked_partial3_direct_F','partial3_pair_direct_F','partial3_F')]:assert np.array_equal(bp[k],ps[a0 if gate else b0])
  pn+=1;pg+=gate
 assert pn==1798 and pg==12
 assert n==1798 and active==381;out={'status':'PASS_FROZEN_PAIRS_CONVEX_DIRECT_KKT_NATIVE_READOUT_STATES_STEPS_OBJECTIVES_AND_REFERENCE_REUSE','n':n,'pair_active':active,'arms':13,'task_relevant_policy_inputs':12,'max_CPU_probability_error':maxp,'max_state_error':maxstate,'max_objective_error':maxobj,'max_point_adapter_feature_error':maxfeature,'max_direct_constraint_violation':maxqp,'max_direct_KKT_residual':maxkkt,'full_decoder_gradients_not_independently_CPU_recomputed':True,'engineering_directional_derivative_checked':True,'GT_read':False,'FINAL_access':False};dump(H134/'NATIVE_AUDIT.json',out);dump(H134/'PROCESS_ROWS_PRIVATE.json',process);print(out)
if __name__=='__main__':main()

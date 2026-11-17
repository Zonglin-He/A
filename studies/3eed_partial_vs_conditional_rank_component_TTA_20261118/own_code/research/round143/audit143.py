"""Independent component evidence, native reader, parameter and loss audit."""
from common143 import *
import argparse,numpy as np,torch
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
ARMS=['partial_only','partial_only_wrong_C','rank_only','rank_only_wrong']

def ppartial(p0,gs,S,C):
 p=p0.copy()
 if S.sum()>=2 and C.any() and (S&~C).any():
  si=np.concatenate([gs[g] for g in np.flatnonzero(S)]);ci=np.concatenate([gs[g] for g in np.flatnonzero(C)]);v=p[si].sum();p[si]=0;p[ci]=p0[ci]*v/p0[ci].sum()
 return p
def prank(p0,gs,C,t):
 p=p0.copy()
 if C.sum()>=2:
  ci=np.concatenate([gs[g] for g in np.flatnonzero(C)]);mass=p[ci].sum()
  for g,tg in zip(np.flatnonzero(C),t):p[gs[g]]*=np.exp(np.log(max(float(tg),1e-300))-np.log(max(float(p0[gs[g]].max()),1e-300)))
  p[ci]*=mass/p[ci].sum()
 return p

def main():
 check();torch.set_num_threads(4);root=B/('engineering' if args.engineering else 'formal');seal=rd(root/'SEALED.json');reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));n=0;fits=0;mxp=0.;mxs=0.;mxf=0.;mxo=0.;mxdec=0.;mxdir=0.;process=[];decomps=[]
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  W=tensors(dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))));radius=.01*float(torch.sqrt(sum(v.double().square().sum() for k,v in W.items() if k.startswith('point_adapter.'))))
  for e in [e for e in reg if e['source']==src]:
   tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name;p=D/'samples_PRIVATE'/(sid+'.npz');a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));parent=np.load(R/'crossplatform/round142/method'/name/'samples_PRIVATE'/p.name);op=dict(zip(parent['arm_names'],parent['scores']));assert np.array_equal(a['source_boxes'],parent['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1)) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
   for arm,prob in op.items():assert np.array_equal(ps[arm],prob)
   means=np.load(R/'crossplatform/round142/supplementary_mean'/name/p.name)
   for arm,prob in zip(means['arm_names'],means['scores']):assert np.array_equal(ps[arm],prob)
   ev=np.load(B/'evidence_PRIVATE'/name/p.name);old=np.load(R/'crossplatform/round142/evidence_PRIVATE'/name/p.name)
   for key in ['assignment','S','C','target','wrong_target','p0','eligible','effective_wrong']:assert np.array_equal(ev[key],old[key])
   ass,S,C=ev['assignment'],ev['S'],ev['C'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];p0=ps['F_RGBB_only'];ids=np.flatnonzero(S);mp=np.maximum(np.array([p0[g].mean() for g in gs]),1e-300);bit=np.full(len(gs),-1,int)
   if len(ids):bit[ids]=np.floor((np.log(mp[ids].max())-np.log(mp[ids]))/np.log(2)).astype(int)
   perm=np.arange(len(gs));rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R143_Cconfidence').encode()).hexdigest()[:16],16))
   for size,bb in sorted({(len(gs[g]),int(bit[g])) for g in ids}):
    jj=np.array([g for g in ids if len(gs[g])==size and bit[g]==bb],int);order=rng.permutation(jj)
    if len(order)>1:perm[order]=np.roll(order,1)
   cw=C[perm];assert np.array_equal(cw,ev['C_wrong_membership']) and np.array_equal(perm,ev['C_wrong_permutation']) and np.array_equal(bit,ev['confidence_bit']);assert cw.sum()==C.sum() and not (cw&~S).any();eligible=bool(ev['eligible']);pa=bool(eligible and (S&~C).any());ra=bool(eligible and C.sum()>=2);assert pa==bool(ev['partial_active']) and ra==bool(ev['rank_active']) and bool(ev['C_wrong_effective'])==(not np.array_equal(cw,C));mass=float(mp[C].sum()/mp[S].sum()) if C.any() else 0.;wmass=float(mp[cw].sum()/mp[S].sum()) if C.any() else 0.;assert mass==float(ev['true_C_conditional_group_mass']) and wmass==float(ev['wrong_C_conditional_group_mass'])
   for arm,ex in [('direct_partial',ppartial(p0,gs,S,C)),('direct_wrong_partial',ppartial(p0,gs,S,cw)),('direct_rank',prank(p0,gs,C,ev['target'])),('direct_wrong_rank',prank(p0,gs,C,ev['wrong_target']))]:
    err=float(abs(ps[arm]-ex).max());mxdir=max(mxdir,err);assert err<1e-12 and np.array_equal(ps[arm],ev[arm])
   rec=rd(D/'records_PRIVATE'/(sid+'.json'));ff=np.load(D/'features_PRIVATE'/p.name);ss=np.load(D/'states_PRIVATE'/p.name);rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/p.name)));z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/p.name);f=np.load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/p.name);P=np.where(f['valid_B'][:,None],(z['vis0'][0]+f['B_vis'])/2,z['vis0'][0]);X=torch.from_numpy(P.copy())
   for arm in ARMS:
    fit=rec['fits'][arm];ispartial=arm.startswith('partial');active=pa if ispartial else ra;ident=arm=='partial_only_wrong_C' and not bool(ev['C_wrong_effective']) or arm=='rank_only_wrong' and not bool(ev['effective_wrong'])
    if fit.get('reused'):
     assert not active or ident;ctrl='partial_only' if arm=='partial_only_wrong_C' else 'rank_only' if arm=='rank_only_wrong' else None;expect=ps[ctrl] if ident and active else p0;assert np.array_equal(ps[arm],expect);continue
    assert active and not ident;fits+=1;delta=np.zeros(4896,np.float32);assert len(fit['trace'])==3 and abs(fit['radius']-radius)<1e-10
    for j,tr in enumerate(fit['trace'],1):
     gg=ss[arm+'_gradient'+str(j)]
     if tr['accepted']:delta=(delta-np.float32(radius)*gg/np.float32(tr['gradient_L2'])*np.float32(tr['fraction'])).astype(np.float32)
     assert tr['after']<=tr['before']+1e-8
     for tt in tr['trials']:assert tt['accepted']==(tt['objective']<=tt['bound'])
    err=float(abs(delta-ss[arm+'_delta']).max());mxs=max(mxs,err);assert err<1e-7;feat=tensors({k:ff[arm+'_final_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});prob=read135(feat,rt,W);err=float(abs(prob-ps[arm]).max());mxp=max(mxp,err);assert err<5e-6;h=F.gelu(F.linear(X,W['point_adapter.down.weight'],W['point_adapter.down.bias']));anchor=X+F.linear(h,W['point_adapter.up.weight'],W['point_adapter.up.bias']);adapt=X+F.linear(h,W['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));err=float(abs(adapt.numpy()-ff[arm+'_adapted_A']).max());mxf=max(mxf,err);assert err<2e-5;keep=float((F.normalize(adapt,dim=-1)-F.normalize(anchor,dim=-1)).square().sum(-1).mean()/2);lp=torch.from_numpy(np.log(np.maximum(ps[arm],1e-300)));CC=cw if arm=='partial_only_wrong_C' else C
    if ispartial:lm=torch.stack([lp[g].logsumexp(0)-np.log(len(g)) for g in gs]);task=float(lm[S].logsumexp(0)-lm[CC].logsumexp(0))
    else:
     gm=torch.stack([lp[g].max() for g in gs]);target=torch.from_numpy(ev['wrong_target'] if arm=='rank_only_wrong' else ev['target']);task=float((target*(target.clamp_min(1e-300).log()-gm[CC].log_softmax(0))).sum())
    last=fit['trace'][-1];err=max(abs(task-last['task_after']),abs(keep-last['keep_after']),abs(task+keep-last['after']));mxo=max(mxo,err);assert err<2e-6;process.append({'source':src,'target':tgt,'id_PRIVATE':sid,'arm':arm,'parameter_L2':float(np.linalg.norm(delta.astype(float))),'nonzero':bool(delta.any()),'accepted_steps':sum(t['accepted'] for t in fit['trace']),'trials':sum(len(t['trials']) for t in fit['trace']),'task_after':task,'keep_after':keep,'finite_difference':fit['finite_difference']})
   if eligible:
    gp=ss['partial_only_gradient1'].astype(float) if pa else np.zeros(4896);gr=ss['rank_only_gradient1'].astype(float) if ra else np.zeros(4896);gf=np.load(R/'crossplatform/round142/method'/name/'states_PRIVATE'/p.name)['image_B_TTA_gradient1'].astype(float);np_,nr_=np.linalg.norm(gp),np.linalg.norm(gr);err=float(np.linalg.norm(gp+gr-gf)/max(np_+nr_,1e-12));mxdec=max(mxdec,err);assert err<.001 and abs(err-rec['gradient_decomposition']['normalized_sum_residual'])<1e-12;decomps.append({'source':src,'target':tgt,'id_PRIVATE':sid,'normalized_residual':err,'partial_gradient_norm':float(np_),'rank_gradient_norm':float(nr_),'cosine':float(gp@gr/(np_*nr_)) if np_ and nr_ else None,'true_C_conditional_group_mass':mass,'wrong_C_conditional_group_mass':wmass,'wrong_C_effective':bool(ev['C_wrong_effective'])})
   if not e['has_B']:
    for arm in ARMS:assert np.array_equal(ps[arm],ps['Frozen_A'])
   n+=1
   if n%128==0:print('AUDIT143',n,fits,flush=True)
 assert n==len(reg);out={'status':'PASS','n':n,'fit_count':fits,'max_CPU_full_reader_error':mxp,'max_state_error':mxs,'max_Ffeature_error':mxf,'max_objective_error':mxo,'max_component_gradient_sum_relative_error':mxdec,'max_independent_direct_error':mxdir,'all_R142_references_exact_and_noB_complete_A':True,'fullTransformer_gradients_not_independently_CPU_regenerated':True,'new_loss_or_sourceweight_training':False,'FINAL_access':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),out);dump(H/('ENGINEERING_PROCESS_ROWS_PRIVATE.json' if args.engineering else 'PROCESS_ROWS_PRIVATE.json'),process);dump(H/('ENGINEERING_GRADIENT_ROWS_PRIVATE.json' if args.engineering else 'GRADIENT_ROWS_PRIVATE.json'),decomps);print(out,flush=True)
if __name__=='__main__':main()

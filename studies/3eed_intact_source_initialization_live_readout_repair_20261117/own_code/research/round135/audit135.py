"""Independent CPU reconstruction of live competition, native matching and updates."""
from common135 import *
import numpy as np,torch,sys,argparse
from torch.nn import functional as F
sys.path.insert(0,str(R135/'research/round129'));import support129 as U
sys.path.insert(0,str(R135/'research/round134'));from audit134 import qpcheck
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def tensors(a):return {k:torch.from_numpy(v.copy()) for k,v in a.items()}
def read135(a,rt,W,late=None,res=None):
 def linear(x,n):return F.linear(x,W[n+'.weight'],W[n+'.bias'])
 def seq(x,n):return linear(F.gelu(linear(x,n+'.0')),n+'.2')
 h=a['h'];x=h[None]
 for j in [0,2]:x=F.relu(linear(x,'match.'+str(j)))
 q=F.normalize(linear(x,'match.4')[0]+(a['late'] if late is None else late),dim=-1);v=q@a['u'].T/.07;mask=a['mask'].bool();s=v[:,mask].logsumexp(-1)-v[:,~mask].logsumexp(-1)
 if res is None:
  E=len(a['local'])
  if E==0:res=h.sum(-1)*0
  else:
   Q=F.layer_norm(linear(h.float(),'interface.q'),(64,));Z=F.layer_norm(linear(a['local'].float(),'interface.z'),(64,));O=F.layer_norm(linear(a['own'].float(),'interface.z'),(64,));T=F.layer_norm(linear(a['text'].float(),'interface.t'),(64,));R=F.layer_norm(linear(a['relation_text'].float(),'interface.t'),(64,))*rt['relation_tokens'].any(-1)[:,None];valid=rt['weights_plain']>0
   def read(query,key,g,v):
    logits=(query[:,None]*key).sum(-1)/8+seq(g,'interface.geo').squeeze(-1);w=logits.masked_fill(~v,-1e4).softmax(-1)*v;w=w/w.sum(-1,keepdim=True).clamp_min(1e-12);return (w[:,:,None]*key).sum(1),(w[:,:,None]*g).sum(1)
   key=torch.cat([O[:,None],Z[0][None].expand(256,-1,-1)],1);g=torch.cat([rt['own_geo'][:,None],rt['hyp_geo'][:,0]],1);valids=torch.cat([rt['own_valid'][:,None],valid[0][None].expand(256,-1)],1);sub,sg=read(T[0][None]+Q,key,g,valids);refs=[];geos=[]
   for e in range(1,E):
    val,geom=read(T[e][None]+R[e][None]+Q+sub,Z[e][None].expand(256,-1,-1),rt['hyp_geo'][:,e],valid[e][None].expand(256,-1));refs.append(val);geos.append(geom)
   ref=torch.stack(refs).mean(0) if refs else Q*0;rel=R[1:].mean(0) if E>1 else T[0]*0;g=torch.stack(geos).mean(0) if geos else sg*0;inp=torch.cat([Q*0,sub,ref,T[0][None].expand(256,-1)+rel[None],g],-1);res=seq(inp,'interface.out').squeeze(-1)
 return (s.double()+res.double()).softmax(0).numpy()
def main():
 basecheck();torch.set_num_threads(4);root=B135/('engineering' if args.engineering else 'formal');seal=rd(root/'MATRIX_SEALED.json')
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 reg=rd(H135/'INPUT_REGISTRY_PRIVATE.json');ix={(e['source']+'_to_'+e['target'],e['row']['id']):e for e in reg};n=0;fits=0;noB=0;maxprob=0.;maxobj=0.;maxstate=0.;maxadapt=0.;maxqp=0.;counts={};process=[];kappa=rd(R135/'research/round132/SOURCE_LOG_CALIBRATION.json')['kappa']
 for src in ['waymo','drone','quad']:
  W=tensors(dict(np.load(B135/(src+'_SOURCE_MODEL_PRIVATE.npz'))))
  for D in sorted(root.glob(src+'_to_*')):
   for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
    sid=p.stem;e=ix[(D.name,sid)];a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));ff=dict(np.load(D/'features_PRIVATE'/(sid+'.npz')));ff.update(dict(np.load(D/'fit_features_PRIVATE'/(sid+'.npz'))));rt=tensors(dict(np.load(D/'route_PRIVATE'/(sid+'.npz'))));ev=dict(np.load(D/'evidence_PRIVATE'/(sid+'.npz')));rec=rd(D/'fit_records_PRIVATE'/(sid+'.json'));ss=dict(np.load(D/'states_PRIVATE'/(sid+'.npz')));n+=1;assert np.array_equal(a['queries'],a['scores'].argmax(1));assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
    def features(prefix):return tensors({k:ff[prefix+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']})
    for sk in ['intact','reset']:
     for v in ['A','F']:
      feat=features(sk+'_'+v)
      for reader in ['live','fixed']:
       calc=read135(feat,rt,W,torch.from_numpy(ff['fixed_late']) if reader=='fixed' else None,torch.from_numpy(ff['fixed_r']) if reader=='fixed' else None);err=float(abs(calc-ps[sk+'_'+reader+'_'+v]).max());assert err<5e-6,(D.name,sid,sk,v,reader,err);maxprob=max(maxprob,err)
    old=np.load(R135/'crossplatform/round97'/D.name/'samples_PRIVATE'/(sid+'.npz'));oldps=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(old['source_boxes'],a['source_boxes']);assert abs(ps['intact_live_A']-oldps['prepared_frozen']).max()<1e-7;assert ps['intact_live_A'].argmax()==oldps['prepared_frozen'].argmax()
    pB=read135(features('teacher_B'),rt,W);assert abs(pB-ev['pB']).max()<5e-6;assert np.array_equal(ev['pA'],ps['intact_live_A']);ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];scope=ev['S'];C=np.zeros_like(scope)
    if scope.sum()>=2:
     for prob in [ev['pA'],ev['pB']]:
      lp=np.log(np.maximum([prob[g].mean() for g in groups],1e-300))[scope];C[scope]|=lp.max()-lp<=kappa
    assert np.array_equal(C,ev['C']);logs=[np.array([np.log(max(float(prob[g].max()),1e-300)) for g in groups]) for prob in [ev['pA'],ev['pB'],ps['intact_live_F']]];gap=np.minimum(logs[0][:,None]-logs[0][None],logs[1][:,None]-logs[1][None]);w,l=np.where((gap>1e-10)&C[:,None]&C[None]&(logs[2][:,None]<logs[2][None]));assert np.array_equal(w,ev['win']) and np.array_equal(l,ev['lose']);assert np.allclose(ev['margin'],np.minimum(gap[w,l],kappa),atol=1e-12);perm=ev['wrong_perm'];assert np.array_equal(np.sort(perm),np.arange(len(groups))) and np.array_equal(C[perm],C);assert all(len(groups[i])==len(groups[perm[i]]) for i in range(len(groups)));assert np.array_equal(ev['wrong_win'],perm[w]) and np.array_equal(ev['wrong_lose'],perm[l])
    for v in ['A','F']:
     direct=U.direct(ps['intact_live_'+v],groups,scope,C,None,'partial');viol,kkt=qpcheck(direct,ps['direct_'+v],groups,w,l,ev['margin']);maxqp=max(maxqp,viol,kkt)
    if not rec['has_B']:
     noB+=1
     for k in ['intact_live_F','direct_A','direct_F','AB_mean','tta_A','tta_F','wrong_A','wrong_F']:assert np.array_equal(ps[k],ps['intact_live_A'])
    z0=np.load(R135/'crossplatform/round131/formal'/D.name/'prefix_PRIVATE'/(sid+'.npz'));A=torch.from_numpy(z0['vis0'][0].copy());B=torch.from_numpy(ff['B_vis']);valid=torch.from_numpy(ff['valid_B']);FAB=torch.where(valid[:,None],(A+B)/2,A)
    for kind,fit in rec['fits'].items():
     if fit.get('reused'):continue
     fits+=1;delta=np.zeros(4896,np.float32)
     for j,t in enumerate(fit['trace'],1):
      grad=ss[kind+'_gradient'+str(j)]
      if t['accepted']:delta=(delta-np.float32(fit['radius'])*grad/np.float32(t['gradient_L2'])*np.float32(t['fraction'])).astype(np.float32)
      assert t['after']<=t['before']+1e-8
      for trial in t['trials']:assert trial['accepted']==(trial['objective']<=trial['bound'])
     err=float(abs(delta-ss[kind+'_delta']).max());maxstate=max(maxstate,err);assert err<1e-7
     P=A if kind.endswith('A') else FAB;h=F.gelu(F.linear(P,W['point_adapter.down.weight'],W['point_adapter.down.bias']));base=P+F.linear(h,W['point_adapter.up.weight'],W['point_adapter.up.bias']);adapt=P+F.linear(h,W['point_adapter.up.weight']+torch.from_numpy(delta[:4608].reshape(288,16)),W['point_adapter.up.bias']+torch.from_numpy(delta[4608:]));err=float(abs(adapt.numpy()-ff[kind+'_adapted']).max());maxadapt=max(maxadapt,err);assert err<2e-5
     prob=read135(features(kind),rt,W);err=float(abs(prob-ps[kind]).max());maxprob=max(maxprob,err);assert err<5e-6
     lp=torch.from_numpy(np.log(np.maximum(ps[kind],1e-300)));gs=U.glogits(lp,groups);pt=float(gs[scope].logsumexp(0)-gs[C].logsumexp(0));gm=np.array([float(lp[g].max()) for g in groups]);ww=ev['wrong_win'] if kind.startswith('wrong') else w;ll=ev['wrong_lose'] if kind.startswith('wrong') else l;ol=float(np.maximum(ev['margin']-gm[ww]+gm[ll],0).mean()) if len(ww) else 0.;keep=float((F.normalize(adapt,dim=-1)-F.normalize(base,dim=-1)).square().sum(-1).mean()/2);last=fit['trace'][-1];err=max(abs(pt-last['partial_after']),abs(ol-last['ordinal_after']),abs(keep-last['keep_after']),abs(pt+ol+keep-last['after']));maxobj=max(maxobj,err);assert err<2e-6
     process.append({'source':src,'target':e['target'],'id_PRIVATE':sid,'kind':kind,'parameter_L2':float(np.linalg.norm(delta.astype(float))),'accepted_steps':sum(t['accepted'] for t in fit['trace']),'trials':sum(len(t['trials']) for t in fit['trace']),'partial_after':pt,'ordinal_after':ol,'keep_after':keep})
    if n%128==0:print('AUDIT135',n,'fits',fits,flush=True)
 assert n==(12 if args.engineering else 1798);out={'status':'PASS','n':n,'fit_count':fits,'no_B_complete_frozen_fallback':noB,'max_probability_error':maxprob,'max_state_error':maxstate,'max_objective_error':maxobj,'max_adapter_error':maxadapt,'max_projection_violation_or_KKT':maxqp,'GT_read':False,'FINAL_access':False,'full_decoder_gradients_not_independently_CPU_recomputed':True};dump(H135/('ENGINEERING_CPU_AUDIT.json' if args.engineering else 'NATIVE_AUDIT.json'),out)
 if not args.engineering:dump(H135/'PROCESS_ROWS_PRIVATE.json',process)
 print(out,flush=True)
if __name__=='__main__':main()

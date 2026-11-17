"""Independent CPU readouts, teacher, feature-keep and parameter-step replay.
Full decoder backprop is validated by engineering directional differences,
not claimed independently CPU-recomputed over the entire experiment.
"""
from pathlib import Path
import json,sys,hashlib,numpy as np,torch,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round127';B=R/'crossplatform/round127'
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
ST=['ab_native','ab_full','fab_native','fab_full','wrong_fab_full','random']
def soft(x):x=x-x.max();p=np.exp(x);return p/p.sum()
def validate(e,D,S):
 sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));f=np.load(D/'features_PRIVATE'/(sid+'.npz'));ss=np.load(D/'states_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'))
 z=np.load(R/e['prefix']);o=np.load(R/e['observations']);pre='real_B_' if e['cohort']=='DEV' else 'B_';A=torch.from_numpy(z['vis0'][0]);BB=torch.from_numpy(o[pre+'vis']) if pre+'vis' in o else A;valid=o[pre+'valid'] if pre+'valid' in o else np.zeros(1024,bool);assert torch.equal(BB[~valid],A[~valid]);FAB=torch.where(torch.from_numpy(valid)[:,None],(A+BB)/2,A)
 head={k[6:]:v for k,v in S.items() if k.startswith('match.')};late=f['fixed_late'];res=f['fixed_r'];mask=f['mask'];zl=np.zeros_like(late);zr=np.zeros_like(res);po=dict(zip(a['arm_names'],a['scores']));maxp=maxloss=maxstate=0.;calls=0;targets={}
 for mode,full in [('native',False),('full',True)]:
  preds=[cpu112(f['teacher_'+v+'_h'],f['teacher_'+v+'_u'],mask,late if full else zl,res if full else zr,head) for v in ['A','B']];T=sum(preds)/2;maxp=max(maxp,float(abs(T-f['teacher_'+mode]).max()));targets[mode]=f['teacher_'+mode];calls+=2
 assert maxp<5e-6
 salt='R127|'+e['source']+'|'+sid;cycle=np.random.default_rng(int(hashlib.sha256((salt+'|wrong_teacher').encode()).hexdigest()[:16],16)).permutation(256);perm=np.empty(256,int);perm[cycle]=np.roll(cycle,1);assert np.array_equal(perm,f['teacher_permutation'])
 radius=.01*np.sqrt(sum(np.square(v.astype(float)).sum() for k,v in S.items() if k.startswith('point_adapter.')));assert abs(radius-rec['fit']['parameter_radius'])<1e-9
 for kind in ST[:-1]:
  ds=torch.zeros(4896)
  for t,trace in enumerate(rec['traces'][kind]):
   g=torch.from_numpy(ss[kind+'_step'+str(t+1)+'_gradient']);old=ds.clone();direction=-radius*g/g.norm() if rec['fit']['active'] and g.norm()>0 else g*0;frac=trace['fraction'];assert frac==0 or frac in [2.**(-j) for j in range(9)];ds=ds+frac*direction
   if trace['accepted']:
    assert trace['trials'][-1]['accepted'] and all(not x['accepted'] for x in trace['trials'][:-1]);assert trace['after']<=trace['before']+1e-4*float(g.double()@(ds-old).double())+2e-12
   else:assert frac==0
  maxstate=max(maxstate,float(abs(ds-torch.from_numpy(ss[kind])).max()))
  full=not kind.endswith('native');target=targets['full' if full else 'native'];target=target[perm] if kind.startswith('wrong_') else target
  views=[('A',A),('B',BB)] if kind.startswith('ab_') else [('F',FAB)];lps=[];ks=[]
  for v,P in views:
   pp=cpu112(f[kind+'_fit_'+v+'_h'],f[kind+'_fit_'+v+'_u'],mask,late if full else zl,res if full else zr,head);calls+=1;lps.append(np.log(pp))
   hidden=F.gelu(F.linear(P,torch.from_numpy(S['point_adapter.down.weight']),torch.from_numpy(S['point_adapter.down.bias'])));adapt=P+F.linear(hidden,ds[:4608].reshape(288,16),ds[4608:]);ks.append(float((F.normalize(adapt,dim=-1)-F.normalize(P,dim=-1)).square().sum(-1).mean()/2))
  kl=float((target*(np.log(target)-np.mean(lps,axis=0))).sum());keep=float(np.mean(ks));end=rec['traces'][kind][-1];err=max(abs(kl-end['KL_after']),abs(keep-end['keep_after']),abs(kl+keep-end['after']));maxloss=max(maxloss,err);assert err<2e-5,(kind,'loss',err)
 rr=torch.tensor(np.random.default_rng(int(hashlib.sha256((salt+'|norm_random').encode()).hexdigest()[:16],16)).normal(size=4896),dtype=torch.float32);want=rr/rr.norm()*torch.from_numpy(ss['fab_full']).norm();maxstate=max(maxstate,float(abs(want-torch.from_numpy(ss['random'])).max()));assert maxstate<1e-5
 for setting in ['zero']+ST:
  wr=next(x for x in rec['writebacks'] if x['setting']==setting);norm=0 if setting=='zero' else float(torch.from_numpy(ss[setting]).double().norm());assert abs(norm-wr['parameter_L2'])<1e-8
  for v in ['F','C']:
   p=cpu112(f[setting+'_'+v+'_h'],f[setting+'_'+v+'_u'],mask,late,res,head);calls+=1;target=po[setting+'_'+v];er=float(abs(p-target).max());maxp=max(maxp,er);assert er<5e-6 and p.argmax()==target.argmax(),(setting,v,er)
   if setting=='zero':
    pn=cpu112(f[setting+'_'+v+'_h'],f[setting+'_'+v+'_u'],mask,zl,zr,head);direct=soft(np.log(targets['native'])+np.log(p)-np.log(pn));calls+=1;er=float(abs(direct-po['direct_native_bridge_'+v]).max());maxp=max(maxp,er);assert er<5e-6 and direct.argmax()==po['direct_native_bridge_'+v].argmax();assert np.array_equal(po['direct_full_'+v],targets['full'])
 assert rec['C_arrays_used_in_gradient'] is False and rec['scope_parameters']==4896 and rec['no_source_GT_used'] is True
 provider=rec['source_A_provider'];assert sha(R/provider['path'])==provider['sha256']
 with np.load(R/provider['path']) as old:
  pre='frozen_full_A_' if e['cohort']=='DEV' else 'real_A_';ll=old[pre+'late'];ll=ll[0] if e['cohort']=='DEV' else ll;assert np.array_equal(ll,late) and np.array_equal(old[pre+'r'],res)
 old=np.load(R/f"crossplatform/round125/formal/{e['cohort']}/{e['source']}_to_{e['target']}/features_PRIVATE/{sid}.npz")
 for v in ['A','B']:
  for k in ['h','u']:assert np.array_equal(f['teacher_'+v+'_'+k],old['teacher_'+v+'_'+k]),'same frozen source native teacher bridge'
 return {'max_probability_error':maxp,'max_loss_error':maxloss,'max_state_error':maxstate,'CPU_head_calls':calls}
def main(probe=False):
 torch.set_num_threads(4);reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');cols=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=aliases=active=heads=0;mxp=mxl=mxs=0.;out=[]
 if not probe:
  mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==2304
  for combo,h in mat['directions'].items():
   D=B/'formal'/combo;assert sha(D/'SEALED.json')==h
   for p,v in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==v
 for src in ['waymo','drone','quad']:
  S=dict(np.load(R/cols[src]['state_path']));assert sha(R/cols[src]['state_path'])==cols[src]['state_sha256']
  for co in ['DEV','EXP']:
   for e in reg:
    if e['source']!=src or e['cohort']!=co:continue
    D=B/('engineering' if probe else 'formal')/co/(src+'_to_'+e['target']);sid=e['row']['id'];rp=D/'records_PRIVATE'/(sid+'.json')
    if probe and (not rp.exists() or e['protected_alias_fallback'] or not rd(rp)['fit']['active']):continue
    rec=rd(rp);n+=1;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));assert np.array_equal(a['queries'],a['scores'].argmax(1));old=np.load(R/f'crossplatform/round125/formal/{co}/{src}_to_{e["target"]}/samples_PRIVATE/{sid}.npz');po=dict(zip(a['arm_names'],a['scores']));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes'])
    for k,ok in [('original_source','original_source'),('prepared_A','prepared_A'),('legacy_ABC_context','mean_context')]:assert np.array_equal(po[k],op[ok])
    if e['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(x,op['prepared_A']) for k,x in po.items() if k not in ['original_source','prepared_A','legacy_ABC_context']);continue
    active+=rec['fit']['active'];v=validate(e,D,S);out.append(v);heads+=v['CPU_head_calls'];mxp=max(mxp,v['max_probability_error']);mxl=max(mxl,v['max_loss_error']);mxs=max(mxs,v['max_state_error'])
    if probe:break
    if n%32==0:print('AUDIT127',n,flush=True)
 if probe:assert n==6
 else:assert n==2304 and aliases==154 and active==786
 summary={'state':'PASS_INDEPENDENT_CPU_TEACHERS_ALL_READOUTS_OBJECTIVES_STATE_ARITHMETIC_PROVIDERS','n':n,'aliases':aliases,'active':active,'CPU_head_calls':heads,'max_probability_error':mxp,'max_loss_error':mxl,'max_state_error':mxs,'GT_used':False,'FINAL_access':False,'full_decoder_backprop_independently_CPU_recomputed':False}
 (H/('CPU_PROBE.json' if probe else 'NATIVE_AUDIT.json')).write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main('--probe' in sys.argv)

"""Independent CPU task-native targets, step/free-field arithmetic and all head reads.
Decoder gradients: engineering central differences, not claimed all-CPU backprop.
"""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round125';B=R/'crossplatform/round125';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
sys.path.insert(0,str(R/'research/round118'));import core118 as C
ST=['corr1','corr3','wrong1','wrong3','random1','random3','free3'];REF=['original_source','prepared_A','prepared_F','native_F','native_ABC','lateOnly_F','mean_context','headmean_context','robust_context','weight_mean_context','zero_C']
def soft(x):x=x-x.max();p=np.exp(x);return p/p.sum()
def validate(e,D,head,S):
 sid=e['row']['id'];src=e['source'];co=e['cohort'];name=src+'_to_'+e['target'];sample=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));st=np.load(D/'states_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));z=np.load(R/e['prefix']);ob=np.load(R/e['observations']);pre='real_B_' if co=='DEV' else 'B_';A=torch.from_numpy(z['vis0'][0]);BB=torch.from_numpy(ob[pre+'vis']) if pre+'vis' in ob else A;mask=z['token_mask'];late=ff['fixed_late'];res=ff['fixed_r'];zeroL=np.zeros_like(late);zeroR=np.zeros_like(res);po=dict(zip(sample['arm_names'],sample['scores']));radius=float(rec['fit']['parameter_radius']);original=.01*np.sqrt(sum(np.square(v.astype(float)).sum() for k,v in S.items() if k.startswith('point_adapter.')));assert abs(original-radius)<1e-9;maxp=maxd=maxloss=0.;heads=0
 pa=cpu112(ff['teacher_A_h'],ff['teacher_A_u'],mask,zeroL,zeroR,head);pb=cpu112(ff['teacher_B_h'],ff['teacher_B_u'],mask,zeroL,zeroR,head);teacher=(pa+pb)/2;T=ff['teacher_probability'];maxp=max(maxp,float(abs(teacher-T).max()));assert maxp<5e-6;heads+=2;salt='R125|'+src+'|'+sid;cycle=np.random.default_rng(int(hashlib.sha256((salt+'|querywrong').encode()).hexdigest()[:16],16)).permutation(256);perm=np.empty(256,int);perm[cycle]=np.roll(cycle,1);assert np.array_equal(perm,ff['teacher_permutation']);RA=F.normalize(A,dim=-1);RB=F.normalize(BB,dim=-1);hiddenA=F.gelu(F.linear(A,torch.from_numpy(S['point_adapter.down.weight']),torch.from_numpy(S['point_adapter.down.bias'])));hiddenB=F.gelu(F.linear(BB,torch.from_numpy(S['point_adapter.down.weight']),torch.from_numpy(S['point_adapter.down.bias'])))
 def point(ds):return A+F.linear(hiddenA,ds[:4608].reshape(288,16),ds[4608:]),BB+F.linear(hiddenB,ds[:4608].reshape(288,16),ds[4608:])
 for kind in ['corr','wrong']:
  ds=torch.zeros(4896)
  for t,trace in enumerate(rec['traces'][kind]):
   g=torch.from_numpy(st[kind+str(t+1)+'_gradient']);old=ds.clone();step=-radius*g/g.norm() if rec['fit']['active'] and g.norm()>0 else g*0;fraction=trace['fraction'];assert fraction==0 or fraction in [2.**(-j) for j in range(9)];ds=ds+fraction*step
   if trace['accepted']:assert trace['trials'][-1]['accepted'] and all(not x['accepted'] for x in trace['trials'][:-1]);assert trace['after']<=trace['before']+1e-4*float(g.double()@(ds-old).double())+2e-12
   else:assert fraction==0
   # Parameter endpoint and gradient names deliberately differ in serialized output.
   if t in [0,2]:
    endpoint=st[kind+str(t+1)];er=float(abs(ds-torch.from_numpy(endpoint)).max());maxd=max(maxd,er)
 for t in [1,3]:
  v=torch.tensor(np.random.default_rng(int(hashlib.sha256((salt+'|random|'+str(t)).encode()).hexdigest()[:16],16)).normal(size=4896),dtype=torch.float32);want=v/v.norm()*torch.from_numpy(st['corr'+str(t)]).norm();er=float(abs(want-torch.from_numpy(st['random'+str(t)])).max());maxd=max(maxd,er)
 XA,XB=A.clone(),BB.clone()
 for t,trace in enumerate(rec['traces']['free']):
  ga=torch.from_numpy(st['free'+str(t+1)+'_A_gradient']);gb=torch.from_numpy(st['free'+str(t+1)+'_B_gradient']);n=float(torch.sqrt(ga.double().square().sum()+gb.double().square().sum()));factor=trace['feature_step_energy_cap']/n if n else 0.;XA=XA-factor*trace['fraction']*ga;XB=XB-factor*trace['fraction']*gb
 for key,want in [('free_A_delta',XA-A),('free_B_delta',XB-BB),('free3',((XA-A)+(XB-BB))/2)]:maxd=max(maxd,float(abs(want-torch.from_numpy(st[key])).max()))
 assert maxd<1e-5,(co,name,'state arithmetic',maxd)
 for setting in ['zero']+ST:
  delta=torch.zeros(4896) if setting=='zero' else torch.from_numpy(st[setting]);wr=next(x for x in rec['writebacks'] if x['setting']==setting);expected_norm=0. if setting=='free3' else float(delta.double().norm());assert abs(expected_norm-wr['actual_parameter_L2'])<1e-8;aa,bb=(A+torch.from_numpy(st['free_A_delta']),BB+torch.from_numpy(st['free_B_delta'])) if setting=='free3' else point(delta);keep=float(((F.normalize(aa,dim=-1)-RA).square().sum(-1).mean()+(F.normalize(bb,dim=-1)-RB).square().sum(-1).mean())/4);fa=cpu112(ff[setting+'_fit_A_h'],ff[setting+'_fit_A_u'],mask,zeroL,zeroR,head);fb=cpu112(ff[setting+'_fit_B_h'],ff[setting+'_fit_B_u'],mask,zeroL,zeroR,head);kl=float((T*(np.log(T)-(np.log(fa)+np.log(fb))/2)).sum());err=abs(kl+keep-rec['auxiliary'][setting]['fit_objective']);maxloss=max(maxloss,err);assert err<2e-5;heads+=2
  if setting.startswith('corr') or setting.startswith('wrong'):
   kind=setting[:-1];t=int(setting[-1])-1;target=T if kind=='corr' else T[perm];value=float((target*(np.log(target)-(np.log(fa)+np.log(fb))/2)).sum())+keep;assert abs(value-rec['traces'][kind][t]['after'])<2e-5
  if setting=='free3':
   last=rec['traces']['free'][-1];value=last['trials'][-1]['objective'] if last['accepted'] else last['before'];assert abs(kl+keep-value)<2e-5
  for v,reference in [('F','mean_context'),('C','zero_C')]:
   hp,up=ff[setting+'_'+v+'_h'],ff[setting+'_'+v+'_u'];p=cpu112(hp,up,mask,late,res,head);heads+=1
   if setting=='zero':
    target=po[reference];pn=cpu112(hp,up,mask,zeroL,zeroR,head);heads+=1;direct=soft(np.log(T)+np.log(p)-np.log(pn));er=float(abs(direct-po['direct_'+v]).max());maxp=max(maxp,er);assert er<5e-6 and direct.argmax()==po['direct_'+v].argmax()
   else:target=po[setting+'_'+v]
   er=float(abs(p-target).max());maxp=max(maxp,er);assert er<5e-6 and p.argmax()==target.argmax(),(co,name,setting,v,er)
   if v=='C' and 'heldout_C_KL' in rec['auxiliary'][setting]:
    native=cpu112(hp,up,mask,zeroL,zeroR,head);value=float((T*(np.log(T)-np.log(native))).sum());err=abs(value-rec['auxiliary'][setting]['heldout_C_KL']);maxloss=max(maxloss,err);assert err<2e-5,(co,name,setting,'heldoutC',err);heads+=1
 return {'maxp':maxp,'maxd':maxd,'maxloss':maxloss,'heads':heads}
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==2304 and rd(H/'ENGINEERING.json')['n']==24
 for p,h in rd(H/'experiment/DESIGN_LOCK.json')['files'].items():assert sha(R/p)==h
 reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');cols=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=aliases=heads=active=0;maxp=maxd=maxloss=0.
 for key,sealed in mat['directions'].items():
  co,name=key.split('/');src,_,tgt=name.split('_');D=B/'formal'/co/name;assert sha(D/'SEALED.json')==sealed
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  assert sha(R/cols[src]['state_path'])==cols[src]['state_sha256'];S=dict(np.load(R/cols[src]['state_path']));head={k[6:]:v for k,v in S.items() if k.startswith('match.')}
  for e in [e for e in reg if e['cohort']==co and e['source']==src and e['target']==tgt]:
   n+=1;sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(a['arm_names'],a['scores']));assert np.array_equal(a['queries'],a['scores'].argmax(1));old=np.load(R/f'crossplatform/round122/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');op=dict(zip(old['arm_names'],old['scores']));hm=np.load(R/f'crossplatform/round122/headmean_controls/{co}/{name}/{sid}.npz');op.update(dict(zip(hm['arm_names'],hm['scores'])));priorC=np.load(R/f'crossplatform/round120/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');op['zero_C']=dict(zip(priorC['arm_names'],priorC['scores']))['context_zero_C'];assert np.array_equal(a['source_boxes'],old['source_boxes'])
   for k in REF:assert np.array_equal(po[k],op[k])
   if e['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[k],op['prepared_A']) for k in mat['new_arms']);continue
   rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert rec['C_arrays_used_in_gradient'] is False and rec['no_source_GT_used'] is True;active+=rec['fit']['active'];N,SS,proof=C.components(co,name,sid);ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));assert proof==rec['providers'] and np.array_equal(ff['fixed_late'],SS['A']['late']) and np.array_equal(ff['fixed_r'],SS['fusedABC']['r'])
   for v,oldv in [('F','fusedABC'),('C','C')]:assert np.array_equal(ff['zero_'+v+'_h'],N[oldv]['h']) and np.array_equal(ff['zero_'+v+'_u'],N[oldv]['u'])
   for v in ['A','B']:assert np.array_equal(ff['teacher_'+v+'_h'],N[v]['h']) and np.array_equal(ff['teacher_'+v+'_u'],N[v]['u'])
   out=validate(e,D,head,S);maxp=max(maxp,out['maxp']);maxd=max(maxd,out['maxd']);maxloss=max(maxloss,out['maxloss']);heads+=out['heads']
   if n%32==0:print('AUDIT125',key,n,flush=True)
 assert n==2304 and aliases==154 and active==786;out={'state':'PASS_ALL_NATIVE_TEACHER_DIRECT_READOUTS_HEADS_PARAMETER_AND_FREE_FIELD_ARITHMETIC','n':n,'aliases':aliases,'active_AB_inputs':active,'CPU_head_calls':heads,'max_probability_error':maxp,'max_state_arithmetic_error':maxd,'max_objective_error':maxloss,'full_decoder_gradients_independently_CPU_recomputed':False,'gradient_validation':'engineering finite differences plus all step arithmetic/head/teacher/endpoint objectives; no all-CPU decoder backprop claim','GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

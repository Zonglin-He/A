"""Independent CPU low-rank correspondence loss/SGD/free-field and task head audit."""
from pathlib import Path
import json,sys,hashlib,numpy as np,torch,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round123';B=R/'crossplatform/round123';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round118'));import core118 as C
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
ST=['corr1','corr3','wrong1','wrong3','random1','random3','free3'];REF=['original_source','prepared_A','prepared_F','native_F','native_ABC','lateOnly_F','mean_context','headmean_context','robust_context','weight_mean_context','zero_C']
def loss(A,B,RA,RB,ids,far,perm=None):
 a=F.normalize(A,dim=-1);b=F.normalize(B,dim=-1);q=a[ids];k=b[ids];k=k[perm] if perm is not None else k
 if len(ids)>=2 and far.any():
  s=q@k.T/.07;mask=far|torch.eye(len(ids),dtype=torch.bool);s=s.masked_fill(~mask,-torch.inf);good=far.any(1);v=(F.cross_entropy(s[good],torch.arange(len(ids))[good])+F.cross_entropy(s.T[good],torch.arange(len(ids))[good]))/2
 else:v=A.sum()*0+B.sum()*0
 keep=((a-RA).square().sum(-1).mean()+(b-RB).square().sum(-1).mean())/4;return v+keep

def reproduce(A,B,ids,far,perm,S,radius,salt):
 hiddenA=F.gelu(F.linear(A,torch.from_numpy(S['point_adapter.down.weight']),torch.from_numpy(S['point_adapter.down.bias'])));hiddenB=F.gelu(F.linear(B,torch.from_numpy(S['point_adapter.down.weight']),torch.from_numpy(S['point_adapter.down.bias'])));RA=F.normalize(A,dim=-1);RB=F.normalize(B,dim=-1);states={};energies=[]
 def get(delta):return A+F.linear(hiddenA,delta[:4608].reshape(288,16),delta[4608:]),B+F.linear(hiddenB,delta[:4608].reshape(288,16),delta[4608:])
 for key in ['corr','wrong']:
  delta=torch.zeros(4896);oldA,oldB=A,B
  for t in range(3):
   delta.requires_grad_(True);a,b=get(delta);L=loss(a,b,RA,RB,ids,far,perm if key=='wrong' else None);g=torch.autograd.grad(L,delta)[0];v=-radius*g/g.norm() if g.norm()>0 and len(ids)>=2 and far.any() else g*0;delta=(delta+v).detach();a,b=get(delta)
   if key=='corr':energies.append(float(torch.sqrt((a-oldA).double().square().sum()+(b-oldB).double().square().sum())))
   oldA,oldB=a.detach(),b.detach()
   if t in [0,2]:states[key+str(t+1)]=delta.clone()
 for t in [1,3]:
  rng=np.random.default_rng(int(hashlib.sha256((salt+'|random|'+str(t)).encode()).hexdigest()[:16],16));v=torch.tensor(rng.normal(size=4896),dtype=torch.float32);states['random'+str(t)]=v/v.norm()*states['corr'+str(t)].norm()
 a,b=A.clone(),B.clone()
 for energy in energies:
  a.requires_grad_(True);b.requires_grad_(True);L=loss(a,b,RA,RB,ids,far);ga,gb=torch.autograd.grad(L,[a,b]);n=float(torch.sqrt(ga.double().square().sum()+gb.double().square().sum()));k=energy/n if n else 0;a=(a-k*ga).detach();b=(b-k*gb).detach()
 v=((a-A)+(b-B))/2;valid=torch.zeros(len(A),dtype=torch.bool);valid[ids]=True;v[~valid]=0;states['free3']=v;return states,get

def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==2304 and rd(H/'ENGINEERING.json')['n']==24
 for p,h in rd(H/'experiment/DESIGN_LOCK.json')['files'].items():assert sha(R/p)==h,p
 reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');n=aliases=heads=statesn=active=ties=0;maxp=maxdelta=maxloss=maxnorm=0.
 for key,sealed in mat['directions'].items():
  co,name=key.split('/');src,_,tgt=name.split('_');D=B/'formal'/co/name;assert sha(D/'SEALED.json')==sealed
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  S=dict(np.load(B/(src+'_AUDIT_MODEL_PRIVATE.npz')));radius=float(S['original_adapter_radius']);head={k[6:]:v for k,v in S.items() if k.startswith('match.')}
  for e in [r for r in reg if r['cohort']==co and r['source']==src and r['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(a['arm_names'],a['scores']));assert np.array_equal(a['queries'],a['scores'].argmax(1));old=np.load(R/f'crossplatform/round122/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');op=dict(zip(old['arm_names'],old['scores']));hm=np.load(R/f'crossplatform/round122/headmean_controls/{co}/{name}/{sid}.npz');op.update(dict(zip(hm['arm_names'],hm['scores'])));c0=np.load(R/f'crossplatform/round120/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');op['zero_C']=dict(zip(c0['arm_names'],c0['scores']))['context_zero_C'];assert np.array_equal(a['source_boxes'],old['source_boxes']);n+=1
   for ref in REF:assert np.array_equal(po[ref],op[ref])
   if e['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[s+'_'+v],op['prepared_A']) for s in ST for v in ['F','C']);continue
   rec=rd(D/'records_PRIVATE'/(sid+'.json'));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));saved=np.load(D/'states_PRIVATE'/(sid+'.npz'));z=np.load(R/e['prefix']);ob=np.load(R/e['observations']);pre='real_B_' if co=='DEV' else 'B_';has=pre+'vis' in ob;AA=torch.from_numpy(z['vis0'][0]);BB=torch.from_numpy(ob[pre+'vis']) if has else AA;valid=ob[pre+'valid'] if has else np.zeros(1024,bool);ids=torch.from_numpy(np.flatnonzero(valid));xx=z['xyz'][0][ids.numpy()].astype(float);far=torch.from_numpy(((xx[:,None]-xx[None])**2).sum(-1)>3.2**2);assert np.array_equal(ids.numpy(),ff['fit_ids']) and np.array_equal(far.numpy(),ff['fit_far']);salt='R123|'+src+'|'+sid;order=np.random.default_rng(int(hashlib.sha256((salt+'|wrong').encode()).hexdigest()[:16],16)).permutation(len(ids));p=np.empty(len(ids),int)
   if len(ids):p[order]=np.roll(order,1)
   assert np.array_equal(p,ff['fit_permutation']);perm=torch.tensor(p);pred,get=reproduce(AA,BB,ids,far,perm,S,radius,salt);active+=rec['fit']['active'];assert rec['C_arrays_used_in_gradient'] is False and rec['fit']['no_C_decoded_before_all_states_fixed'];N,SS,hashes=C.components(co,name,sid);assert hashes==rec['providers'] and np.array_equal(ff['fixed_late'],SS['A']['late']) and np.array_equal(ff['fixed_r'],SS['fusedABC']['r'])
   wr={r['setting']:r for r in rec['writebacks']}
   for s in ST:
    er=float(abs(pred[s]-torch.from_numpy(saved[s])).max());maxdelta=max(maxdelta,er);assert er<5e-6,(co,name,s,er)
    delta=torch.from_numpy(saved[s]);norm=0. if s=='free3' else float(delta.double().norm());er=abs(norm-wr[s]['actual_parameter_L2']);maxnorm=max(maxnorm,er);assert er<1e-8
    pa,pb=(AA+delta,BB+delta) if s=='free3' else get(delta);L=float(loss(pa,pb,F.normalize(AA,dim=-1),F.normalize(BB,dim=-1),ids,far));er=abs(L-rec['auxiliary'][s]['fit_objective']);maxloss=max(maxloss,er);assert er<1e-5;statesn+=1
    for view in ['F','C']:
     pp=cpu112(ff[s+'_'+view+'_h'],ff[s+'_'+view+'_u'],ff[s+'_'+view+'_mask'],ff['fixed_late'],ff['fixed_r'],head);want=po[s+'_'+view];er=float(abs(pp-want).max());maxp=max(maxp,er);assert er<5e-6
     if pp.argmax()!=want.argmax():assert want.max()-want[pp.argmax()]<=2*er+1e-12;ties+=1
     heads+=1
   for view,nview,reference in [('F','fusedABC','mean_context'),('C','C','zero_C')]:
    assert np.array_equal(ff['zero_'+view+'_h'],N[nview]['h']) and np.array_equal(ff['zero_'+view+'_u'],N[nview]['u']);pp=cpu112(ff['zero_'+view+'_h'],ff['zero_'+view+'_u'],ff['zero_'+view+'_mask'],ff['fixed_late'],ff['fixed_r'],head);er=float(abs(pp-po[reference]).max());maxp=max(maxp,er);assert er<5e-6;assert pp.argmax()==po[reference].argmax()

   if n%32==0:print('AUDIT123',key,n,flush=True)
 assert n==2304 and aliases==154 and heads==30100 and statesn==15050;out={'state':'PASS_ALL_POINT_CORRESPONDENCE_SGD_FREE_FEATURE_FIELDS_SCOPE_NORMS_AND_NATIVE_READOUTS','n':n,'aliases':aliases,'CPU_heads':heads,'updated_states':statesn,'active_AB_inputs':active,'max_probability_error':maxp,'max_state_element_error':maxdelta,'max_loss_error':maxloss,'max_actual_norm_error':maxnorm,'near_ties':ties,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

"""Independent CPU objective/gradient audit on fixed hash-selected active cases."""
import sys,time,hashlib,json
from pathlib import Path
import torch,numpy as np
R=Path(__file__).resolve().parents[2]
def rd(p):return json.loads(Path(p).read_text())
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(4);start=time.perf_counter();rows=rd(R/'research/round24/input/ALL336_PRIVATE.json');pred=R/'crossplatform/round55/adapted336_attempt002';obs=R/'crossplatform/round55/conflicts336_attempt001';selected=[]
for pl in ['drone','quad']:
 pool=[r for r in rows if r['platform']==pl and rd(pred/'cases_PRIVATE'/(r['id']+'.json'))['statistics']['real']['nonzero_conflict_pairs']>0];selected+=sorted(pool,key=lambda r:hashlib.sha256(('R55_GRAD_AUDIT:'+r['id']).encode()).digest())[:2]
source=rd(R/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json');state=torch.load(source['source_checkpoint_path'],map_location='cpu',weights_only=True)['model'];prefix='contrastive_align_projection_image.';w={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)};del state
records=[];maxerr=0.
for row in selected:
 sid=row['id'];l=load(R/'crossplatform/round53/latent336_attempt001/samples_PRIVATE'/(sid+'.npz'),['h','tokens']);o=load(obs/'samples_PRIVATE'/(sid+'.npz'),['role_masks','conflicts']);f=load(pred/'fitted_PRIVATE'/(sid+'.npz'));rec=rd(pred/'cases_PRIVATE'/(sid+'.json'));hidden=torch.from_numpy(l['h'])[None]
 for k in [0,2]:hidden=torch.relu(torch.nn.functional.linear(hidden,w[str(k)+'.weight'],w[str(k)+'.bias']))
 def zscores(weight):
  q=torch.nn.functional.normalize(torch.nn.functional.linear(hidden,weight,w['4.bias']),dim=-1);logits=(q@torch.from_numpy(l['tokens'])[None].transpose(-1,-2)/.07)[0];out=[]
  for m in o['role_masks']:
   m=torch.from_numpy(m);out.append(torch.logsumexp(logits[:,m],-1)-torch.logsumexp(logits[:,~m],-1))
  return torch.stack(out)
 z0=zscores(w['4.weight']).detach();p0=z0.double().softmax(-1);grads=[]
 for vi,v in enumerate(['real','RGBshuffle','endpointshuffle']):
  A=torch.tensor(f[v+'_1__A'],requires_grad=True);B=torch.zeros((64,4),requires_grad=True);C=torch.from_numpy(o['conflicts'][vi]).double();z=zscores(w['4.weight']+B@A);known=torch.cat([C.any(0).any(1)[None],C.any(1)],0) if len(C) else torch.zeros_like(z,dtype=torch.bool);pp=[]
  for e in range(len(z)):
   d=z[e].double()*0
   for m in [known[e],~known[e]]:
    if m.any():d=d.masked_scatter(m,z[e,m].double().softmax(-1)*p0[e,m].sum())
   pp.append(d)
  p=torch.stack(pp)
  if len(C):mass=(p[0][None,:,None]*p[1:,None,:]*C).sum((1,2));conf=-torch.log1p(-mass).mean()
  else:conf=z.sum()*0
  anchor=z0.argmax(1,keepdim=True);keep=((z-z.gather(1,anchor))-(z0-z0.gather(1,anchor))).square().mean();loss=conf+keep+.01*(B@A).square().mean();loss.backward();g=torch.cat([A.grad.flatten(),B.grad.flatten()]);norm=float(g.norm());expected=rec['statistics'][v]['steps'][0]['gradient_norm'];maxerr=max(maxerr,abs(norm-expected));assert abs(norm-expected)<1e-5 and torch.count_nonzero(A.grad)==0
  grads.append(g.detach());records.append({'platform':row['platform'],'variant':v,'gradient_norm':norm,'producer_norm_abs_error':abs(norm-expected),'first_A_gradient_exactly_zero':True,'conflict_loss':float(conf.detach())})
 records.append({'platform':row['platform'],'real_vs_RGBshuffle_gradient_difference_L2':float((grads[0]-grads[1]).norm()),'real_vs_endpointshuffle_gradient_difference_L2':float((grads[0]-grads[2]).norm())})
out=R/'research/round55/gradient_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R55_INDEPENDENT_CPU_NATIVE_PROJECTION_OBJECTIVE_GRADIENT_AUDIT','cases':len(selected),'selection':'two active-observation cases per platform by fixed input hash, no GT','independent_gradients':12,'max_gradient_norm_abs_error':maxerr,'first_Bzero_Agrad_zero_expected':True,'trainable_parameters':1408,'records':records,'no_optimizer_or_new_full_grounder':True,'GT_reads':0,'seconds':time.perf_counter()-start,'program_sha256':sha(__file__)}
(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

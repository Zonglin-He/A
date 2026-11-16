"""Separate CPU construction of late-stage consensus and original spatial IDs."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,torch.nn.functional as F
from scipy.special import logsumexp,softmax
R=Path(__file__).resolve().parents[2];H=R/'research/round122';B=R/'crossplatform/round122';sys.path.insert(0,str(R/'research/round118'));import core118 as C
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);mat=rd(B/'headmean_controls/MATRIX_SEALED.json');n=aliases=ties=0;mx=0.
 for key,sealed in mat['directions'].items():
  co,name=key.split('/');src,_,tgt=name.split('_');D=B/'headmean_controls'/co/name;assert sha(D/'SEALED.json')==sealed;W=C.head(src)
  for f,h in rd(D/'SEALED.json')['files'].items():assert sha(D/f)==h
  for e in [x for x in rd(H/'CACHE_REGISTRY_PRIVATE.json') if x['cohort']==co and x['source']==src and x['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/(sid+'.npz'));assert np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   if e['protected_alias_fallback']:aliases+=1;continue
   N,S,_=C.components(co,name,sid);raw=[]
   for view in ['A','B','C']:
    xx=torch.from_numpy(N[view]['h'])
    for layer in ['0','2']:xx=F.relu(F.linear(xx,W[layer+'.weight'],W[layer+'.bias']))
    raw.append(F.linear(xx,W['4.weight'],W['4.bias']))
   raw=torch.stack(raw).mean(0);u=F.normalize(torch.stack([torch.from_numpy(N[v]['u']) for v in ['A','B','C']]).mean(0),dim=-1);mask=N['A']['mask'].astype(bool)
   for i in range(2):
    q=F.normalize(raw+(torch.from_numpy(S['A']['late']) if i else 0),dim=-1);logits=(q@u.T/.07).numpy().astype(float);tokenp=np.exp(logits-logsumexp(logits,axis=1,keepdims=True));mass=tokenp[:,mask].sum(1);score=np.log(np.maximum(mass,1e-300))-np.log(np.maximum(1-mass,1e-300));score+=S['fusedABC']['r'] if i else 0;pp=softmax(score);want=a['scores'][i];er=float(abs(pp-want).max());mx=max(mx,er);assert er<5e-6,er
    if pp.argmax()!=want.argmax():assert want.max()-want[pp.argmax()]<=2*er+1e-12;ties+=1
  print('AUDIT_HEADMEAN122',key,flush=True)
 assert n==2304 and aliases==154;out={'state':'PASS_SAME_OBSERVATION_RAW_PROJECTED_CONSENSUS_DIRECT_CONTROLS','n':n,'readouts':4608,'aliases':aliases,'max_probability_error':mx,'near_ties':ties,'GT_used':False};(H/'HEADMEAN_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

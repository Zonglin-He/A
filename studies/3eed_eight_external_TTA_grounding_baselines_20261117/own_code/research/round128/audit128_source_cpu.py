from pathlib import Path
import sys,json,numpy as np,torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'research/round128'));from models128 import PairHead,RGBHead,MaskedRegion
from memory128 import UniCache
h=R/'research/round128';reg=json.load(open(h/'REGION_REGISTRY_PRIVATE.json'));torch.set_num_threads(4);checks=[]
for src in ['waymo','drone','quad']:
 sel=json.load(open(R/f'crossplatform/round128/prepared_PRIVATE/{src}/SELECTED.json'));assert 'MATE_Chamfer_correction' in sel
 m=PairHead();m.load_state_dict(torch.load(R/sel['models']['pair']['path'],weights_only=True,map_location='cpu'));m.eval();x=np.load(R/next(e['path'] for e in reg if e['role']=='source' and e['source']==src and e['split']=='VAL128'));f=torch.nn.functional.normalize(torch.from_numpy(x['joint']),dim=1)
 with torch.no_grad():
  prob=m(f).softmax(1);cache=UniCache();y,meta=cache.forward(f,prob,m);assert torch.isfinite(y).all() and torch.allclose(y.sum(1),torch.ones(256),atol=1e-6);items=cache.cache[0]+cache.cache[1];K=torch.stack([a[0] for a in items]).numpy().astype(float);P=torch.stack([a[2] for a in items]).numpy().astype(float);W=K@K.T;W[W<.5]=0;dinv=1/np.sqrt(W.sum(1)+1e-8);L=np.eye(len(K))-dinv[:,None]*W*dinv[None];Y=np.linalg.solve(L+.26*np.eye(len(K)),.26*P);Y/=Y.sum(1,keepdims=True);assert np.isfinite(Y).all();checks.append({'source':src,'cache_items':len(K),'cache_writes':meta['cache_writes'],'row_probability_error':float(abs(y.sum(1)-1).max()),'independent_graph_solve_finite':True,'source_only':True})
(h/'SOURCE_INTERFACE_AUDIT.json').write_text(json.dumps({'status':'PASS','checks':checks,'target_labels_or_examples_loaded':False},indent=2)+'\n');print(checks)

"""Unchanged native network with graph offsets derived from actual CNN grids."""
from common152 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from geometry152 import merged
from common152 import R,H,B,rd,sha,dump,check,utc
def read_dynamic(base,m,z,graphs,rt,row):
 assert all(g['image'].shape==(512,768,3) and len(g['i2p'])==6144 for g in graphs)
 g=merged(graphs);d=tensor_data(g,row);d['wrong_permutation']=torch.arange(1024,device='cuda');route=tensor_route89(rt);extra=[torch.as_tensor(q['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255 for q in graphs[1:]]
 with torch.no_grad(),rgb_banks141(m,extra):lp,a,_=live135(m,base,z,d,z['vis0'][0],route)
 return lp.exp().cpu().numpy(),{k:a[k].cpu().numpy() for k in FEATURES},g

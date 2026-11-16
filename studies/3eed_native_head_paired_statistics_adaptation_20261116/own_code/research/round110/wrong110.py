"""Prelocked label-free within-direction moment-anchor derangement."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round110';B=R/'crossplatform/round110';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(B/'original_PRIVATE/SEALED.json')['n']==724;out={}
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');ids=[r['id'] for r in rr if (B/'original_PRIVATE'/name/(r['id']+'.npz')).exists()];rng=np.random.default_rng(int(hashlib.sha256(('R110anchor|'+name).encode()).hexdigest()[:16],16));ix=rng.permutation(len(ids));mapping={ids[ix[j]]:ids[ix[(j+1)%len(ids)]] for j in range(len(ids))};assert all(k!=v for k,v in mapping.items()) and set(mapping)==set(mapping.values());out[name]=mapping
 p=H/'WRONG_ANCHOR_PRIVATE.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('WRONG110_LOCKED',sum(map(len,out.values())))
if __name__=='__main__':main()

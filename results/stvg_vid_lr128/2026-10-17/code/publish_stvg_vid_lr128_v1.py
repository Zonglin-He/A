"""Explicit allowlist publication for the completed Vid LR controls."""
from pathlib import Path
import inspect,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_vid_lr128_v1 import BASE
def main(mode,pixel=None):
 if mode=='close':
  from scripts import close_stvg_referent_abc_publication_v1 as c
  ns=dict(c.__dict__);ns['BASE']=BASE;s=inspect.getsource(c.main).replace('count=16','count=128');exec(compile(s,__file__,'exec'),ns);return ns['main']('lr128',pixel)
 from scripts import publish_stvg_controls256_v1 as p
 p.SCOPES['global32']=BASE.name
 return p.main('global32','lr128',mode,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)

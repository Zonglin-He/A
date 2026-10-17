"""Established manifest/CAS/actual remote-content publication for T2/T4."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_t2_v1 import BASE
def main(mode,pixel=None):
 if mode=='close':
  from scripts import close_stvg_referent_abc_publication_v1 as c
  src=inspect.getsource(c.main);assert src.count('count=16')==1;src=src.replace('count=16','count=256');ns=dict(c.__dict__);ns['BASE']=BASE;exec(compile(src,__file__,'exec'),ns);return ns['main']('t2t4',pixel)
 from scripts import publish_stvg_controls256_v1 as p
 p.SCOPES['global32']=BASE.name;return p.main('global32','t2t4',mode,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[3] if len(sys.argv)>3 else None)

"""Existing manifest/CAS/remote byte workflow for each complete64 family."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_followups64_v1 import BASE
def main(family,mode,pixel=None):
 if mode=='close':
  from scripts import close_stvg_referent_abc_publication_v1 as c
  src=inspect.getsource(c.main);assert src.count('count=16')==1;src=src.replace('count=16','count=64');ns=dict(c.__dict__);ns['BASE']=BASE;exec(compile(src,__file__,'exec'),ns);return ns['main'](family,pixel)
 from scripts import publish_stvg_controls256_v1 as p
 p.SCOPES['global32']=BASE.name;return p.main('global32',family,mode,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)

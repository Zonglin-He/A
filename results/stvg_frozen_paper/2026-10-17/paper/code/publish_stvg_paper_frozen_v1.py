"""Existing manifest/CAS and actual remote byte/tree checks; explicit scope."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_stvg_paper_frozen_v1 import BASES

def main(kind,mode,pixel=None):
 base=BASES['paper'] if kind=='freeze' else BASES[kind]
 if mode=='close':
  from scripts import close_stvg_referent_abc_publication_v1 as c
  count={'freeze':0,'paper':256,'qcqp':32}[kind];s=inspect.getsource(c.main).replace('count=16','count='+str(count));ns=dict(c.__dict__);ns['BASE']=base;exec(compile(s,__file__,'exec'),ns);return ns['main'](kind,pixel)
 from scripts import publish_stvg_controls256_v1 as p
 from scripts.stvg_paper_public_privacy_v1 import verify_privacy
 p.pub.verify_privacy=verify_privacy;p.SCOPES['global32']=base.name
 if mode=='stage':
  # Updating the explicitly exported manuscript/code is authorized by this
  # completed group's publication. Original clean-tree/privacy guards remain.
  import json
  paths=set(json.loads((base/'publication'/kind/'EXPORT_MANIFEST.json').read_text())['paths'])
  p.pub.BASE=base/'publication'/kind;p.pub.expected_paths=lambda:paths
  return p.pub.stage(p.pub.DEFAULT_CHECKOUT,None,True)
 return p.main('global32',kind,mode,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)

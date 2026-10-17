"""Existing anonymous allowlist, fresh CAS and remote bytes/mode/tree workflow."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main(mode,arg=None):
    from scripts import publish_stvg_controls256_v1 as p,verify_stvg_controls_public_remote005 as v
    scope='stvg_pairwise_caption256_v1';group='pairwise256';p.SCOPES['core256']=scope
    if mode=='verify_remote':
        source=inspect.getsource(v.main).replace('artifacts/stvg_spatial_controls256_v1/publication','artifacts/'+scope+'/publication');ns=dict(v.__dict__);exec(compile(source,__file__,'exec'),ns);return ns['main'](group,arg)
    if mode=='close':return p.main('core256',group,'close',arg)
    return p.main('core256',group,mode)
if __name__=='__main__':main(*sys.argv[1:])

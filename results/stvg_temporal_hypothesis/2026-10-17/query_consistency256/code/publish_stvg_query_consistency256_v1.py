"""Allowlist/remote publication for a role note(count0) and sealed CPU diagnosis256."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_query_consistency_diagnosis256_v1'
def main(group,mode,arg=None):
    from scripts import publish_stvg_controls256_v1 as p,verify_stvg_controls_public_remote005 as v
    p.SCOPES['core256']=BASE.name
    if mode=='verify_remote':
        src=inspect.getsource(v.main).replace('artifacts/stvg_spatial_controls256_v1/publication','artifacts/'+BASE.name+'/publication');ns=dict(v.__dict__);exec(compile(src,__file__,'exec'),ns);ns['main'](group,arg)
    elif mode=='close':
        from scripts import close_stvg_referent_abc_publication_v1 as close
        count=0 if group=='position001'else 256;src=inspect.getsource(close.main).replace('count=16','count='+str(count));ns=dict(close.__dict__);ns.update(BASE=BASE);exec(compile(src,__file__,'exec'),ns);ns['main'](group,arg)
    else:p.main('core256',group,mode)
if __name__=='__main__':main(*sys.argv[1:])

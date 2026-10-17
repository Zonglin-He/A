"""Use existing allowlist/CAS workflow for this isolated completed256 group."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_controls256_v1 as p
from scripts import verify_stvg_controls_public_remote005 as v

def main(mode,arg=None):
    p.SCOPES['core256']='stvg_temporal_marginal_train256_v1'
    if mode=='verify_remote':
        assert arg
        src=inspect.getsource(v.main).replace('artifacts/stvg_spatial_controls256_v1/publication','artifacts/stvg_temporal_marginal_train256_v1/publication');ns=dict(v.__dict__);exec(compile(src,__file__,'exec'),ns);ns['main']('marginal_train256',arg)
    elif mode=='close':p.main('core256','marginal_train256','close','results/stvg_temporal_hypothesis/2026-10-17/marginal_train256/effects.png')
    else:p.main('core256','marginal_train256',mode)
if __name__=='__main__':main(*sys.argv[1:])

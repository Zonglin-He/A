"""Existing allowlist workflow, original256 publication only."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main(mode,pixel=None):
    from scripts import publish_stvg_controls256_v1 as p
    p.SCOPES['core256']='stvg_temporal_hypothesis256_v1'
    return p.main('core256','matched256',mode,pixel)
if __name__=='__main__':main(*sys.argv[1:])

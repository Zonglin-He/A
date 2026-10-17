"""Original256 actual HTTP bytes/hash and fetched Git mode/tree verification."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import verify_stvg_controls_public_remote005 as old
def main(commit):
    src=inspect.getsource(old.main).replace('artifacts/stvg_spatial_controls256_v1/publication','artifacts/stvg_temporal_hypothesis256_v1/publication');ns=dict(old.__dict__);exec(compile(src,__file__,'exec'),ns);ns['main']('matched256',commit)
if __name__=='__main__':main(sys.argv[1])

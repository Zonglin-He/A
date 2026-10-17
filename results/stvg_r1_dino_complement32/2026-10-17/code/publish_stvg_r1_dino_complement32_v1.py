"""Explicit allowlist and verified remote closure for the finite complement group."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_dino_complement32_v1 import BASE
def main(mode,pixel=None):
 from scripts import publish_stvg_controls256_v1 as p
 p.SCOPES['global32']=BASE.name
 return p.main('global32','complement32',mode,pixel)
if __name__=='__main__':main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)

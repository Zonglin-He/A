"""Public closure for the separately versioned caption-repaired Native factorial."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import FIXED
from scripts import publish_stvg_factorial_native256_v1 as p
if __name__=='__main__':
    p.BASE=FIXED;p.main(sys.argv[1],sys.argv[3] if len(sys.argv)>3 else None)

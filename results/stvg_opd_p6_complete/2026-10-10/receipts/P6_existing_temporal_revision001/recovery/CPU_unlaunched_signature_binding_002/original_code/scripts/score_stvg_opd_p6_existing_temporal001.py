"""P6 post-deployment-seal original offline oracle and full CPU arithmetic."""
import copy
import gc
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import *


def run():
    # The implementation is separately pinned before the first GPU qualification.
    # No oracle can be called until every deployment prediction is hash-sealed.
    verify();b=read(BASE/'P6_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['queries']==64 and b['logical_outputs']==256 and not b['GT_read']
    from scripts.stvg_opd_p6_postseal_root001 import score
    return score(b)


if __name__=='__main__':run()

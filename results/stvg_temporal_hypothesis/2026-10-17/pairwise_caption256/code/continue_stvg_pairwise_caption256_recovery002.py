"""Exact remaining readouts then postseal scoring; same six-hour deadline."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_pairwise_caption256_v1 as old
from scripts.stvg_pairwise_caption256_recovery002 import BASE,REC,verify,read,write
UNIT='stvg-pairwise-caption256-002.service'
def main():
    src=inspect.getsource(old.main)
    needle="write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+6*3600,GT_read=False));";assert src.count(needle)==1;src=src.replace(needle,'')
    start=src.index("    stages=[");end=src.index('\n    try:',start);src=src[:start]+"    stages=[('readout002','readout',[]),('score','score',[]),('report','report',[])]"+src[end:]
    src=src.replace("BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json'","REC/'ACTUAL_LIVE_RESOURCE_LIMITS.json'").replace("BASE/'FAILURE.json'","REC/'FAILURE.json'").replace("else'stvg_pairwise_caption256_v1.py'","else'stvg_pairwise_caption256_recovery002.py'")
    ns=dict(old.__dict__,UNIT=UNIT,REC=REC,verify=verify);exec(compile(src,__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()

"""Same finite temporal supervisor with inherited original deadline, no reset."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_qwen_b3_temporal32_v1 as old
from scripts.stvg_qwen_b3_temporal32_448_v2 import BASE,verify,read,write
UNIT='stvg-qwen-b3-temporal32-002.service'
def main():
    source=inspect.getsource(old.main);needle="write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+4*3600,GT_read=False));";assert source.count(needle)==1;source=source.replace(needle,'')
    source=source.replace('stvg_qwen_b3_temporal32_v1.py','stvg_qwen_b3_temporal32_448_v2.py').replace('report_stvg_qwen_b3_temporal32_v1.py','report_stvg_qwen_b3_temporal32_448_v2.py');ns=dict(old.__dict__,BASE=BASE,UNIT=UNIT,verify=verify);exec(compile(source,__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()

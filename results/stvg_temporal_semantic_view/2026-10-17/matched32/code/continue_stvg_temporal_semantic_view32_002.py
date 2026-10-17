"""Resume only targets onward after constructor annotation read was blocked."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_temporal_semantic_view32_v1 as first
from scripts.stvg_temporal_semantic_grounder32_v2 import BASE,read,sha,verify
UNIT='stvg-temporal-semantic-view32-002.service'
def main():
    verify();assert read(BASE/'TEACHER_ROOT_READBACK.json')['status']=='pass';assert read(BASE/'execution/targets/WORKER_EXIT.json')['exit_code']!=0;assert not (BASE/'targets').exists()
    src=inspect.getsource(first.main)
    src=src.replace("'ACTUAL_LIVE_RESOURCE_LIMITS.json'","'ACTUAL_LIVE_RESOURCE_LIMITS002.json'").replace("'QWEN_INACTIVE_READBACK.json'","'QWEN_INACTIVE_READBACK002.json'").replace("'FAILURE.json'","'FAILURE002.json'")
    # Preserve elapsed scientific budget from original actual start.
    needle="tick=time.time();write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=tick,deadline_epoch=tick+14400,GT_read=False));";assert src.count(needle)==1;src=src.replace(needle,'')
    line=next(s for s in src.splitlines() if s.strip().startswith('stages='));src=src.replace(line,"        stages=[('targets002','stvg_temporal_semantic_grounder32_v2.py'),('fit','stvg_temporal_semantic_view32_v1.py'),('readout002','stvg_temporal_semantic_grounder32_v2.py')]")
    src=src.replace("/'scripts'/script),stage]","/'scripts'/script),stage.removesuffix('002')]")
    ns=dict(first.__dict__);ns.update(UNIT=UNIT,verify=verify);exec(compile(src,__file__,'exec'),ns);return ns['main']()
if __name__=='__main__':first.release() if len(sys.argv)>1 and sys.argv[1]=='release' else main()

"""Same bounded003 supervisor, separate immutable execution/profile namespace."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_qwen35_geometry32_003 as prior
from scripts.stvg_qwen35_geometry_run_v4 import BASE,UNIT,verify_run
def main():
    source=inspect.getsource(prior.main).replace('ACTUAL_LIVE_RESOURCE_LIMITS003.json','ACTUAL_LIVE_RESOURCE_LIMITS004.json').replace("'execution003'","'execution004'").replace('geometry_run_v3.py','geometry_run_v4.py').replace('geometry32_v3.py','geometry32_v4.py').replace("revision='003'","revision='004'").replace('GROUP_PUBLICATION_HANDOFF003.json','GROUP_PUBLICATION_HANDOFF004.json').replace('FAILURE003.json','FAILURE004.json').replace("stage='003'","stage='004'")
    ns=dict(prior.__dict__,BASE=BASE,UNIT=UNIT,verify_run=verify_run);exec(compile(source,__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()

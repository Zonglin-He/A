"""Keep strict and compatibility failures; disclose unique-box recovery and cost."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_qwen35_geometry32_v3 as prior
def report():
    src=inspect.getsource(prior.report).replace('recovery/singleton_wire003','recovery/unique_bbox_wire004').replace('WIRE_RECOVERY003','WIRE_RECOVERY004').replace('previous_responses_reused=3','previous_responses_reused=4').replace('new_unique_requests=125','new_unique_requests=124').replace('reused_original_generations=3','reused_original_generations=4').replace('new_recovery_generations=125','new_recovery_generations=124').replace("'execution003'","'execution004'").replace('geometry_run_v3.py','geometry_run_v4.py').replace('geometry32_v3.py','geometry32_v4.py').replace('first three raw outputs','first four raw outputs').replace('125 remaining','124 remaining').replace('One returned exactly one bbox_2d/label object in a list.','Two envelope variants returned a unique bbox payload: one list, one object lacking only status.').replace('one GT-free singleton compatibility rule','one GT-free unique-payload compatibility rule')
    ns=dict(prior.__dict__);exec(compile(src,__file__,'exec'),ns);ns['report']()
if __name__=='__main__':{'report':report,'render':prior.old.render}[sys.argv[1]]()

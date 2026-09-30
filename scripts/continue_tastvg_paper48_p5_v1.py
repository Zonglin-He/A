"""Required P5 follows P0-P4 without wall-clock or readiness cutoffs."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_p5_common_v1 import BASE,read,write,sha

def run():
    from scripts.continue_tastvg_paper48_v1 import stage
    from scripts.prepare_tastvg_paper48_p5_v1 import run as prepare
    prepare()
    # Shared serial helper stores stages under the top-level Paper48 logs.
    stage('P5_spatial','scripts/run_tastvg_paper48_p5_experts_v1.py',['spatial'],'P5/experts/SPATIAL_BARRIER.json')
    stage('P5_temporal','scripts/run_tastvg_paper48_p5_experts_v1.py',['temporal'],'P5/experts/TEMPORAL_BARRIER.json',python='.venv-exost/bin/python')
    stage('P5_online','scripts/run_tastvg_paper48_p5_online_v1.py',[],'P5/PREDICTION_BARRIER.json')
    stage('P5_score','scripts/score_tastvg_paper48_p5_v1.py',[],'P5/COMPLETION.json',gpu=False)
    assert read(BASE/'AUDIT.json')['status']=='pass'
    panels=['P0','P1','P2','P3_b0','P3_b25','P3_b100','P4','P5']
    write(BASE.parent/'ALL_PHASES_COMPLETION.json',dict(status='completed_pending_root_review_and_publication',phases=panels,completion_hashes={n:sha(BASE.parent/n/'COMPLETION.json') for n in panels},deadline_unix=None,time=time.time()))
    write(BASE/'EXECUTION_COMPLETION.json',dict(status='completed_pending_root_review',completion_sha256=sha(BASE/'COMPLETION.json'),time=time.time()))
if __name__=='__main__':run()

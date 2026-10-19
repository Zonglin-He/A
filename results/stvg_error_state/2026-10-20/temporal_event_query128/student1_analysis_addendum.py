"""Before-GT supplementary contrasts: don't confuse inherited S1 gain with feedback."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.score_stvg_where_when_event128_v1 import paired
P=R/'results/stvg_error_state/2026-10-20/temporal_event_query128';D=R/'artifacts/stvg_temporal_event_query128_v1'

def main():
    lock=read(D/'ANALYSIS_ADDENDUM_LOCK.json');assert sha(Path(__file__))==lock['script_sha256'];exposure=read(D/'GT_EXPOSURE.json');assert lock['created_epoch']<exposure['epoch'];rows=read(P/'RESULTS_PUBLIC.json')['rows'];out={}
    for a,b in [('ROcc1Mix','Student1'),('ROcc0Mix','Student1'),('ROcc1','Student1'),('ROcc1','ROcc0'),('Sample1Mean','Student1')]:
        out[a+'_minus_'+b]={k:paired([100*(r['values']['LiveMean_'+a][k]-r['values']['LiveMean_'+b][k])for r in rows])for k in ['v','t']}
    put(P/'STUDENT1_SUPPLEMENT_PUBLIC.json',dict(status='completed',pre_GT_addendum=lock,contrasts=out,main_science_predictions_selection_unchanged=True));print(json.dumps(out,indent=2))

if __name__=='__main__':main()

"""Unchanged full128 independent scoring, with pre-GT old-pilot replay audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_temporal_local_motion_completion128_v1 import D,P
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
    import numpy as np,torch
    from scripts import score_stvg_temporal_local_motion_fixedbatch128_v1 as scoring
    p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
    assert all(sha(R/f)==h for f,h in p['pins'].items())
    checked=0
    for a,receipt in p['prior_pilot_predictions'].items():
        assert sha(R/receipt['path'])==receipt['sha256']
        old=load(R/receipt['path']);new=load(D/'predictions'/f'{a}.pt')
        for arm in ['DenseLocal','SparseLocal']:
            assert torch.equal(old['motion_features'][arm],new['motion_features'][arm])
            assert np.array_equal(old['clip_indices'][arm],new['clip_indices'][arm]);checked+=2
        for arm in p['arms']:
            assert np.array_equal(old['outputs'][arm]['q'],new['outputs'][arm]['q'])
            assert old['outputs'][arm]['interval']==new['outputs'][arm]['interval'];checked+=2
        for space,b in old['trajectories'].items():
            assert np.array_equal(b,new['trajectories'][space]);checked+=1
    put(D/'OLD_PILOT_REPLAY_BEFORE_GT.json',dict(status='pass',checks=checked,parents=2,new_GT=False,old_budget_stop_preserved=True))
    scoring.D,scoring.P=D,P
    scoring.main()
    decision=read(P/'DECISION_PUBLIC.json')
    previous=decision['decision']
    strong=decision['versus_Strong']
    hard=decision['primary_hard']
    qualifies=hard['mean']>=2 and hard['CI95'][0]>0 and strong['CI95'][0]>0
    decision.update(previous_fixedbatch_rule_decision=previous,
        decision='eligible_frozen_component_confirmation_only'if qualifies else'close_this_local_motion_configuration',
        pure_expected_quality_is_diagnostic_only=True,
        actual_full_v_and_Strong_required=True,no_automatic_OPD_or_next_run=True)
    put(P/'COMPLETION_DECISION_PUBLIC.json',decision)
    report=P/'REPORT.md'
    report.write_text('# Budget-only completion of fixed-batch local motion128\n\n'
        'Actual execution entry: scripts/run_stvg_temporal_local_motion_completion128_v1.py; '
        'scoring entry: scripts/score_stvg_temporal_local_motion_completion128_v1.py. '
        'User explicitly reopened the unfinished task measurement. New finite GPU cap1800s; '
        'old900s budget stop, two unscored pilots and cost remain immutable. '
        'All128 rerun in one loaded-model process so terminal weight/flag verification covers this run. '
        'Two old pilot features/posteriors/intervals/spatial arrays independently replay exactly before GT. '
        'No changed numerical gate, clip, input, arm, score, threshold, teacher, spatial state or parameter. '
        'Previous engineering and old fixedbatch costs separately disclosed; no automatic confirmation/OPD.\n\n'
        +report.read_text()+'\n\n## Current user decision boundary\n\n'
        'The inherited fixedbatch decision above is retained as its original diagnostic. '
        'Current final decision is COMPLETION_DECISION_PUBLIC.json: actual full-v must improve '
        'Native and Strong; pure expected quality cannot authorize OPD. No automatic next experiment.\n'
        +__import__('json').dumps(decision,indent=2)+'\n')
    put(P/'OLD_PILOT_REPLAY_PUBLIC.json',read(D/'OLD_PILOT_REPLAY_BEFORE_GT.json'))

if __name__=='__main__':main()

"""Final immutable-input, omitted-readout, budget and seal audit on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,collections
from scripts.tastvg_oracle_event5_common_v1 import *
def run():
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    tick=time.monotonic();r=verified(include_GT=True);checks=collections.Counter()
    for f,h in r['inputs'].items():
        assert sha(ROOT/f)==h,f;checks['old_input_file_sha256']+=1
    partial=read(BASE/'recovery/confirmation_compact_readout_005/PARTIAL_RESULT_HASHES.json')
    for f,h in partial.items():
        assert sha(PUBLIC/f)==h,f;checks['valid_partial_byte_reproduction']+=1
    events=read(BASE/'EVENT_PLAN.json')['events'];cohort=read(BASE/'COHORT.json')
    assert len(cohort['cells'])==1152 and len(events)==288 and sum(c['eligible'] for c in events)==276
    assert sum(not c['eligible'] for c in events)==12
    global_seal=read(BASE/'GLOBAL_INTERVENTION_BARRIER.json');assert global_seal['donors']==288
    attempts=[json.loads(s) for s in (BASE/'EXPERT_ATTEMPTS.jsonl').read_text().splitlines()]
    resource=read(BASE/'EXPERT_RESOURCES.json')
    assert len(attempts)==resource['new_calls']==67 and resource['reused']==209
    assert len({a['input_sha256'] for a in attempts})==67
    checks['new_expert_calls_input_unique']=len(attempts)
    latest_U_time=0;old_U_readout=0
    for ds in DATASETS:
        out=BASE/ds/'cached_U_readout';bar=read(out/'PREDICTION_BARRIER.json');u=read(out/'RESOURCES.json')
        assert bar['status']=='sealed' and bar['cells']==48 and bar['model_restored'] and not bar['raw_GT_read']
        assert u['suffix_replays']==50 and u['live_pre_state_controls']==2 and u['backwards']==0
        assert u['new_expert_calls']==u['new_backbone_calls']==0
        for f,h in bar['files'].items():assert sha(out/f)==h;checks['U_receipt_sha256']+=1
        for c in [c for c in events if c['dataset']==ds]:
            a=old_a(c);new=checked(BASE/ds/'updates/predictions'/f'{prefix(c)}.pt')
            assert new['pre_sha']==a['pre_sha'] and new['persistent_post_sha']==a['post_sha'] and new['persistent_unchanged']
            checks['fixed_A_donor_state_chain']+=1
            if a['update_steps'][0].get('post_prediction') is None:
                uv=checked(out/'predictions'/f'{prefix(c)}.pt')
                assert uv['cached_first_post_state_sha256']==state_hash(a['update_steps'][0]['post_state'])
                assert uv['no_learning'] and not uv['raw_GT_read'] and not uv['GT_assisted_observation']
                assert uv['cached_gradient_arithmetic_bitwise'];old_U_readout+=1
                checks['U_post_state_exact']+=1
        latest_U_time=max(latest_U_time,bar['time'])
    assert old_U_readout==96 and read(BASE/'EXPERIMENT2_ROOT_AUDIT.json')['time']>latest_U_time
    assert read(BASE/'EXPERIMENT1_ROOT_AUDIT.json')['status']==read(BASE/'EXPERIMENT2_ROOT_AUDIT.json')['status']=='pass'
    assert not torch.cuda.is_initialized();verified(include_GT=True)
    result=dict(status='pass',checks=dict(checks),old_A_unchanged=True,production_unchanged=True,
        old_uniform_readouts_completed_without_learning=old_U_readout,
        final_metrics_after_all_cached_U_readouts=True,
        cached_U_readout_after_initial_CPU_partial=True,
        fixed_U_states_not_chosen_by_GT=True,
        raw_GT_not_read_by_model_workers=True,CUDA_initialized=False,
        max_new_expert_budget=288,actual_new_expert_calls=67,
        worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'FINAL_ROOT_AUDIT.json',result);print('FINAL_ROOT_AUDIT',result,flush=True)
if __name__=='__main__':run()

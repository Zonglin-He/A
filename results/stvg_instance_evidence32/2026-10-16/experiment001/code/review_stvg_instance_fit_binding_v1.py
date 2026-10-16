"""Complete independent locked-config/seed/common-state binding before GT."""
from pathlib import Path
import gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def main():
    import torch
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from scripts.audit_stvg_soft_ep_v1 import treeeq
    b=read(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='complete'and b['count']==32 and not b['GT_read']
    roster={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};count=0
    for tid in ('E0','E1'):
        td=BASE/'trials'/tid;cfg=read(td/'DESIGN_LOCK.json')['configs'];sb=read(td/'STUDENT_GLOBAL_BARRIER.json');assert read(td/'ROOT_READBACK.json')['status']=='pass'
        for f in sb['files']:
            assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);r=roster[d['case_id']]
            for m,fit in d['fits'].items():
                if fit is None:continue
                assert fit['config']==cfg[r['dataset']][m]and fit['seed_key']==stable_fit_key(r['dataset'],r['query_key_sha256']);count+=1
            if tid=='E1'and d['fits']['M0']is not None:assert treeeq(d['fits']['M0']['initial_state'],d['fits']['M2']['initial_state'])
            del d;gc.collect()
    put(BASE/'ROOT_LOCKED_FIT_BINDING.json',dict(status='pass',count=32,actual_fits_checked=count,GT_read=False,CPU_only=True,shared_E1_E2_reference_and_initial_state_exact=True,all_fit_configs_and_seed_keys_exact=True,prediction_barrier_sha256=sha(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json'),scorer_sha256=sha(ROOT/'scripts/score_stvg_instance_experiment32_v1.py'),advancement_rule_sha256=sha(BASE/'ADVANCEMENT_RULE_LOCK.json')))
    print('INSTANCE_LOCKED_FITS_PASS',count)
if __name__=='__main__':main()

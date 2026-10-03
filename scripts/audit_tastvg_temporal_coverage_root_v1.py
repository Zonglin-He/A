"""Immutable input, regenerated unlabeled pool and exact predecessor readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_coverage_common_v1 import *

def run():
    import torch,numpy as np,collections,functools
    from vg_tta.tastvg_temporal_candidate_coverage_v1 import allocate,choose
    from scripts.audit_tastvg_temporal_coverage_public_v1 import run as public_audit
    tick=time.monotonic();runtime=verify(inputs=True,labels=True);checks=collections.Counter()
    bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');exposure=read(BASE/'GT_EXPOSURE.json')
    assert bar['time']<exposure['new_GT_score_started']<read(BASE/'SCORE_ROOT_CHECKS.json')['time']
    assert bar['runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
    # Previous verified result rows are read only AFTER the new global seal.
    prior=ROOT/'results/tastvg_oracle_event5/2026-10-03/experiment1'
    @functools.lru_cache(maxsize=144)
    def head(ds,parent,condition):
        d,r,f=capture(dict(dataset=ds,parent=parent,condition=condition))
        return d['prediction']['logits'],d['records'],d['frame_ids']
    for ds in DATASETS:
        for split in SPLITS:
            newrows=read(PUBLIC/split/ds/'ROWS.json')
            oldrows=read(prior/split/ds/'ORACLE_ROWS.json')
            lookup={(r['order'],r['condition'],r['arrival']):r for r in oldrows}
            for row in newrows:
                previous=lookup[row['order'],row['condition'],row['arrival']]
                for f in ['A_v','A_t','GT_time_v']:
                    assert abs(row[f]-previous[f])<1e-12;checks['predecessor_scalar_parity']+=1
                if row['expert_scheduled']:
                    assert abs(row['old_oracle_v']-previous['temporal_oracle_v'])<1e-12
                    checks['predecessor_old_pool_oracle_parity']+=1
    for c in read(BASE/'COHORT.json')['cells']:
        f=BASE/c['dataset']/'predictions'/f'{prefix(c)}.json';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
        pred=read(f);a=acell(c);x=donor(c)
        assert pred['persistent_pre_sha']==a['pre_sha']==x['persistent_pre_sha']
        assert pred['persistent_post_sha']==a['post_sha']==x['persistent_post_sha']
        assert torch.equal(x['A']['boxes'],a['slow']['boxes']);checks['A_state_and_space_unchanged']+=1
        if c['scheduled']:
            logits,records,ids=head(c['dataset'],c['parent'],c['condition'])
            regenerated=allocate(a['slow']['indices'],ids,logits,records)
            assert regenerated==pred['new_candidates'];checks['unlabeled_allocation_reconstruction']+=1
            e,receipt=old.expert(c['dataset'],'temporal',c['parent'],c['condition'],c['pixel_sha256'])
            assert choose(regenerated,e['proposals'],e['proposal_confidence'])==pred['new_decision']
            checks['original_critic_reconstruction']+=1
        else:assert pred['new_indices']==a['final_indices'];checks['nonexpert_exact_A']+=1
    anonymous=public_audit(PUBLIC)
    write(BASE/'PUBLIC_AUDIT.json',anonymous)
    assert not torch.cuda.is_initialized();verify(labels=True)
    result=dict(status='pass',private_input_hashes_verified=len(runtime['inputs']),checks=dict(checks),
        generation_global_seal_precedes_GT=True,predecessor_readout_exact=True,old_A_unchanged=True,
        production_unchanged=True,all_queues_remain_paused=True,CUDA_initialized=False,
        worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'FINAL_ROOT_AUDIT.json',result);write(PUBLIC/'FINAL_ROOT_AUDIT.json',result)
    print('ROOT_AUDIT_PASS',result,flush=True)
if __name__=='__main__':run()

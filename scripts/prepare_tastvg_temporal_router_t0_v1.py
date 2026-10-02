"""Bind sealed A trajectories and actual expert inputs before diagnostic GT."""
from tastvg_temporal_router_common_v1 import *
import numpy as np


def run():
    assert not BASE.exists(),'Use the existing namespace; do not overwrite a run.'
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='completed'
    inputs={};cells={};inventory={}
    def bind(f):inputs[str(f.relative_to(ROOT))]=sha(f)
    bind(PRIOR/'FINAL_COMPLETION.json');bind(PRIOR/'GLOBAL_PREDICTION_BARRIER.json')
    bind(ROOT/'methods/CURRENT_METHOD.json')
    for ds in DATASETS:
        p=read(PRIOR/ds/'PLAN.json');bind(PRIOR/ds/'PLAN.json')
        bind(PRIOR/ds/'A'/'PREDICTION_BARRIER.json');bind(PRIOR/ds/'A'/'AUDIT.json')
        bind(PRIOR/ds/'A'/'ROWS.json');bind(PRIOR/ds/'A'/'SUPPORT.json')
        bind(POOL/ds/'CAPTURE_BARRIER.json')
        bind(POOL/ds/'GT_LABELS_search.json')
        assert read(PRIOR/ds/'A'/'AUDIT.json')['status']=='pass'
        records=[];ms=[];conf=[];invalid=duplicates=0
        for cond in p['conditions']:
            for order,seq in p['splits']['search']['orders'].items():
                for at,parent in enumerate(seq):
                    f=PRIOR/ds/'A'/'online'/cond/order/f'{at:05}.pt';rf=f.with_suffix('.json')
                    receipt=read(rf);assert sha(f)==receipt['sha256'];bind(f);bind(rf)
                    r=dict(parent=parent,condition=cond,order=order,arrival=at,
                           expert_scheduled=at%4==0,payload=str(f.relative_to(ROOT)))
                    if r['expert_scheduled']:
                        r['expert_receipts']={}
                        for stage in ['temporal','spatial']:
                            ef=POOL/ds/'experts'/stage/cond/f'{parent:05}.json';er=read(ef)
                            cf=POOL/ds/'experts'/er['cache'];assert sha(cf)==er['cache_sha256'];bind(ef);bind(cf)
                            r['expert_receipts'][stage]=str(ef.relative_to(ROOT))
                        e=load(POOL/ds/'experts'/read(ROOT/r['expert_receipts']['temporal'])['cache'])
                        proposals=np.asarray(e['proposals']).reshape(-1,2);c=np.asarray(e['proposal_confidence'])
                        assert e['GT_read'] is False and np.isfinite(proposals).all() and np.isfinite(c).all()
                        assert (proposals[:,1]>proposals[:,0]).all() and (c>=0).all()
                        ms.append(len(proposals));conf.extend(c.tolist())
                        duplicates+=len(proposals)-len(set(map(tuple,proposals.tolist())))
                    records.append(r)
        assert len(records)==384 and sum(r['expert_scheduled'] for r in records)==96
        cells[ds]=records
        inventory[ds]=dict(sources=32,queries=32,arrivals=384,expert_arrivals=96,
            unique_expert_sources=len({r['parent'] for r in records if r['expert_scheduled']}),
            proposal_counts=dict(min=min(ms),max=max(ms)),confidence_min=min(conf),confidence_max=max(conf),
            exact_duplicate_proposals=duplicates,invalid_proposals=invalid,params=p['params'],
            conditions=p['conditions'],orders=p['splits']['search']['orders'],
            source_checkpoint_state_sha256=read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'])
    write(BASE/'INPUT_BINDINGS.json',cells);write(BASE/'INVENTORY.json',inventory)
    pins=[str(f.relative_to(ROOT)) for f in [
        ROOT/'vg_tta/tastvg_temporal_router_t0_v1.py',
        *[ROOT/'scripts'/n for n in ['tastvg_temporal_router_common_v1.py',
          'test_tastvg_temporal_router_t0_v1.py','prepare_tastvg_temporal_router_t0_v1.py',
          'run_tastvg_temporal_router_t0_v1.py','score_tastvg_temporal_router_t0_v1.py',
          'audit_tastvg_temporal_router_public_v1.py']],
        ROOT/'vg_tta/tastvg_temporal_qualification_v1.py',
        ROOT/'vg_tta/tastvg_selected_rollout_v1.py',ROOT/'vg_tta/tastvg_event_support_v1.py',
        ROOT/'scripts/diagnose_tastvg_pipeline_cpu_v1.py',
        ROOT/'scripts/score_tastvg_best_quick_v1.py',
        ROOT/'protocols/tastvg_temporal_router_t0_v1.md',
        ROOT/'docs/tastvg_temporal_router_t0_v1/EXECUTION.md']]
    inputs[str((BASE/'INPUT_BINDINGS.json').relative_to(ROOT))]=sha(BASE/'INPUT_BINDINGS.json')
    inputs[str((BASE/'INVENTORY.json').relative_to(ROOT))]=sha(BASE/'INVENTORY.json')
    write(BASE/'RUNTIME_LOCK.json',dict(pins={n:sha(ROOT/n) for n in pins},inputs=inputs,
        GT_used=False,prior_GT_already_exposed=True,revision=1,time=time.time()))
    status(BASE/'STATUS.json',dict(status='ready_cpu_no_new_model_execution',done=0,total=768,GT_used=False))
    print('BOUND', {k:{x:v[x] for x in ['sources','arrivals','expert_arrivals','proposal_counts']} for k,v in inventory.items()},flush=True)


if __name__=='__main__':run()

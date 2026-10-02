"""Lock cohorts, hashes and two-round decision rules before any new inference."""
from scripts.tastvg_correction_views_common_v1 import *
def run():
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    inputs={};cells=[]
    for ds in DATASETS:
        p=read(POOL/ds/'PLAN.json');a=read(OLD/ds/'PLAN.json');p['params']=a['params']
        assert len(p['rows'])==48
        assert not {p['rows'][i]['source'] for i in range(32)}&{p['rows'][i]['source'] for i in range(32,48)}
        write(BASE/ds/'PLAN.json',p);inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))]=sha(BASE/ds/'PLAN.json')
        inputs[str((POOL/ds/'CAPTURE_BARRIER.json').relative_to(ROOT))]=sha(POOL/ds/'CAPTURE_BARRIER.json')
        for split in ['search','confirm']:
            for cond in p['conditions']:
                for order,seq in p['splits'][split]['orders'].items():
                    prev=None
                    for at,parent in enumerate(seq):
                        f=oldfile(ds,split,cond,order,at);x=oldcell(ds,split,cond,order,at)
                        assert x['parent']==parent and x['GT_read'] is False and x['expert_scheduled']==(at%4==0)
                        assert x['pre_sha']==state_hash(x['pre_state']) and x['post_sha']==state_hash(x['post_state'])
                        assert prev is None or prev==x['pre_sha'];prev=x['post_sha']
                        for step in x['update_steps']:
                            u=step['update']
                            if u:
                                assert u['lr']==p['params']['lr'] and u['teacher_temperature']==p['params']['teacher_temperature']
                                for n,v in step['pre_state'].items():
                                    torch.testing.assert_close(step['post_state'][n],v-p['params']['lr']*u['gradients'][n],rtol=2e-6,atol=1e-6)
                        inputs[str(f.relative_to(ROOT))]=sha(f);inputs[str(f.with_suffix('.json').relative_to(ROOT))]=sha(f.with_suffix('.json'))
                        cells.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,
                            old_payload=str(f.relative_to(ROOT)),pre_sha=x['pre_sha'],post_sha=x['post_sha'],pixel_sha256=x['pixel_sha256'],scheduled=at%4==0))
    write(BASE/'COHORT.json',dict(cells=cells,search_cells=768,confirm_cells=384,history_exposed=True,GT_read=False))
    inputs[str((BASE/'COHORT.json').relative_to(ROOT))]=sha(BASE/'COHORT.json')
    files=['scripts/tastvg_correction_views_common_v1.py','scripts/prepare_tastvg_correction_views_v1.py',
        'scripts/run_tastvg_correction_experts_v1.py','scripts/run_tastvg_current_correction_v1.py',
        'vg_tta/tastvg_current_correction_views_v1.py','protocols/tastvg_current_correction_views_v1.md']
    inherited=read(OLD/'RUNTIME_LOCK.json')['pins'];files+=list(inherited)
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in sorted(set(files))},inputs=inputs,
        producer='isolated two-round current correction',persistent='Uniform A exact saved trajectory',
        parameters=1792,expert_fraction=.25,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_live_controls',GT_read=False,time=time.time()))
    archive('科学配置与两集开发/确认名单和原A完整参数链已锁定，尚无新预测')
if __name__=='__main__':run()

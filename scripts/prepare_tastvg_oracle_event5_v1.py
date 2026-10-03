"""Freeze old trajectories, GT-derived observation positions and code pins on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.tastvg_oracle_event5_common_v1 import *
def run():
    import torch,collections
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_ur_write_decomposition_v1 import sgd
    from vg_tta.tastvg_oracle_event5_v1 import event_positions
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(OLD/'FINAL_COMPLETION.json')['status']=='completed_and_verified_publication'
    cells=read(OLD/'COHORT.json')['cells'];assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    metadata={str((OLD/'FINAL_COMPLETION.json').relative_to(ROOT)):sha(OLD/'FINAL_COMPLETION.json'),
        'methods/CURRENT_METHOD.json':sha(ROOT/'methods/CURRENT_METHOD.json')}
    inputs={};events=[];counts=collections.Counter()
    for ds in DATASETS:
        p=plan(ds);assert len(p['rows'])==48
        for path in [OLD/ds/'PLAN.json',POOL/ds/'CAPTURE_BARRIER.json']:
            metadata[str(path.relative_to(ROOT))]=sha(path)
        for split in SPLITS:
            gf=POOL/ds/f'GT_LABELS_{split}.json';labels=read(gf)
            metadata[str(gf.relative_to(ROOT))]=sha(gf)
            assert len(p['splits'][split]['orders']['order1'])==(32 if split=='search' else 16)
            for c in [c for c in cells if c['dataset']==ds and c['split']==split]:
                one=cached_payload(c);two=cached_payload(c,'round2');a=old_a(c)
                assert torch.equal(one['A']['boxes'],two['A']['boxes']) and one['A']['indices']==two['A']['indices']
                assert one['persistent_pre_sha']==two['persistent_pre_sha']==a['pre_sha']
                assert one['persistent_post_sha']==two['persistent_post_sha']==a['post_sha']
                for stage in ['round1','round2']:
                    f=payload_path(c,stage);inputs[str(f.relative_to(ROOT))]=sha(f)
                    inputs[str(f.with_suffix('.json').relative_to(ROOT))]=sha(f.with_suffix('.json'))
                f=ROOT/c['old_payload'];inputs[str(f.relative_to(ROOT))]=sha(f)
                counts['A_arrivals']+=1
                if not c['scheduled']:continue
                assert state_hash(one['pre_state'])==a['pre_sha']
                for b1,b2 in zip(one['probes'],two['probes']):assert torch.equal(b1['boxes'],b2['boxes'])
                assert len(one['probes'])==len(two['probes'])==9 and torch.equal(one['probes'][0]['boxes'],one['A']['boxes'])
                q=sgd(one['pre_state'],one['gradients']['U'],p['params']['lr']) if one['rewards']['U'] is not None else one['pre_state']
                assert state_hash(q)==state_hash(a['update_steps'][0]['post_state']);counts['U_one_step_bitwise_reuse']+=1
                g=labels[str(c['parent'])];selection=event_positions(p['rows'][c['parent']]['frame_ids'],g['span'])
                e=dict(c,**selection,GT_assisted=True)
                events.append(e);counts['event_eligible' if e['eligible'] else 'event_unsupported']+=1
                counts[ds+'_'+split+'_eligible' if e['eligible'] else ds+'_'+split+'_unsupported']+=1
                for branch in ['U','R','U2']:
                    ev,r=cached_evidence(c,branch);inputs[r['cache']]=r['cache_sha256']
                    if ev['positions']==e['positions']:counts['matching_'+branch+'_positions']+=1
    write(BASE/'COHORT.json',dict(cells=cells,counts=dict(counts),historically_exposed=True,source_counts_each=[32,16],time=time.time()))
    write(BASE/'EVENT_PLAN.json',dict(events=events,counts=dict(counts),quantiles=[0,.25,.5,.75,1],
        GT_read=True,scope='GT_oracle observation positions only',outside_event_fill=False,time=time.time()))
    metadata[str((BASE/'COHORT.json').relative_to(ROOT))]=sha(BASE/'COHORT.json')
    metadata[str((BASE/'EVENT_PLAN.json').relative_to(ROOT))]=sha(BASE/'EVENT_PLAN.json')
    code=['protocols/tastvg_oracle_event5_v1.md','scripts/tastvg_oracle_event5_common_v1.py',
        'scripts/prepare_tastvg_oracle_event5_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
        'scripts/test_tastvg_oracle_event5_v1.py','scripts/score_tastvg_residual_oracles_v1.py',
        'scripts/run_tastvg_gt_event5_expert_v1.py','scripts/run_tastvg_gt_event5_update_v1.py',
        'scripts/continue_tastvg_gt_event5_v1.py','scripts/score_tastvg_gt_event5_v1.py']
    oldpins=previous.verify()['pins']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in sorted(set(code+list(oldpins)))},
        protected_metadata=metadata,inputs=inputs,GT_observation_intervention=True,parameters=1792,
        max_logical_requests=288,max_new_expert_calls=288,new_backbone_calls=0,physical_GT_only_in_CPU_plan=True,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_CPU_oracle',counts=dict(counts),time=time.time()))
    assert not torch.cuda.is_initialized()
    archive('CPU名单、A状态链与GT-event分位取帧规则已锁，1152固定到达及288候选位置，尚无新增专家或预测')
    print('PREPARED',dict(counts),flush=True)
if __name__=='__main__':run()

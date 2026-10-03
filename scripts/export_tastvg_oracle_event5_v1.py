"""Anonymous public provenance and allowlist; restricted assets are never exported."""
from scripts.tastvg_oracle_event5_common_v1 import *
def run():
    verified(include_GT=True)
    assert read(BASE/'EXPERIMENT1_ROOT_AUDIT.json')['status']=='pass' and read(BASE/'EXPERIMENT2_ROOT_AUDIT.json')['status']=='pass'
    cohort=read(BASE/'COHORT.json');events=read(BASE/'EVENT_PLAN.json')['events'];runtime=read(BASE/'RUNTIME_LOCK.json')
    call=read(BASE/'EXPERT_RESOURCES.json');updates={d:read(BASE/d/'updates/RESOURCES.json') for d in DATASETS}
    write(PUBLIC/'CONFIG.json',dict(version='tastvg_oracle_event5_v1',authorized_date='2026-10-03',
        predecessor_commit='3b3ebd297f621667ce03f1291d130b6fd71236db',
        params={d:plan(d)['params'] for d in DATASETS},datasets=DATASETS,
        development_sources_each=32,confirmation_sources_each=16,queries_per_source=1,orders=2,
        conditions=plan('vidstg')['conditions'],history_exposed=True,GT_assisted_diagnostic=True,
        persistent_A_frozen=True,no_parameter_selection=True,source_bootstrap_draws=10000,source_bootstrap_seed=20261003,
        full_arrivals=1152,candidate_arrivals=288,candidate_support=[8,9],combination_count=20736,
        event5_logical_requests=288,event5_eligible=call['eligible'],event5_unsupported=call['unsupported'],
        event5_corruption_eligible=sum(c['eligible'] and c['condition']!='clean' for c in events),
        event5_clean_eligible=sum(c['eligible'] and c['condition']=='clean' for c in events),
        quantiles=[0,.25,.5,.75,1],no_outside_event_fill=True,temporary_SGD_steps=1,temporary_lifespan='one_query',
        production_unchanged=True,all_old_queues_remain_paused=True))
    write(PUBLIC/'COHORT_CONFIG.json',dict(anonymous_source_ordinals_by_dataset={d:{s:sorted({c['parent'] for c in cohort['cells'] if c['dataset']==d and c['split']==s}) for s in SPLITS} for d in DATASETS},
        original_cohort_sha256=sha(OLD/'COHORT.json'),original_plan_sha256={d:sha(OLD/d/'PLAN.json') for d in DATASETS},
        GT_event_plan_sha256=sha(BASE/'EVENT_PLAN.json'),support_exclusion_counts=cohort['counts'],
        sampling='original_Paper48_observed_grid',confirmation='existing_source_disjoint_within_batch_historical_exposure'))
    import re
    revisions=sorted((BASE/'revisions').glob('*.json'),key=lambda f:int(re.search(r'_(\d+)\.json$',f.name).group(1)))
    effective=dict(runtime['pins'])
    for f in revisions:effective.update(read(f).get('pin_overrides',{}))
    code=list(effective)+['scripts/report_tastvg_oracle_event5_v1.py','scripts/audit_tastvg_oracle_event5_public_v1.py','scripts/export_tastvg_oracle_event5_v1.py',
        'scripts/audit_tastvg_oracle_event5_root_v1.py','scripts/derive_tastvg_oracle_event5_v1.py']
    write(PUBLIC/'RUNTIME_PROVENANCE.json',dict(pins={f:sha(ROOT/f) for f in sorted(set(code))},
        original_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),protected_input_file_count=len(runtime['inputs']),
        effective_runtime_pins=effective,engineering_revisions=[dict(name=f.name,record=read(f)) for f in revisions],
        exact_inputs_private_for_media_annotation_weight_tensor_exclusions=True,
        checkpoint_state_sha256={d:read(POOL/d/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'] for d in DATASETS},
        CURRENT_METHOD_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        expert_precision='Sa2VA4B_BF16_eager_first_SEG_greedy_unchanged_official_interface',
        no_expert_pseudo_box_as_output=True,GT_time_never_fed_to_decoder=True))
    barriers={d:read(BASE/d/'updates/PREDICTION_BARRIER.json') for d in DATASETS}
    write(PUBLIC/'BARRIERS_AND_CONTROLS.json',dict(experiment1=read(BASE/'EXPERIMENT1_ROOT_AUDIT.json'),
        experiment2=read(BASE/'EXPERIMENT2_ROOT_AUDIT.json'),global_intervention=read(BASE/'GLOBAL_INTERVENTION_BARRIER.json'),
        dataset_intervention_barriers={d:{k:v for k,v in b.items() if k!='files'} for d,b in barriers.items()},
        old_U_SGD_bitwise_reused_donors=cohort['counts']['U_one_step_bitwise_reuse'],
        new_uniform_VJP_bitwise_controls=sum(u['uniform_gradient_controls'] for u in updates.values()),
        omitted_U_post_tubes_reconstructed=96,cached_U_pre_state_prediction_bitwise_controls=4,
        cached_U_readout_barriers={d:{k:v for k,v in read(BASE/d/'cached_U_readout/PREDICTION_BARRIER.json').items() if k!='files'} for d in DATASETS},
        final_root_audit=read(BASE/'FINAL_ROOT_AUDIT.json'),
        model_restored=all(b['model_restored'] for b in barriers.values()),
        GT_sampling_read_pre_inference_explicitly_authorized=True,post_intervention_metric_read_after_global_seal=True))
    ledger=BASE/'EXPERT_ATTEMPTS.jsonl'
    import json
    attempts=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    write(PUBLIC/'ACTUAL_NEW_CALLS.json',attempts)
    recoveries=[]
    for folder in sorted((BASE/'recovery').glob('*')):
        recoveries.append(dict(name=folder.name,files=[p.name for p in folder.iterdir()],preserved=True))
    write(PUBLIC/'ENGINEERING_HISTORY.json',dict(recoveries=recoveries,science_changed=False))
    files=['protocols/tastvg_oracle_event5_v1.md','docs/tastvg_oracle_event5_v1/EXECUTION.md','docs/TA_ORACLE_EVENT5_REVIEW.md',
        'vg_tta/tastvg_oracle_event5_v1.py']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*oracle_event5*v1.py')]
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*gt_event5*v1.py')]
    files += ['scripts/score_tastvg_residual_oracles_v1.py','scripts/replay_tastvg_cached_uniform_readout_v1.py']
    files += [str(p.relative_to(ROOT)) for p in PUBLIC.rglob('*') if p.is_file()]
    files=sorted(set(files));assert all(not any(k in f for k in ['GT_LABELS','EVENT_PLAN.json','.pt','checkpoints/','/expert_cache/']) for f in files)
    write(BASE/'PUBLIC_MANIFEST.json',dict(files={f:dict(sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files},
        file_count=len(files),bytes=sum((ROOT/f).stat().st_size for f in files),time=time.time()))
    print('PUBLIC_ALLOWLIST',len(files),'files',flush=True)
if __name__=='__main__':run()

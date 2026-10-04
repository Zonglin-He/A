"""Finite CPU teacher-only study: prepare -> guarded selection -> GT scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, time, json, hashlib, subprocess, traceback
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha, status
from scripts.teacher_purification_import_v1 import core
select, truth_iou, add_metrics, summaries, read_guard = core.select, core.truth_iou, core.add_metrics, core.summaries, core.read_guard
BASE = ROOT/'artifacts/tastvg_teacher_purification_v1'
PUB = ROOT/'results/tastvg_teacher_purification/2026-10-04'
R2 = ROOT/'artifacts/tastvg_dta_expert_r2_v1'
R2PUB = ROOT/'results/tastvg_dta_expert_r2/2026-10-04'
POOL = ROOT/'artifacts/tastvg_extended_sensitivity_v3'
OWN = ['vg_tta/tastvg_teacher_purification_v1.py', 'scripts/run_tastvg_teacher_purification_v1.py',
       'scripts/teacher_purification_import_v1.py',
       'scripts/test_tastvg_teacher_purification_v1.py', 'scripts/audit_tastvg_teacher_purification_v1.py',
       'scripts/report_tastvg_teacher_purification_v1.py', 'protocols/tastvg_teacher_purification_v1.md',
       'docs/tastvg_teacher_purification_v1/EXECUTION.md', 'scripts/decota_matrix_common_v1.py',
       'scripts/audit_tastvg_dta_oracle_r1_v1.py']
def key(c): return '/'.join(str(c[k]) for k in ['dataset', 'split', 'condition', 'order', 'arrival'])
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def st(value, **kw): status(BASE/'STATUS.json', dict(status=value, pid=os.getpid(), time=time.time(), **kw))
def verify(postseal=False):
    lock=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**lock['code'], **lock['inputs'], **(lock['post_seal_inputs'] if postseal else {})}.items():
        assert sha(ROOT/p)==h, p
    return lock


def prepare():
    sys.addaudithook(read_guard)
    assert not (BASE/'RUNTIME_LOCK.json').exists(), 'Existing prepared run cannot restart'
    BASE.mkdir(parents=True, exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)
    final=read(R2/'FINAL_COMPLETION.json'); assert final['status']=='verified_complete'
    assert final['GitHub_commit']=='324b3357dbd67b435d1fb507ae516a44ecf53961'
    cfg=read(R2PUB/'CONFIG.json'); cohort=read(R2/'COHORT.json'); support=read(R2/'EXPERT_SUPPORT.json')
    cells=[c for c in cohort['cells'] if c['scheduled']]
    assert len(cohort['cells'])==1152 and len(cells)==len(support)==288
    inputs={}
    def pin(p): inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in [R2/'RUNTIME_LOCK.json', R2/'COHORT.json', R2/'EXPERT_SUPPORT.json', R2/'FINAL_COMPLETION.json', R2PUB/'CONFIG.json',
              ROOT/'docs/TA_TEMPORAL_ROUTER_REVIEW.md', ROOT/'protocols/tastvg_temporal_router_t0_v1.md', ROOT/'vg_tta/tastvg_temporal_router_t0_v1.py',
              ROOT/'methods/CURRENT_METHOD.json']: pin(p)
    assert inputs['methods/CURRENT_METHOD.json']==cfg['production_method_sha256']
    cache_files={}; counts=[]
    for c in cells:
        z=support[key(c)]; chosen=select(z['proposals'], z['confidence']); counts.append(chosen['proposal_count'])
        assert chosen['Confidence_index']==z['deploy']['index'] and not z['deploy']['GT_used']
        assert z['pixel_sha256']==c['pixel_sha256']
        receipt=POOL/c['dataset']/'experts/temporal'/c['condition']/f'{c["parent"]:05}.json'
        e=read(receipt); pin(receipt)
        assert not e['GT_read'] and e['pixel_sha256']==c['pixel_sha256'] and e['cache_sha256']==z['expert_cache_sha256']
        f=POOL/c['dataset']/'experts'/e['cache']
        assert f.stem==z['input_sha256']; cache_files[str(f.relative_to(ROOT))]=z['expert_cache_sha256']
    for path,h in cache_files.items(): assert sha(ROOT/path)==h, path
    inputs.update(cache_files)
    assert len(cache_files)==cfg['unique_expert_cache_files']==235 and [min(counts),max(counts)]==[26,312]
    manifest=read(R2/'PUBLIC_EXPORT_MANIFEST.json')
    oldbaseline={f['path']:f['sha256'] for f in manifest['files'] if f['path'].endswith(('DEPLOY_SELECTION.json','TEACHER_SELECTION_SCORED.json'))}
    assert len(oldbaseline)==2
    config=dict(version='tastvg_teacher_purification_v1', question='Can confidence-free peer overlap identify useful raw expert proposals?',
        confidence='first raw confidence argmax', consensus='mean peer interval IoU excluding self; confidence-free first exact argmax',
        raw_duplicates_and_order_preserved=True, fractional_endpoints_unclipped=True, singleton='sole index0, peer agreement undefined and score0',
        empty_invalid='engineering error, no fallback', oracle='first continuous physical GT-tIoU argmax, diagnostic only after selection seal',
        tIoU_metric='continuous half-open physical-frame coordinates; same R2 teacher support metric',
        success='tIoU>.5 strict', severe_wrong_event='tIoU==0 disjoint or touching intervals',
        raw_proposal_count_range=[min(counts),max(counts)], unique_expert_cache_files=len(cache_files),
        parent_design_arrivals=1152, scheduled_teacher_cells=288, corrupt_teacher_cells=240, clean_teacher_cells=48,
        nonexpert_cells_evaluated=0, target_design_sources=cfg['target_design_sources'], target_expert_sources=cfg['target_expert_sources'],
        history_exposure=True, source_queries_per_source=1, orders=['order1','order2'], conditions=cfg['conditions'],
        checkpoints_provenance_only=cfg['checkpoints'], sampling=cfg['sampling'], schedule_fraction=.25,
        production_method_sha256=cfg['production_method_sha256'], spatial_A_fixed=True, persistent_state_unchanged=True,
        R2_commit=final['GitHub_commit'], R2b_commit='866ec35bd0dc0c41b938e656e9c9e430d18c18af',
        prior_T0='confidence times overlap routing/native critic NO_GO; new pure raw medoid teacher endpoint',
        bootstrap_draws=10000, bootstrap_seed=20261004, aggregation='equal source mean of repeated condition/order cells; paired source bootstrap',
        continuation='all four corrupt panels Consensus-Confidence lower95CI>0; no adaptation in this audit',
        target_GT_used_in_unlabeled_selection=False, parameter_tuning=False, confidence_weighting=False, threshold=False, clustering=False,
        deduplication=False, new_view=False, model_loads=0, new_GPU_calls=0, new_backbone_calls=0, new_expert_calls=0, backward_calls=0,
        head_updates=0, spatial_updates=0, temporal_persistence_writes=0, R2c_started=False, production_promoted=False)
    write(BASE/'COHORT.json', dict(cells=cells)); pin(BASE/'COHORT.json')
    write(PUB/'CONFIG.json', config); pin(PUB/'CONFIG.json')
    code={p:sha(ROOT/p) for p in OWN}
    write(BASE/'RUNTIME_LOCK.json',dict(code=code, inputs=inputs,
        post_seal_inputs={**read(R2/'RUNTIME_LOCK.json')['oracle_label_inputs'], **oldbaseline}, time=time.time()))
    write(PUB/'CODE_BINDINGS.json', code)
    write(BASE/'PREPARATION.json',dict(status='pass', GT_read=False, scored_control_read=False, cache_files_verified=235,
        parent_cells=1152, scheduled_cells=288, invalid_supports=0, singleton_supports=0, raw_range=[26,312], time=time.time()))
    st('prepared_pending_selection', GT_read=False)


def selections():
    sys.addaudithook(read_guard); verify(); tick=time.monotonic()
    cells=read(BASE/'COHORT.json')['cells']; support=read(R2/'EXPERT_SUPPORT.json'); out=[]
    for c in cells:
        k=key(c); z=support[k]; r=select(z['proposals'],z['confidence'])
        r.update(cell_key=k, dataset=c['dataset'], split=c['split'], source_id=c['parent'], condition=c['condition'],
            order=c['order'], arrival=c['arrival'], raw_proposals_sha256=digest(z['proposals']), confidence_sha256=digest(z['confidence']),
            expert_input_sha256=z['input_sha256'], expert_cache_sha256=z['expert_cache_sha256'], pixel_sha256=c['pixel_sha256'],
            A_state_pre_sha256=c['pre_sha'], A_state_post_sha256=c['post_sha'])
        out.append(r)
    assert len(out)==288 and all(not x['GT_used'] for x in out)
    write(PUB/'UNLABELED_SELECTION.json',out)
    barrier=dict(status='sealed', cells=288, GT_read=False, GT_guard_active=True,
        selection_sha256=sha(PUB/'UNLABELED_SELECTION.json'), config_sha256=sha(PUB/'CONFIG.json'),
        code_bindings_sha256=sha(PUB/'CODE_BINDINGS.json'), runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        inputs_digest=digest(read(BASE/'RUNTIME_LOCK.json')['inputs']), selection_CPU_wall_seconds=time.monotonic()-tick,
        new_GPU_calls=0, new_model_calls=0, backward_calls=0, time=time.time(), pid=os.getpid())
    write(BASE/'UNLABELED_SELECTION_BARRIER.json',barrier); write(PUB/'UNLABELED_SELECTION_BARRIER.json',barrier)
    st('selection_sealed_pending_score', GT_read=False, cells=288)


def score():
    barrier=read(BASE/'UNLABELED_SELECTION_BARRIER.json'); assert barrier['status']=='sealed' and not barrier['GT_read']
    assert sha(PUB/'UNLABELED_SELECTION.json')==barrier['selection_sha256']; verify(postseal=True); tick=time.monotonic()
    support=read(R2/'EXPERT_SUPPORT.json'); selections=read(PUB/'UNLABELED_SELECTION.json')
    oldsel={r['cell_key']:r for r in read(R2PUB/'DEPLOY_SELECTION.json')}
    oldscored={(r['cell_key'],r['arm']):r for r in read(R2PUB/'TEACHER_SELECTION_SCORED.json')}
    labels={(ds,sp):read(POOL/ds/f'GT_LABELS_{sp}.json') for ds in ['vidstg','hc2'] for sp in ['search','confirm']}
    rows=[]; maxerror=0.
    for sel in selections:
        k=sel['cell_key']; values=truth_iou(support[k]['proposals'],labels[(sel['dataset'],sel['split'])][str(sel['source_id'])]['span'])
        r=dict(sel, proposal_GT_tIoU=values.tolist(), Oracle_index=int(np.argmax(values)), Oracle_GT_used=True,
               Confidence_GT_used=False, Consensus_GT_used=False, evaluation_GT_read_after_seal=True)
        add_metrics(r)
        assert r['Confidence_index']==oldsel[k]['selected_index']==oldscored[(k,'EDeploy')]['selected_index']
        assert r['Oracle_index']==oldscored[(k,'EOracle')]['selected_index']
        for arm,oldarm in [('Confidence','EDeploy'),('Oracle','EOracle')]:
            error=abs(r[arm+'_t']-oldscored[(k,oldarm)]['continuous_teacher_tIoU']); assert error<1e-12; maxerror=max(maxerror,error)
        rows.append(r)
    write(PUB/'ROWS.json',rows); summary=summaries(rows); write(PUB/'SUMMARY.json',summary)
    panels={ds+'/'+sp:summary[ds][sp]['corrupt']['metrics']['Consensus_minus_Confidence_t'] for ds in ['vidstg','hc2'] for sp in ['search','confirm']}
    all_positive=all(m['ci95'][0]>0 for m in panels.values())
    decision=dict(status='GO_TEACHER_ONLY' if all_positive else 'NO_GO', criterion='all four corrupt panel paired lower95CI>0',
        panel_pass={p:m['ci95'][0]>0 for p,m in panels.items()}, teacher_primary=panels, adaptation_started=False, R2c_started=False,
        production_promoted=False, no_universal_impossibility_claim=True,
        scope='teacher quality only, no adaptation or final tube metric',time=time.time())
    write(PUB/'DECISION.json',decision)
    cases=[]
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            q=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
            for tag,chosen in [('worst', sorted(q,key=lambda r:r['Consensus_minus_Confidence_t'])[:3]),
                               ('best',sorted(q,key=lambda r:-r['Consensus_minus_Confidence_t'])[:3])]:
                for row in chosen:
                    take=['cell_key','dataset','split','source_id','condition','order','proposal_count','unique_intervals',
                          'Confidence_index','Consensus_index','Oracle_index','Confidence_t','Consensus_t','Oracle_t',
                          'Consensus_minus_Confidence_t','selected_Consensus_agreement','selected_Confidence_agreement']
                    cases.append(dict(case=tag,**{x:row[x] for x in take}))
    write(PUB/'CASES.json',cases)
    resource=dict(selection_CPU_wall_seconds=barrier['selection_CPU_wall_seconds'], scoring_CPU_wall_seconds=time.monotonic()-tick,
        timing_excludes_preparation_cache_hashes_root_audit_report_publication=True, teacher_cells=288, scored_nonexpert_cells=0,
        new_GPU_calls=0, model_loads=0, new_backbone_calls=0, new_expert_calls=0, backward_calls=0, head_updates=0,
        spatial_updates=0, temporal_persistence_writes=0, old_Confidence_Oracle_max_error=maxerror,
        loaded_frameworks=[x for x in ['torch','tensorflow','jax'] if x in sys.modules], time=time.time())
    assert not resource['loaded_frameworks']; write(PUB/'RESOURCES.json',resource)
    done=dict(status='completed_pending_root_audit_publication', rows=288, GT_read_after_seal=True, time=time.time(),
        selection_barrier_sha256=sha(PUB/'UNLABELED_SELECTION_BARRIER.json'), rows_sha256=sha(PUB/'ROWS.json'),
        summary_sha256=sha(PUB/'SUMMARY.json'), decision_sha256=sha(PUB/'DECISION.json'))
    write(BASE/'SCORE_COMPLETION.json',done); write(PUB/'SCORE_COMPLETION.json',done)
    st('completed_pending_root_audit_publication', teacher_cells=288, decision=decision['status'])


def main():
    if len(sys.argv)>1: return {'prepare':prepare, 'select':selections, 'score':score}[sys.argv[1]]()
    BASE.mkdir(parents=True,exist_ok=True)
    for phase in ['prepare','select','score']:
        with (BASE/(phase.upper()+'.log')).open('x') as log:
            worker=subprocess.run([sys.executable,'-B',__file__,phase], stdout=log, stderr=subprocess.STDOUT)
        if worker.returncode:
            st('failed',phase=phase,returncode=worker.returncode); raise RuntimeError('Purification phase failed: '+phase)
        print('TEACHER_PURIFICATION_STAGE_COMPLETED',phase,flush=True)
if __name__=='__main__':
    try: main()
    except Exception: traceback.print_exc(); raise

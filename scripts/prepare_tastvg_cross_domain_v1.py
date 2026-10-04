"""Metadata-only cohort selection and frozen CPU subject parsing."""
import sys, hashlib, collections, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *

def query_hash(q):
    return hashlib.sha256((str(q['index']) + '|' + q['caption'] + '|'
                           + str(q['original_video_id'])).encode()).hexdigest()

def prepare():
    assert not (BASE / 'RUNTIME_LOCK.json').exists()
    cohort = dict(version='tastvg_cross_domain_qualification_v1', condition='clean',
                  historical_exposure=True, GT_used_for_selection=False, directions={})
    specs = [('vid_to_hc2', 'dense_expansion_evaluation_v1', 'vid_to_hc', 'vidstg', 'hc2', 413, 135),
             ('hc2_to_vid', 'official_dense_evaluation_v2', 'hc_to_vid', 'hc2', 'vidstg', 707, 384)]
    allowed = {'caption','index','width','height','fps','source','original_video_id',
               'video_path','video_sha256','frame_ids','frame_count','duration',
               'start_frame','end_frame','kind'}
    labels = {}; metadata = {}
    for direction, parent, group, source, target, nq, ns in specs:
        lf = ROOT / 'artifacts' / parent / 'lock.json'; spec = read(lf)['groups'][group]
        qs = spec['queries']; assert len(qs) == nq and len({q['source'] for q in qs}) == ns
        bysource = collections.defaultdict(list)
        for q in qs: bysource[q['source']].append(q)
        selected = [min(bysource[s], key=query_hash) for s in sorted(bysource)]
        rows = []
        for i, q in enumerate(selected):
            inp = {k:q[k] for k in allowed if k in q}
            inp['caption'] = inp['caption'].lower()  # Both native TA dataset loaders.
            if target == 'hc2':
                inp.update(fps=q['frame_count'] / 20., duration=20., start_frame=0,
                           end_frame=q['frame_count'])
            ids = list(inp['frame_ids']); assert ids == sorted(set(ids)) and len(ids) >= 2
            assert sha(inp['video_path']) == inp['video_sha256']
            rows.append(dict(parent=i, source=q['source'], key=query_hash(q), input=inp,
                frame_ids=ids, parent_index=q['index'], qtype=q.get('qtype',q.get('query_type')),
                historical_exposure=True, query_hash=query_hash(q),
                parent_caption_sha256=hashlib.sha256(q['caption'].encode()).hexdigest()))
        assert len({r['input']['video_sha256'] for r in rows}) == ns
        orders = {f'order{k}': sorted(range(ns), key=lambda i:hashlib.sha256(
            (str(seed) + '|' + rows[i]['source']).encode()).hexdigest())
            for k, seed in enumerate([2026100401, 2026100402, 2026100403])}
        needed = sorted({seq[a] for seq in orders.values() for a in range(0,ns,4)})
        out = BASE / direction
        plan = dict(direction=direction, source_dataset=source, target_dataset=target,
            rows=rows, params=BUNDLES[source], conditions=['clean'], orders=orders,
            order_seeds=[2026100401,2026100402,2026100403], total=3*ns, queries=ns,
            sources=ns, expert_scheduled_total=3*((ns+3)//4), expert_needed=needed,
            original_sampling_preserved=True, parent_lock=str(lf), parent_lock_sha256=sha(lf),
            parent_queries=nq, source_checkpoint=CHECKPOINTS[source],
            source_checkpoint_sha256=sha(ROOT/CHECKPOINTS[source]),
            target_checkpoint=CHECKPOINTS[target], target_checkpoint_sha256=sha(ROOT/CHECKPOINTS[target]),
            GT_read=False)
        write(out/'PLAN.json', plan)
        write(out/'EXPERT_PLAN.json', dict(rows=rows, conditions=['clean'], expert_needed=needed,
            conditions_by_parent={str(i):['clean'] for i in needed}, total=len(needed),
            observation_rule='Uniform5 unchanged; UVTG2Hz unchanged', GT_read=False))
        labels[direction] = {k:spec[k] for k in ['annotation','annotation_sha256',
            'scorer_sidecar','scorer_sidecar_sha256'] if k in spec}
        for k,v in labels[direction].items():
            if k in ['annotation','scorer_sidecar']: assert sha(v) == labels[direction][k+'_sha256']
        cohort['directions'][direction] = dict(queries=ns,sources=ns,total=3*ns,
            scheduled=plan['expert_scheduled_total'],unique_expert_inputs=len(needed),
            parent_queries=nq, parent_lock_sha256=sha(lf), params=BUNDLES[source])
        for name in ['PLAN.json','EXPERT_PLAN.json']:
            metadata[str((out/name).relative_to(BASE))] = sha(out/name)
        print('COHORT', direction, ns, 3*ns, 'scheduled',plan['expert_scheduled_total'],
              'unique_experts',len(needed), flush=True)
    write(BASE/'COHORT.json',cohort); write(BASE/'GT_INPUT_LOCK.json',labels)
    metadata['COHORT.json'] = sha(BASE/'COHORT.json')
    write(BASE/'DESIGN_LOCK.json', dict(query_rule='min sha256(index|caption|original_video_id) per source',
        order_rule='sha256(seed|source) ascending', model='TA-STVG', methods=['Frozen','Fast-only',
        'Spatial-only','Full A','Target-trained'], primary='Full A-Frozen vIoU paired source-macro',
        persistence='Full A-Fast-only, especially nonexpert', bootstrap=10000, bootstrap_seed=20261004,
        expert_schedule='arrival % 4 == 0', total=1557, scheduled=390,
        source_bundle_tuning_provenance='Historically exposed source development; no target retuning this run',
        promote=False, GT_before_prediction_seal=False, time=time.time()))
    metadata['DESIGN_LOCK.json'] = sha(BASE/'DESIGN_LOCK.json')
    files = ['scripts/tastvg_cross_domain_common_v1.py','scripts/prepare_tastvg_cross_domain_v1.py',
        'scripts/run_tastvg_cross_domain_experts_v1.py','scripts/run_tastvg_cross_domain_v1.py',
        'scripts/continue_tastvg_cross_domain_v1.py','protocols/tastvg_cross_domain_qualification_v1.md',
        'scripts/tastvg_best_quick_expert_worker_v1.py','vg_tta/tastvg_best_full_method_v1.py',
        'scripts/run_tastvg_paper48_p5_online_v1.py','vg_tta/tastvg_native_spatial_rollout_s05_v1.py',
        'vg_tta/tastvg_spatial_rank_s11_v1.py','vg_tta/tastvg_spatial_critic_s06_v1.py',
        'vg_tta/tastvg_corruption_c0c1_v1.py','vg_tta/tastvg_temporal_qualification_v1.py',
        'scripts/run_tastvg_evidence_vulnerability_v2.py','scripts/run_spatial_regression_alignment_v1.py',
        'methods/decota_final_simplified_v1/backbone.py','methods/decota_final_simplified_v1/objectives.py',
        'vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},metadata=metadata,
        CURRENT_METHOD_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),time=time.time(),GT_read=False))
    status(BASE/'STATUS.json',dict(status='locked_pending_subjects_and_predecessor_gpu',GT_read=False,
        predictions=0,time=time.time()))
    archive('名单与科学配置锁定，零预测；排在现有 correction-scope GPU 队列后串行')

def subjects(direction):
    p=verify(direction); out=BASE/direction
    if (out/'SUBJECT_BARRIER.json').exists(): return
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from vg_tta.foreground_runtime import QuerySubjectParser
    parser=QuerySubjectParser(ROOT/'.cache/stanza'); files={}
    for row in p['rows']:
        f=out/'subjects'/f"{row['parent']:05}.json"
        if not f.exists(): write(f,dict(parent=row['parent'],caption_sha256=hashlib.sha256(
            row['input']['caption'].encode()).hexdigest(),parses=dict(subject=parser(row['input']['caption'])),GT_read=False))
        files[f.name]=sha(f)
    write(out/'SUBJECT_BARRIER.json',dict(count=len(files),files=files,GT_read=False,time=time.time()))
    print('SUBJECTS',direction,len(files),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    else: subjects(sys.argv[1])

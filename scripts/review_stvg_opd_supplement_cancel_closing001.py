"""Rebind saved complete evidence and cancelled-baseline custody; no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import copy
import gzip
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write as immutable_write, sha
from scripts.audit_stvg_opd_LN_complete_public_v1 import independent, compare
from scripts.audit_stvg_opd_supplement_cancel_public_v1 import REL, FIELDS, ABS_FIELDS, ARMS, ORDERS, run as public_audit, development, check_dev
BASE = ROOT/'artifacts/stvg_opd_supplement_v1'
OLD = ROOT/'results/stvg_opd_supplement/2026-10-11'
OUT = ROOT/REL
REV = BASE/'after_cancel_closing001'

def write(p, v):
    if p.exists():
        old = read(p)
        assert {k:x for k,x in old.items() if k != 'time'} == {k:x for k,x in v.items() if k != 'time'}, str(p)
    else: immutable_write(p, v)
LN = ROOT/'artifacts/stvg_opd_mechanism_review_v1/cross_parent_LN'

def rows(p):
    with gzip.open(p, 'rt') as f: return [json.loads(line) for line in f]

def gz(p, data, lines=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, allow_nan=False, separators=(',', ':'))+'\n' for r in data).encode() if lines else json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
    if p.exists():
        with gzip.open(p, 'rb') as f: assert f.read() == content
        return
    with p.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as f:
            f.write(content)

def checked_root(ns):
    final = read(ns/'FINAL_COMPLETION.json')
    assert final['status'] == 'complete'
    assert all(final[k] for k in ['actual_root', 'actual_PDF_pixel_view', 'actual_remote_publication', 'actual_archive'])
    pub = read(ns/'CLOSING_FINAL_GITHUB_RECEIPT.json')
    assert pub['status'] == 'pass' and pub['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    return dict(final_sha256=sha(ns/'FINAL_COMPLETION.json'), root_closing_sha256=sha(ns/'ROOT_CLOSING.json'),
                closing_public_sha256=sha(ns/'CLOSING_FINAL_GITHUB_RECEIPT.json'), commit=pub['commit'])

def source_pins(lock, design_base=None):
    r = read(lock)
    for bucket in ['pins', 'protected']:
        for name, h in r.get(bucket, {}).items(): assert sha(ROOT/name) == h, name
    for field, name in [('design_sha256', 'DESIGN_LOCK.json'), ('CPU_contracts_sha256', 'CPU_CONTRACTS.json')]:
        if field in r: assert sha((design_base or lock.parent)/name) == r[field]
    return sha(lock)

def identity(expected):
    p = Path('/proc')/str(expected['pid'])
    stat = (p/'stat').read_text().split(') ', 1)[1].split()
    assert stat[0] == 'T' and stat[19] == str(expected['start_ticks'])
    command = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert command == expected['command'] and str((p/'cwd').resolve()) == expected['cwd']
    return dict(pid=expected['pid'], state='T', start_ticks=stat[19], same_command=True, same_cwd=True)

def custody():
    authority = read(BASE/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')
    assert authority['status'] == 'remaining_formal_cancelled_by_explicit_human_instruction'
    assert authority['current_Vid_TENT_accepted'] == 13747 and not authority['formal82710_global_completed']
    assert not authority['remaining_formal_GT_scoring_executed']
    priority = ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001'
    prefix = priority/'BASELINE_ACCEPTED_PREFIX.json'; pause = priority/'PRIORITY_PAUSE_RECEIPT.json'
    assert sha(prefix) == authority['current_prefix_sha256'] and sha(pause) == authority['priority_pause_sha256']
    pp = read(prefix); assert pp['accepted_prefix_count'] == len(pp['files']) == 13747
    count = total = 0
    def byte_check(name, digest, receipt_digest=None, size=None):
        nonlocal count, total
        p = ROOT/name; rc = p.with_suffix('.json'); r = read(rc)
        assert sha(p) == digest == r['sha256'] and not r['GT_read']
        assert p.stat().st_size == r['bytes']
        if size is not None: assert p.stat().st_size == size
        if receipt_digest is not None: assert sha(rc) == receipt_digest
        count += 1; total += p.stat().st_size
    for name, b in pp['files'].items(): byte_check(name, b['sha256'], b['receipt_sha256'], b['bytes'])
    sealed = []
    for item in authority['already_sealed_stages']:
        p = BASE/'baselines/formal'/item['dataset']/item['candidate']/'PREDICTION_BARRIER.json'
        assert sha(p) == item['barrier_sha256']
        b = read(p); assert b['status'] == 'sealed' and len(b['files']) == b['arrivals'] == 10446 and not b['GT_read']
        for name, digest in b['files'].items(): byte_check(name, digest)
        sealed.append(dict(dataset=item['dataset'], candidate=item['candidate'], arrivals=10446, barrier_sha256=sha(p)))
    assert count == 34639
    pause0 = read(pause)
    identities = [identity(x) for x in pause0['baseline_processes']]
    assert read(BASE/'baselines/formal/vidstg/TENT_lr0.0001/STATUS.json')['done'] == 13747
    development_barrier = BASE/'baselines/DEVELOPMENT_GLOBAL_BARRIER.json'
    d = read(development_barrier); assert d['status'] == 'sealed' and d['arrivals'] == 1152 and not d['GT_read']
    dev_files = dev_bytes = 0
    for relative, digest in d['barriers'].items():
        p = ROOT/relative; assert sha(p) == digest
        b = read(p); assert b['status'] == 'sealed' and b['arrivals'] == 64 and not b['GT_read']
        for name, digest in b['files'].items():
            before = total; byte_check(name, digest); dev_files += 1; dev_bytes += total-before
    assert dev_files == 1152
    return dict(status='pass', scope='opaque byte/SHA256/receipt custody of all34639 accepted formal artifacts and all1152 sealed development artifacts; no payload deserialization or missing formal scoring',
        formal_accepted_artifact_count=34639, Vid_TENT_accepted_prefix=13747, HC_sealed_arrivals=20892,
        all_accepted_formal_bytes=total-dev_bytes, development_artifact_count=1152, development_bytes=dev_bytes,
        sealed_stages=sealed, baseline_process_identities=identities,
        cancellation_receipt_sha256=sha(BASE/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json'),
        accepted_prefix_manifest_sha256=sha(prefix), priority_pause_sha256=sha(pause),
        original_optimizer_process_identity_preserved=True, formal82710_global_completed=False,
        remaining_formal_GT_scoring_executed=False, no_signals_sent=True, no_payload_deserialization=True,
        CPU_only=True, GT_read=False, time=time.time())

def map_stats(v, mapping):
    if isinstance(v, dict):
        return {('parent_id' if k == 'source_id' else k): (mapping[x] if k == 'source_id' else map_stats(x, mapping)) for k, x in v.items()}
    if isinstance(v, list): return [map_stats(x, mapping) for x in v]
    return v

def run():
    REV.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    assert not (BASE/'ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json').exists()
    protected = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'methods/CURRENT_METHOD.json',
        ROOT/'scripts/publish_stvg_opd_supplement_v1.py', BASE/'orders/RUNTIME_LOCK.json', BASE/'baselines/RUNTIME_LOCK.json', BASE/'cost/RUNTIME_LOCK.json']}
    source_pins(BASE/'orders/RUNTIME_LOCK.json')
    source_pins(BASE/'baselines/recovery/qualification_optimizer_isolation005/REVISION_RUNTIME.json', BASE/'baselines')
    source_pins(LN/'RUNTIME_LOCK.json')
    if not (REV/'SERIAL_HANDOFF_SCOPE.json').exists():
        write(REV/'SERIAL_HANDOFF_SCOPE.json', dict(status='authorized_existing_evidence_publication_only', scope='post-cancellation current S evidence closure; no cancelled formal baseline restore',
            original_publisher_unchanged=True, new_GPU_calls=0, new_model_calls=0, new_DINO_calls=0, raw_GT_read=False, CPU_only=True, GT_read=False, time=time.time()))
    assert read(REV/'SERIAL_HANDOFF_SCOPE.json')['new_GPU_calls'] == 0
    custody_path = REV/'ROOT_BASELINE_CUSTODY_READBACK.json'
    if custody_path.exists():
        c = read(custody_path); assert c['status'] == 'pass' and c['formal_accepted_artifact_count'] == 34639
        assert c['accepted_prefix_manifest_sha256'] == sha(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001/BASELINE_ACCEPTED_PREFIX.json')
        assert c['cancellation_receipt_sha256'] == sha(BASE/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')
        for e in read(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001/PRIORITY_PAUSE_RECEIPT.json')['baseline_processes']: identity(e)
        assert read(BASE/'baselines/formal/vidstg/TENT_lr0.0001/STATUS.json')['done'] == 13747
    else: write(custody_path, custody())
    print('S_ACCEPTED_PREFIX_AND_DEVELOPMENT_BYTES_PASS', flush=True)
    order_root = read(BASE/'orders/ROOT_ALL_FITS_AND_STATISTICS.json')
    assert order_root['status'] == 'pass' and order_root['complete_fit_math'] == 2304 and order_root['actual_first_formal_pairs'] == 36
    ln_root = read(LN/'ROOT_COMPLETE_READBACK.json')
    assert ln_root['status'] == 'pass' and ln_root['counts']['reused_fits'] == 2304 and ln_root['original_source_and_runtime_unchanged']
    ln_final = checked_root(LN)
    saved_stats = read(OLD/'orders/ORDER_STATISTICS.json')
    old_ind = read(OLD/'orders/ROOT_ORDER_STATISTICS.json')
    assert old_ind['status'] == 'pass'
    public_rows = []; stats = {}; mapping = {}; scalar_checks = 0
    for ds, stage in [('hc2', 'O_hc2_cross_clean'), ('vidstg', 'O_vidstg_cross_clean')]:
        file = OLD/'orders'/stage/'ROWS.jsonl.gz'; rr = rows(file)
        assert len(rr) == 1152 and sha(file) == saved_stats['statistics'][stage]['rows_sha256']
        sources = sorted({r['source_id'] for r in rr}); assert len(sources) == 128
        ids = {s: i+len(mapping)*128 for i, s in enumerate(sources)}; mapping[ds] = ids
        part = [dict({k: v for k, v in r.items() if k not in {'source_id', 'query_ordinal'}}, parent_id=ids[r['source_id']]) for r in rr]
        for a, b in zip(rr, part): assert {k:v for k,v in a.items() if k not in {'source_id','query_ordinal'}} == {k:v for k,v in b.items() if k != 'parent_id'}
        public_rows += part
        cells = {(r['arm'], r['order'], r['parent_id']): r for r in part}; paired = []
        for r in part:
            if r['arm'] != 'full': continue
            z = dict(r)
            for arm in ARMS[1:]: z['Full_minus_'+arm+'_v'] = r['After_v']-cells[arm, r['order'], r['parent_id']]['After_v']
            paired.append(z)
        contrast = {}; absolute = {a:{} for a in ARMS}
        for order in ['pooled']+ORDERS:
            chosen = paired if order == 'pooled' else [r for r in paired if r['order'] == order]
            actual = independent(chosen, FIELDS)
            expected0 = old_ind['statistics'][stage]['pooled'] if order == 'pooled' else old_ind['statistics'][stage]['per_order'][order]
            scalar_checks += compare(actual, map_stats(expected0, ids)); contrast[order] = actual
            for arm in ARMS:
                chosen = [r for r in part if r['arm'] == arm and (order == 'pooled' or r['order'] == order)]
                absolute[arm][order] = independent(chosen, ABS_FIELDS)
        stats[ds] = dict(contrasts=contrast, all_arms=absolute)
    gz(OUT/'ORDER_ROWS.jsonl.gz', public_rows, True)
    gz(OUT/'ORDER_STATISTICS.json.gz', dict(status='pass', statistics=stats, bootstrap=10000, seed=20261011, historical_cohorts=True, orders_not_independent_populations=True, intervals_unadjusted=True))
    write(OUT/'STREAM_PROGRESS.json', read(OLD/'orders/STREAM_PROGRESS.json'))
    development_stats = read(OLD/'baselines/DEVELOPMENT_STATISTICS.json'); dev_rows = []
    for ds in ['hc2', 'vidstg']:
        all_part = [r for p in sorted((OLD/'baselines/development'/ds).glob('*/ROWS.jsonl.gz')) for r in rows(p)]
        ids = {s: i+(0 if ds == 'hc2' else 32) for i,s in enumerate(sorted({r['source_id'] for r in all_part}))}
        part = [dict({k:v for k,v in r.items() if k not in {'source_id','query_ordinal'}}, parent_id=ids[r['source_id']]) for r in all_part]
        for candidate in development_stats['statistics'][ds]:
            selected = [r for r in part if r['candidate'] == candidate]
            scalar_checks += check_dev(development(selected), development_stats['statistics'][ds][candidate]['all_population'])
        dev_rows += part
    assert len(dev_rows) == 1152
    gz(OUT/'DEVELOPMENT_ROWS.jsonl.gz', dev_rows, True)
    write(OUT/'DEVELOPMENT_STATISTICS.json', development_stats)
    write(OUT/'SELECTED_CONFIGS.json', read(BASE/'baselines/SELECTED_CONFIGS.json'))
    old_effect = read(OLD/'saved_evidence/UNOBSERVED_FRAME_EFFECTS.json'); effects = copy.deepcopy(old_effect)
    saved_rows = []; original = ROOT/'results/stvg_opd_paper_hc2_revision/2026-10-08'
    specs = [('Cross VidSTG','P3_vidstg_cross_clean'), ('Same-domain clean VidSTG','P4_vidstg'), ('Cross HC-STVG-v2','P3_hc2_cross_clean')]
    for label, stage in specs:
        p = original/stage/'ROWS.jsonl.gz'; assert sha(p) == old_effect['records'][label]['source_sha256']
        rr = [r for r in rows(p) if r['arm'] == 'on_policy' and r['condition'] == 'clean']
        ids = {s:i for i,s in enumerate(sorted({r['source_id'] for r in rr}))}
        effects['records'][label] = map_stats(effects['records'][label], ids)
        for r in rr: saved_rows.append(dict(cohort=label, parent_id=ids[r['source_id']], order=r['order'],
            observed_iou_delta=r['observed_iou_delta'], unobserved_iou_delta=r['unobserved_iou_delta']))
    assert len(saved_rows) == 1244
    gz(OUT/'SAVED_FRAME_EFFECT_ROWS.jsonl.gz', saved_rows, True); write(OUT/'UNOBSERVED_FRAME_EFFECTS.json', effects)
    table = read(OLD/'saved_evidence/TABLE_ROUNDING_CORRECTION.json')
    p = original/'P4_hc2/ROWS.jsonl.gz'; assert sha(p) == table['old_rows_sha256']
    part = [r for r in rows(p) if r['arm'] == 'on_policy' and r['condition'] in table['conditions']]
    ids = {s:i for i,s in enumerate(sorted({r['source_id'] for r in part}))}
    gz(OUT/'TABLE_ROUNDING_ROWS.jsonl.gz', [dict(parent_id=ids[r['source_id']], condition=r['condition'], order=r['order'], After_v=r['After_v']) for r in part], True)
    write(OUT/'TABLE_ROUNDING_CORRECTION.json', table)
    write(OUT/'METHOD_DIAGRAM_CONTRACT.json', read(OLD/'saved_evidence/METHOD_DIAGRAM_CONTRACT.json'))
    completion = read(BASE/'saved_evidence/COMPLETION.json')
    for n,h in {**completion['pins'], **completion['outputs']}.items(): assert sha(ROOT/n) == h
    case = read(BASE/'saved_evidence/CASE_COMPLETION.json')
    assert case['cases'] == 6
    provenance = OLD/'saved_evidence/FIGURE1_CASE_PROVENANCE.json'
    assert sha(provenance) == case['figure1_provenance_sha256']
    pdfs = {}; view_bindings = {}
    order_view = read(BASE/'orders/ROOT_VISUAL_REVIEW.json'); saved_view = read(BASE/'saved_evidence/ROOT_VISUAL_REVIEW.json')
    assert order_view['status'] == saved_view['status'] == 'pass'
    for e in order_view['files']:
        assert sha(ROOT/e['pdf']) == e['pdf_sha256'] and (ROOT/e['pdf']).stat().st_size == e['pdf_bytes']
        assert sha(ROOT/e['pixel_preview']) == e['pixel_sha256']
        pdfs[e['pdf']] = dict(sha256=e['pdf_sha256'], bytes=e['pdf_bytes'], pages=e['pages_actually_viewed'], public=True, original_actual_view_sha256=sha(BASE/'orders/ROOT_VISUAL_REVIEW.json'))
    for n,h in saved_view['outputs'].items(): assert sha(ROOT/n) == h
    for n,h in saved_view['pdfs'].items():
        assert sha(ROOT/n) == h
        private = n.endswith('fixed_qualitative_cases_private.pdf')
        if private: assert h == case['private_PDF_sha256']
        pdfs[n] = dict(sha256=h, bytes=(ROOT/n).stat().st_size, pages=6 if private else 1, public=not private, original_actual_view_sha256=sha(BASE/'saved_evidence/ROOT_VISUAL_REVIEW.json'))
    cost = checked_root(BASE/'cost'); cost_view = read(BASE/'cost/ROOT_VISUAL_REVIEW.json')
    for n,b in cost_view['PDF_files'].items():
        assert sha(ROOT/n) == b['sha256'] and (ROOT/n).stat().st_size == b['bytes']
        pdfs[n] = dict(b, pages=1, public=True, original_actual_view_sha256=sha(BASE/'cost/ROOT_VISUAL_REVIEW.json'))
    assert len(pdfs) == 6 and sum(b['pages'] for b in pdfs.values()) == 11
    write(BASE/'ROOT_VISUAL_REVIEW.json', dict(status='pass', scope='exact SHA reuse of all original S actual pixel-view receipts; not a new render',
        actual_pixel_view=True, actual_prior_pixel_views_reused_exactly=True, PDFs=6, pages=11,
        public_numerical_PDFs=5, public_numerical_pages=5, private_qualitative_PDFs=1, private_qualitative_pages=6,
        PDF_files=pdfs, original_view_bindings={n:sha(BASE/n) for n in ['orders/ROOT_VISUAL_REVIEW.json','saved_evidence/ROOT_VISUAL_REVIEW.json','cost/ROOT_VISUAL_REVIEW.json']},
        private_media_and_GT_geometry_excluded=True, new_model_calls=0, CPU_only=True, GT_read=False, time=time.time()))
    write(OUT/'DESIGN_PUBLIC.json', dict(status='complete_existing_S_evidence_after_explicit_remaining_formal_cancel',
        order_fits=2304, parents_per_order_target=128, targets=['hc2','vidstg'], orders=ORDERS,
        active_query_dimensions=256, active_LN_dimensions=1536, order_configs={k:v['variant_configs'] for k,v in read(BASE/'orders/DESIGN_LOCK.json')['stages'].items()},
        development_configs=18, development_arrivals=1152, development_parents_per_target=32,
        HC_sealed_selected_TENT=10446, HC_sealed_selected_SAR=10446, Vid_TENT_accepted_prefix=13747,
        formal82710_global_completed=False, remaining_formal_GT_scoring_executed=False, baseline_remaining_formal_cancelled=True,
        cost_actual_paths=32, cost_public_commit=cost['commit'], old_N_control_readback_commit=ln_final['commit'],
        saved_frame_effect_rows=1244, fixed_real_cases=6, actual_viewed_PDFs=6, actual_viewed_pages=11,
        private_case_pixels_and_GT_geometry_public=False, history_exposed=True, unadjusted_intervals=True,
        new_GPU_calls=0, new_model_calls=0, new_DINO_calls=0, new_optimizer_calls=0, raw_GT_read=False,
        production_promoted=False, manuscript_written=False, future_main_experiment_complete=False))
    audit = public_audit(ROOT); assert audit['status'] == 'pass'
    write(OUT/'PUBLIC_SCALAR_AUDIT.json', audit)
    write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json', dict(audit, private_exports_matched_originals=True, independent_original_order_dev_statistic_checks=scalar_checks, time=time.time()))
    bindings = {n:sha(BASE/n) for n in ['orders/ROOT_ALL_FITS_AND_STATISTICS.json', 'orders/ROOT_QUALIFICATION_READBACK.json',
        'baselines/ROOT_DEVELOPMENT_SELECTION_READBACK.json','baselines/ROOT_QUALIFICATION_READBACK.json',
        'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json','after_cancel_closing001/ROOT_BASELINE_CUSTODY_READBACK.json',
        'ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json','cost/FINAL_COMPLETION.json','cost/CLOSING_FINAL_GITHUB_RECEIPT.json']}
    write(BASE/'ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json', dict(status='pass', scope='current S existing orders/dev/saved evidence/cost and explicit cancelled-formal custody; missing82710 formal not required or scored',
        bindings=bindings, original_complete_order_math_fits=2304, original_complete_order_round_records=57600,
        original_full_state_coordinates=4128768, original_qualified_formal_pairs=36,
        original_independent_order_statistic_checks=30904, later_N_independent_control_readback_sha256=sha(LN/'ROOT_COMPLETE_READBACK.json'),
        original_development_arrivals=1152, baseline_opaque_accepted_artifacts=34639, original_cost_actual_paths=32,
        all_negative_neutral_unknown_retained=True, baseline_remaining_formal_cancelled=True,
        independent_full_decoder_Jacobian=False, new_GPU_calls=0, new_model_calls=0, new_DINO_calls=0,
        new_raw_GT_scoring=False, CPU_only=True, GT_read=False, time=time.time()))
    for n,h in protected.items(): assert sha(ROOT/n) == h
    write(REV/'REVISION_RUNTIME.json', dict(status='pass', scope='new publication/root custody wrappers only; original science and publisher preserved',
        protected_sha256=protected, original_source_and_runtime_unchanged=True,
        pins={n:sha(ROOT/n) for n in ['protocols/stvg_opd_supplement_after_cancel_closing001.md',
            'scripts/review_stvg_opd_supplement_cancel_closing001.py','scripts/audit_stvg_opd_supplement_cancel_public_v1.py']},
        CPU_only=True, GT_read=False, time=time.time()))
    print('S_CURRENT_AUTHORIZED_EVIDENCE_READBACK_PASS', audit['scalar_comparisons'], flush=True)

if __name__ == '__main__': run()

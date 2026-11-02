"""R51 full5912 selected-factor final-safe CPU G. Only raw-safe bundle + same-3EED references.

No hybrid, detector, Qwen, DINO, historical mask/ownership/G artifact is an
input. Source frames and source candidate order/bytes remain unchanged. The
reference diagnostic worklist is a one-way output, never feedback to this run.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence51_fidelity import attr_evidence, object_geometry, raw_relations, propagate_measured, reference_hypotheses

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'research/round8/experiment/full5912_gpu_inputs.json'
FACTORS = (.5, 1., 2.)
ARM_NAMES = ('source', 'replace_all_c05', 'replace_all_c1', 'replace_all_c2')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def path(value):
    p = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if not p.is_relative_to(ROOT) or any(x in str(p) for x in ('round46/final', 'FINAL384')):
        raise ValueError('Outside workspace or closed artifact')
    return p


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


class Inputs:
    def __init__(self, out):
        self.p = out / 'INPUT_LEDGER_PRIVATE.json'
        self.journal = out / 'INPUT_LEDGER_PRIVATE.jsonl'
        self.items = []
        self.finalized = False

    def bind(self, p, role, expected, sid=None, selected_fields=None):
        p = Path(p)
        digest = sha(p)
        item = {'path_PRIVATE': str(p), 'role': role, 'sha256': digest,
                'id_PRIVATE': sid, 'selected_fields': selected_fields}
        # Persist each binding before any caller parses the selected input.
        # Aggregate once instead of rewriting the growing JSON on every read.
        with self.journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.items.append(item)
        if digest != expected:
            raise ValueError('Input drift in ' + role)
        return p

    def finalize(self):
        if not self.finalized:
            dump(self.p, {'schema': 'R51_final_safe_G_input_ledger_v1', 'items': self.items})
            self.finalized = True


def _root_bound(plan, key, state, ledger):
    root = path(plan[key])
    if root.parent != ROOT / 'crossplatform/round51':
        raise ValueError('Final-safe stage must be a separately sealed R51 root')
    sp = root / 'SEALED.json'
    ledger.bind(sp, key.upper() + '_SEAL', plan['inputs_sha256'][sp.relative_to(ROOT).as_posix()])
    seal = read(sp)
    if seal['state'] != state:
        raise ValueError('Wrong safe stage seal')
    return root, seal


def _bound(root, seal, relative, ledger, role, sid, selected_fields=None):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Relative sealed file expected')
    return ledger.bind(root / relative, role, seal['files'][relative], sid, selected_fields)


def _load_npz(p, fields=None):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if fields is None else fields)}


def context(args):
    pp = path(args.plan)
    auth = read(path(args.authorization))
    if auth.get('state') != 'ROOT_AUTHORIZED_R51_SELECTED_FULL_FINAL_SAFE_PREDICTION_NO_GT' or auth.get('plan_sha256') != sha(pp):
        raise ValueError('Separate root final-safe prediction grant required')
    plan = read(pp)
    if plan.get('state') != 'ROOT_LOCKED_R51_SELECTED_FULL_FINAL_SAFE_PREDICTION' or plan.get('n') != 5912 or plan['manifest'] != MANIFEST:
        raise ValueError('Only separately authorized general full5912')
    if len(plan['appearance_factor_values']) != 1 or plan['appearance_factor_values'][0] not in FACTORS or plan.get('no_external_vision_cache_inputs') is not True:
        raise ValueError('Predeclared factor/data boundary required')
    if plan.get('method_selected_before_full_run') is not True or not plan.get('selected_method'):
        raise ValueError('Selected full method/factor must be frozen before new inputs')
    expected_inputs = {MANIFEST} | {plan[k] + '/SEALED.json' for k in ('prepared_root', 'reference_root')}
    if set(plan['inputs_sha256']) != expected_inputs:
        raise ValueError('Unexpected extra data input is forbidden')
    required = [Path(__file__), Path(__file__).with_name('evidence51_fidelity.py')]
    for p in required:
        if plan['programs_sha256'].get(p.relative_to(ROOT).as_posix()) != sha(p):
            raise ValueError('Required runtime code is not pinned')
    for rel, digest in plan['programs_sha256'].items():
        if sha(path(rel)) != digest:
            raise ValueError('Program drift')
    for rel, digest in plan['inputs_sha256'].items():
        if sha(path(rel)) != digest:
            raise ValueError('Pinned stage input drift')
    rows = read(ROOT / MANIFEST)
    if len(rows) != 5912 or len({r['id'] for r in rows}) != 5912:
        raise ValueError('Exact registered full5912 rows required')
    out = path(plan['output_root'])
    if out.parent != ROOT / 'crossplatform/round51' or out.exists():
        raise ValueError('Fresh R51 output attempt required')
    return plan, rows, out


def _empty_reference(M, reason):
    return [], {'unknown_probability': 1., 'source_unknown_probability': 1.,
                'source_truncated_probability': 0., 'hypotheses': 0,
                'retained_probability': 0., 'source_top3_instance_ids': [],
                'source_hypothesis_logits': [None] * M, 'source_probabilities': [0.] * M,
                'source_only_top3_fixed_before_color': True, 'status': reason}


def _decorate(hypotheses, W, H):
    out = []
    for h in hypotheses:
        q = dict(h, instance_domain='raw3d', uncertainty_mass_explicit=True)
        q['xy_norm2'] = [h['xy_pixel'][0] / W, h['xy_pixel'][1] / H] if h['xy_pixel'] is not None else None
        q['width_norm'] = (h['rect'][2] - h['rect'][0]) / W
        out.append(q)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    plan, rows, out = context(args)
    factors = tuple(plan['appearance_factor_values'])
    factor_names = {.5:'replace_all_c05',1.:'replace_all_c1',2.:'replace_all_c2'}
    arm_names = ('source',) + tuple(factor_names[c] for c in factors)
    out.mkdir(parents=True)
    for name in ('samples_PRIVATE', 'cases_PRIVATE', 'references_PRIVATE'):
        (out / name).mkdir()
    ledger = Inputs(out)
    start = time.perf_counter()
    counts = collections.Counter()
    counts['registered_n'] = len(rows)
    work_items = []
    done = []
    try:
        ledger.bind(ROOT / MANIFEST, 'REGISTERED_RAW_INPUTS', plan['inputs_sha256'][MANIFEST])
        prep, ps = _root_bound(plan, 'prepared_root', 'R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT', ledger)
        refroot, rs = _root_bound(plan, 'reference_root', 'R51_FULL5912_SELECTED_METHOD_REFERENCE_ONLY_HEADSPAN_CAPTURE_NO_GT', ledger)
        if ps['n'] != 5912 or ps.get('engineering_limit') is not None or rs['n'] != 5912:
            raise ValueError('Complete registered stages required')
        if ps.get('external_vision_outputs_or_mask_derived_cache_read') is not False or rs.get('external_visual_calls') != 0:
            raise ValueError('Upstream safe dependency assertion failed')
        for i, row in enumerate(rows):
            tick = time.perf_counter()
            sid = row['id']
            ap = _bound(prep, ps, 'arrays_PRIVATE/' + sid + '.npz', ledger, 'RAW_SAFE_POINT_INSTANCES', sid)
            cp = _bound(prep, ps, 'cases_PRIVATE/' + sid + '.json', ledger, 'RAW_SAFE_BINDINGS', sid)
            atp = _bound(prep, ps, 'attributes_PRIVATE/' + sid + '.json', ledger, 'MEASURED_RGB_AND_PURE_TEXT', sid)
            a = _load_npz(ap)
            case, attrs = read(cp), read(atp)
            if case['arrays_sha256'] != sha(ap) or case['attributes_sha256'] != sha(atp):
                raise ValueError('Prepared case binding drift')
            graph = attrs['text_graph_unchanged']
            if hashlib.sha256(row['caption'].encode()).hexdigest() != case['caption_sha256']:
                raise ValueError('Caption mismatch')
            source = a['source_scores']
            boxes = a['source_boxes']
            W, H = case['image_size_wh']
            weights = a['query_instance_conditional_weights']
            if source.shape != (256,) or boxes.shape != (256, 6) or weights.shape != (256, len(attrs['instances'])):
                raise ValueError('Source or safe instance schema mismatch')
            pointpath = Path(row['pointpath'])
            ledger.bind(pointpath, 'ORIGINAL_RAW_XYZ_FOR_REFERENCE_SUPPORT', case['raw_bindings']['point_sha256'], sid)
            raw = np.load(pointpath, allow_pickle=False) if pointpath.suffix == '.npy' else np.fromfile(pointpath, np.float32).reshape(-1, 4)
            if raw.ndim != 2 or raw.shape[1] < 3 or len(raw) != case['n_raw_points']:
                raise ValueError('Original full raw point format/count mismatch')
            raw = np.ascontiguousarray(raw[:, :3], np.float32)
            refcase_path = _bound(refroot, rs, 'cases_PRIVATE/' + sid + '.json', ledger, 'SAME_MODEL_REFERENCE_GRAPH_BINDINGS', sid)
            refcase = read(refcase_path)
            if refcase['graph_sha256'] != case['pure_text_graph_sha256']:
                raise ValueError('Reference graph/input bridge mismatch')
            subject_path = _bound(refroot, rs, refcase['subject']['file'], ledger,
                                  'ORIGINAL_R8_SUBJECT_SOURCE_BRIDGE_ONLY', sid, ['boxes','scores'])
            if sha(subject_path) != refcase['subject']['sha256']:
                raise ValueError('Reference-stage original source bridge hash differs')
            subject = _load_npz(subject_path, ['boxes','scores'])
            if not exact(subject['boxes'], boxes) or not exact(subject['scores'], source):
                raise ValueError('Original R8 source boxes/scores changed')
            objects = object_geometry(a, (W, H))
            e, e_known = attr_evidence(attrs, 'T')
            all_references = []
            reference_storage = {}
            fixed_top3 = {}
            # Load each phrase output only once, independent of factor choices.
            ref_inputs = {}
            for ref in graph['references']:
                entity = ref['entity']
                route = refcase['references'][entity]
                rp = _bound(refroot, rs, route['file'], ledger, 'SAME_GROUNDER_REFERENCE_HEADSPAN_ONLY', sid, ['boxes', 'logodds'])
                if route['sha256'] != sha(rp):
                    raise ValueError('Reference output binding drift')
                callp = _bound(refroot, rs, 'calls_PRIVATE/' + sid + '__' + entity + '.json', ledger, 'REFERENCE_SAMPLING_AND_MASK_RECEIPT', sid)
                call = read(callp)
                if call['state'] != 'COMPLETED' or call['binding']['point_sample_sha256'] != refcase['point_sample_sha256'] or call['binding']['point_indices_sha256'] != refcase['point_indices_sha256']:
                    raise ValueError('Reference did not use the same current-input point sampling')
                if route['mask'] != call['text_head_mask']:
                    raise ValueError('Reference textual-head readout receipt mismatch')
                ref_inputs[entity] = (_load_npz(rp, ['boxes', 'logodds']), route['mask']['reference_usable'])
            full_scores, no_relation_scores, components, raw_pr, known_pr = [], [], [], [], []
            cue_query_known, cue_anchors, cue_known_weight_mass = [], [], []
            for fi, factor in enumerate(factors):
                references = {}
                details = {}
                for ref in graph['references']:
                    entity = ref['entity']
                    z, usable = ref_inputs[entity]
                    hs, info = (reference_hypotheses(raw, a, z['boxes'], z['logodds'], row['platform'], attrs, entity, factor)
                                if usable else _empty_reference(len(e), 'UNKNOWN_GENERIC_OR_UNRESOLVED_TEXT_HEAD'))
                    if fi == 0:
                        fixed_top3[entity] = info['source_top3_instance_ids']
                        reference_storage[entity + '_source_logits'] = np.array([v if v is not None else -np.inf for v in info['source_hypothesis_logits']])
                        reference_storage[entity + '_source_probability'] = np.asarray(info['source_probabilities'])
                        for rank, m in enumerate(info['source_top3_instance_ids']):
                            lo, hi = a['instance_crop_mask_pixel_offsets'][m:m + 2]
                            npx = int(hi - lo)
                            rect = a['instance_rect_xyxy'][m]
                            ready = npx >= 8 and rect[2] > rect[0] and rect[3] > rect[1] and bool(ref['own_query'].strip())
                            work_items.append({'id_PRIVATE': sid, 'source_instance_id': int(m),
                                'kind': 'reference', 'reference_entity': entity,
                                'recipe': 'REFERENCE_PAD10_FULL_GRAY', 'query': ref['own_query'],
                                'status': 'READY' if ready else 'UNKNOWN_INSUFFICIENT_VISIBLE_SUPPORT_OR_EMPTY_CROP',
                                'imagepath': row['imagepath'], 'image_sha256': case['raw_bindings']['image_sha256'],
                                'instance_arrays': ap.relative_to(ROOT).as_posix(), 'instance_arrays_sha256': sha(ap),
                                'n_exclusive_visible_pixels': npx, 'minimum_exclusive_visible_pixels': 8,
                                'rect': rect.tolist(), 'source_probability': float(info['source_probabilities'][m]),
                                'native_probability': float(info['source_probabilities'][m]),
                                'source_top3_rank': rank, 'source_top3_fixed_before_color': True,
                                'source_unknown_probability': info['source_unknown_probability'],
                                'source_truncated_probability': info['source_truncated_probability'],
                                'appearance_factor_invariant_candidate_set': list(factors),
                                'pure_text_graph_sha256': case['pure_text_graph_sha256'],
                                'diagnostic_only': True, 'finalsafe_must_not_read_visual_answer': True})
                        if not info['source_top3_instance_ids']:
                            work_items.append({'id_PRIVATE': sid, 'source_instance_id': -1,
                                'kind': 'reference', 'reference_entity': entity,
                                'recipe': 'REFERENCE_PAD10_FULL_GRAY', 'query': ref['own_query'],
                                'status': 'UNKNOWN_NO_SOURCE_SUPPORTED_REFERENCE_HYPOTHESIS',
                                'source_top3_fixed_before_color': True, 'source_probability': 0.,
                                'native_probability': 0., 'diagnostic_only': True})
                    elif fixed_top3[entity] != info['source_top3_instance_ids']:
                        raise AssertionError('Color changed source-only reference candidate set')
                    decorated = _decorate(hs, W, H)
                    references[entity] = decorated
                    details[entity] = {'hypotheses': decorated, 'distribution': info,
                                       'reference_head_usable': bool(usable)}
                pr, pr_known = raw_relations(graph, objects, references, (W, H))
                final, cues, cue_info = propagate_measured(source, weights, factor * e, pr, e_known, pr_known)
                no_rel = pr.copy()
                no_rel[:, 1] = 0.
                no_rel_known = pr_known.copy()
                no_rel_known[:, 1] = False
                nr, _, _ = propagate_measured(source, weights, factor * e, no_rel, e_known, no_rel_known)
                full_scores.append(final)
                no_relation_scores.append(nr)
                components.append(cues)
                raw_pr.append(pr)
                known_pr.append(pr_known)
                cue_query_known.append(cue_info['query_known'])
                cue_anchors.append(cue_info['anchors'])
                cue_known_weight_mass.append(cue_info['query_known_weight_mass'])
                all_references.append(details)
            scores = np.stack([source.astype(float)] + full_scores)
            queries = scores.argmax(1).astype(np.int64)
            nr_scores = np.stack(no_relation_scores)
            nr_queries = nr_scores.argmax(1).astype(np.int64)
            dest = out / 'samples_PRIVATE' / (sid + '.npz')
            np.savez_compressed(dest, arm_names=np.array(arm_names), scores=scores,
                queries=queries, selected_original_boxes=boxes[queries], source_boxes=boxes,
                source_scores=source, appearance_factor_values=np.array(factors),
                no_relation_scores=nr_scores, no_relation_queries=nr_queries,
                no_relation_selected_original_boxes=boxes[nr_queries],
                components=np.stack(components), instance_attribute_e=e,
                instance_attribute_known=e_known, raw_PR=np.stack(raw_pr),
                raw_PR_known=np.stack(known_pr), query_instance_conditional_weights=weights,
                cue_query_known=np.stack(cue_query_known),cue_anchors=np.stack(cue_anchors),
                cue_query_known_weight_mass=np.stack(cue_known_weight_mass))
            ref_arrays_path = out / 'references_PRIVATE' / (sid + '.npz')
            np.savez_compressed(ref_arrays_path, **reference_storage)
            dump(out / 'references_PRIVATE' / (sid + '.json'),
                 {'id_PRIVATE': sid, 'appearance_factor_values': list(factors), 'references_by_factor': all_references})
            arm_statuses = {'source': {'unknown': False, 'fallback': False, 'no_parameter_update': True}}
            for fi, name in enumerate(arm_names[1:]):
                unavailable = not bool(cue_query_known[fi].any())
                arm_statuses[name] = {'unknown': unavailable, 'fallback': unavailable,
                                      'no_parameter_update': True}
            dump(out / 'cases_PRIVATE' / (sid + '.json'), {'id_PRIVATE': sid,
                'platform': row['platform'], 'source_boxes_scores_R8_bridge_bitexact': True,
                'original_256_source_pool_unchanged': True, 'input_prepared_case_sha256': sha(cp),
                'pure_text_graph_sha256': case['pure_text_graph_sha256'],
                'samples_sha256': sha(dest), 'reference_arrays_sha256': sha(ref_arrays_path),
                'n_instances': len(e), 'n_color_known_instances': int(e_known.sum()),
                'n_queries_with_no_known_instance_mass': int((weights.sum(1) == 0).sum()),
                'n_ref_source_top3_hypotheses': sum(len(x) for x in fixed_top3.values()),
                'reference_top3_ids_PRIVATE': fixed_top3,
                'arm_statuses': arm_statuses,
                'status_definition': 'unknown=fallback only when no query has any available cue; unchanged_top1_alone_is_not_fallback',
                'no_relation_ablation': 'only_R_zeroed_P_color_W_source_unchanged',
                'external_vision_outputs_or_mask_derived_caches_read': False,
                'seconds': time.perf_counter() - tick})
            counts['processed_n'] += 1
            counts['reference_entities'] += len(graph['references'])
            counts['source_fixed_reference_hypotheses'] += sum(len(x) for x in fixed_top3.values())
            counts['unknown_color_instances'] += int((~e_known).sum())
            counts['all_instances'] += len(e)
            done.append(sid)
            if (i + 1) % 16 == 0:
                print('R51_FINAL_SAFE_PREDICT', i + 1, len(rows), flush=True)
        dump(out / 'REFERENCE_WORK_ITEMS_PRIVATE.json', work_items)
        counts['reference_diagnostic_work_items'] = len(work_items)
        counts['reference_diagnostic_READY'] = sum(x['status'] == 'READY' for x in work_items)
        counts['reference_diagnostic_UNKNOWN'] = len(work_items) - counts['reference_diagnostic_READY']
        counts['reference_diagnostic_numeric_call_cap'] = 2 * counts['reference_diagnostic_READY']
        ledger.finalize()
        dump(out / 'COUNTS.json', dict(counts))
        dump(out / 'COSTS.json', {'seconds': time.perf_counter() - start,
            'new_grounder_forwards': 0, 'external_vision_forwards': 0, 'optimizer_steps': 0})
        dump(out / 'SEALED.json', {'state': 'R51_SELECTED_FULL_FINAL_SAFE_PREDICTIONS_NO_GT',
            'n': len(done), 'selected_method':plan['selected_method'], 'plan_sha256': sha(args.plan), 'task_GT_read': False,
            'external_vision_outputs_or_mask_derived_caches_read': False,
            'original_source_boxes_unchanged': True, 'appearance_factor_values': list(factors),
            'source_only_reference_top3_fixed_before_color': True,
            'reference_diagnostic_worklist_is_output_only': True,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'files': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    except BaseException as error:
        ledger.finalize()
        dump(out / 'FAILED.json', {'error_type': type(error).__name__, 'error': str(error),
            'completed_n': len(done), 'seconds': time.perf_counter() - start,
            'automatic_retry_or_method_change': False})
        raise


if __name__ == '__main__':
    main()

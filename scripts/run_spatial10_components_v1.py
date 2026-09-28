"""Finite F35 spatial components + appended temporal-supervision experiments.

Prediction workers never import evaluators or GT labels. Two immutable parent
registries are verified; nothing is promoted or scheduled by this script.
"""
import argparse
import collections
import copy
import fcntl
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, save, sha
from scripts.run_parametric_observation_v1 import rank, capture_input, capture_pair, cpu_tree

OUT = ROOT/'artifacts/decota_spatial10_components_v1'
F34 = ROOT/'artifacts/decota_parametric_observation_v1'
FREEZE = ROOT/'artifacts/decota_spatial10_freeze_v1'
OWN = ['vg_tta/spatial10_components_v1.py', 'scripts/run_spatial10_components_v1.py']
SEEDS = [20260914, 20260915, 20260916]


def dest(stage, key):
    return OUT/stage/(key.replace(':', '_')+'.pt')


def existing(p):
    if not p.exists():
        return False
    assert sha(p) == read(p.with_suffix('.json'))['sha256'], str(p)
    return True


def record(p, x):
    save(p, cpu_tree(x))
    write(p.with_suffix('.json'), dict(sha256=sha(p), key=x['key'], completed=time.time(),
          executed_code_pins={f: sha(ROOT/f) for f in OWN}))


def prepare():
    from methods.decota_spatial10_v1.predictor import verify_dependencies
    verify_dependencies()
    old = read(ROOT/'artifacts/decota_five_round_closure_v1/LOCK.json')
    p34 = read(F34/'LOCK.json')
    pins = {f: sha(ROOT/f) for f in OWN}
    protected = {f: sha(ROOT/f) for f in ['methods/CURRENT_METHOD.json', 'methods/CURRENT_WORKING_METHOD.json',
        'methods/decota_spatial10_v1/configs.json', 'methods/decota_spatial10_v1/WORKING_METHOD.json',
        'methods/decota_spatial10_v1/predictor.py', 'methods/decota_spatial10_v1/METHOD_CARD.md',
        'artifacts/decota_parametric_observation_v1/COMPLETION.json',
        'artifacts/decota_parametric_observation_v1/AB_SELECTION.json',
        'artifacts/decota_parametric_observation_v1/ALL_SOURCE_RESULTS.json',
        'artifacts/decota_spatial10_freeze_v1/COMPLETION.json']}
    work = read(ROOT/'methods/decota_spatial10_v1/WORKING_METHOD.json')
    protected.update(work['inherited_code_pins'])
    rows, counts = {}, {}
    for c, rr in old['rows'].items():
        old34 = {r['key']: r for r in p34['rows'][c]}
        dev = sorted([r for r in old34.values() if r['role'] == 'development'], key=lambda r: rank(r['group']))
        panel = sorted([r for r in old34.values() if r['role'] == 'expansion'], key=lambda r: rank(r['group']))
        guards = [r for r in old34.values() if r['role'] == 'watched']
        excluded = {r['group'] for r in dev+guards}
        pool = collections.defaultdict(list)
        for r in rr:
            if not r['input_unavailable']:
                pool[r['group']].append(r)
        eligible = sorted([g for g in pool if g not in excluded], key=rank)
        cap = len(eligible) if c == 'hcstvg1_test' else 48
        broad = [min(pool[g], key=lambda r: rank(r['key'])) for g in eligible[:cap]]
        assert {r['key'] for r in panel} <= {r['key'] for r in broad}
        timekeys = {r['key'] for r in dev+panel[:16]}
        jointkeys = {r['key'] for r in dev[:4]}
        merged = {}
        for r in broad+guards+dev:
            r = copy.deepcopy(old34.get(r['key'], r))
            r['roles'] = [s for s, active in [
                ('core', r['key'] in {z['key'] for z in broad}),
                ('panel', r['key'] in {z['key'] for z in panel}),
                ('guard', r['key'] in {z['key'] for z in guards}),
                ('development', r['key'] in {z['key'] for z in dev}),
                ('temporal', r['key'] in timekeys), ('coordination', r['key'] in jointkeys)] if active]
            r['in_F34'] = r['key'] in old34
            if 'wrong_query' not in r:
                donor = next(d for d in dev if d['group'] != r['group'])
                r['wrong_query'] = dict(key=donor['key'], source=donor['source'], caption=donor['input']['caption'], subject=donor['subject'])
            assert Path(r['input']['video_path']).is_file()
            merged[r['key']] = r
        rows[c] = sorted(merged.values(), key=lambda r: rank(r['group']))
        counts[c] = dict(pool_queries=len(rr), pool_sources=len({r['group'] for r in rr}),
            usable_queries=sum(not r['input_unavailable'] for r in rr), eligible_sources=len(eligible),
            budget_truncated_sources=eligible[cap:], unique_queries=len(merged),
            core_queries=len(broad), core_sources=len({r['group'] for r in broad}), panel_queries=len(panel),
            panel_sources=len(panel), guard_queries=len(guards), temporal_dev=8, temporal_panel=16,
            core_F34_reused_inputs=sum(r['key'] in old34 for r in broad),
            core_not_in_F34_inputs=sum(r['key'] not in old34 for r in broad),
            historically_unseen_claim=False)
        assert len({r['group'] for r in merged.values()}) == len(merged)
        assert len({r['input']['video_sha256'] for r in merged.values()}) == len(merged)
    allrows = [r for rr in rows.values() for r in rr]
    assert len({r['group'] for r in allrows}) == len(allrows)
    for x in ('development', 'guard'):
        aa = [r for r in allrows if x in r['roles']]
        bb = [r for r in allrows if 'core' in r['roles']]
        assert not {r['group'] for r in aa} & {r['group'] for r in bb}
        assert not {r['input']['video_sha256'] for r in aa} & {r['input']['video_sha256'] for r in bb}
    lock = dict(protocol='Spatial10_components_and_temporal_supervision_F35', created=time.time(),
        attachments={n: sha(Path('./private_authorization_notes')/n/'pasted-text.txt') for n in
                     ['private-authorization-ade406e654b7', 'private-authorization-13d96fed9c7d']},
        own_pins=pins, protected_pins=protected, counts=counts, rows=rows,
        caps=dict(fits=1200, backward=16000, new_DINO=2500), expected_new_fits=1199,
        temporal_selection=dict(teachers=['ensemble', 'original', 'mapped'], primary_new_candidates=['original', 'mapped'],
            fit_sources_per_direction=8, eval_sources_per_direction=16,
            rule='One shared teacher family: maximize equal-direction dev mean fixed-Spatial10 vIoU; tie declared order original,mapped. No LR retuning. Lock before evaluation labels.',
            steps=5, lr_source='F34 AB_SELECTION', no_GT_online=True),
        spatial=dict(lr={'hcstvg1_test': .01, 'vidstg_test': .1}, kappa={'hcstvg1_test': None, 'vidstg_test': 2.},
            gamma=1e-4, denominator=4, spatial_regularizer_denominator=1792, steps=10,
            B3_denominator=4, scope_ablation_denominator=1792, seeds=SEEDS),
        core_comparisons=['A3-A0', 'A3-A2'], auxiliary_core=['A3-A1', 'A3-A4'],
        uncertainty=dict(source_bootstrap=10000, seed=20260914, neutral_pp=.1, noninferiority_pp=.5,
            historical_exposure=True, descriptive_only=True, multiplicity='Exploratory module comparisons, no familywise significance claim'),
        no_promotion=True, no_automatic_simplification=True, no_schedules=True, no_new_models=True,
        no_corruption=True, GT_online=False,
        source_policy='Reuse F34 source/query SHA ordering and alias/media unions. Budget pretruncate Vid to48, all49 HC; panel subsets overlap core. Each selected source one unchanged query.',
        budget_policy='Global caps include appended temporal fits and diagnostic VJPs; exact parent T-A0/dev duplicate/wrong and dev Spatial10 may be reused only after input and dependency audit. Reserve complete blocks. Never outcome-tail-select.',
        cpu_analytic_attachment_available=False)
    write(OUT/'LOCK.json', lock)
    write(OUT/'SOURCE_MANIFEST.json', dict(counts=counts, rows=rows, original_pool_lock_sha256=sha(ROOT/'artifacts/decota_five_round_closure_v1/LOCK.json'),
        F34_lock_sha256=sha(F34/'LOCK.json'), historical_exposure=True, source_equals_media_checked=True,
        overlap_not_additive=True))
    print('LOCK', json.dumps(counts, ensure_ascii=False), flush=True)


def plan():
    p = read(OUT/'LOCK.json')
    own = dict(p['own_pins'])
    for amendment in sorted(OUT.glob('CODE_AMENDMENT_*.json')):
        a = read(amendment)
        assert a['lock_sha256'] == sha(OUT/'LOCK.json')
        for f, h in a['old_pins'].items():
            assert own[f] == h
        own.update(a['new_pins'])
    for f, h in {**p['protected_pins'], **own}.items():
        assert sha(ROOT/f) == h, ('changed_dependency', f)
    return p


class Budget:
    def __init__(self, p):
        self.caps = p['caps']
        self.file = OUT/'COST_COUNTER.json'
        self.counts = read(self.file) if self.file.exists() else {k: 0 for k in self.caps}
    def charge(self, key, count):
        assert self.counts[key]+count <= self.caps[key], ('budget_cap', key, self.counts)
        self.counts[key] += count
        status(self.file, self.counts)
    def reserve(self, **needed):
        assert all(self.counts[k]+v <= self.caps[k] for k, v in needed.items()), ('insufficient_complete_block_budget', needed, self.counts)


def parent(r, stage=None):
    if stage is None:
        stage = 'expand' if 'panel' in r['roles'] else 'joint'
    path = F34/stage/(r['key'].replace(':', '_')+'.pt')
    assert existing(path)
    z = load(path)
    for f, h in z.get('executed_code_pins', {}).items():
        if sha(ROOT/f) != h:
            # F34 dev worker predates later expansion/control orchestration.
            # The immutable fit math/replay hashes and actual args are still
            # exact. No unknown worker or model implementation is whitelisted.
            assert f == 'scripts/run_parametric_observation_v1.py'
            assert h == '30cc51a37a91eebc60b89483cc2ccb6e1fa13cd52f374d7b13d46fa44ebfb2a0'
            for name in ('T_PRIVATE', 'S_PRIVATE5', 'S_PRIVATE10'):
                fit = z['fits'][name]
                assert fit['gamma'] == 1e-4 and fit['lambda_s'] == 1. and fit['planned'] == 4
                assert not fit['old_mean'] and fit['GT_online'] is False and fit['restore_exact']
                assert fit['steps'] == (10 if name == 'S_PRIVATE10' else 5)
    return z, str(path), sha(path)


def observations(expert, r, frames, ids, interval, old, panel, budget):
    import numpy as np
    import torch
    from methods.decota_refine_uniform_v1.api import uniform_positions
    from scripts.run_decota_spatial_extension_v1 import ContextView
    from vg_tta.decota_s2_system_v1 import probe_from_detection
    from vg_tta.parametric_observation_v1 import tensor_hash, image_resample, match_observations
    context, s1 = r['parses']['context'], r['parses']['old']
    eligible = context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids']) <= 256
    p4, p8 = uniform_positions(ids, interval, 4), uniform_positions(ids, interval, 8)
    specs = [('original', p, 1.) for p in sorted(set(p4+(p8 if panel else [])))]+[('weak', p, .9) for p in p4]
    cache = old.get('observations', {}) if old else {}
    obs, reuse, calls = {}, [], 0
    for name, pos, scale in specs:
        if not eligible and not s1['phrase']:
            continue
        rgb = frames[pos] if scale == 1 else image_resample(frames[pos], scale)
        rgbsha = hashlib.sha256(rgb.tobytes()).hexdigest()
        cached = cache.get((name, pos))
        if cached is not None:
            receipt = cached['receipt']
            assert receipt['rgb_sha256'] == rgbsha and receipt['scale'] == scale
            assert receipt['context_active'] == eligible
            assert receipt['target_span'] == (context['span'] if eligible else None)
            assert receipt['expert_snapshot'] == read(ROOT/'artifacts/decota_refine_v1/lock.json')['expert_sha256']
            # Re-run exact processor (not expert) to verify pixels, tokens, precision.
            text = receipt['text']
            inputs = expert.processor(images=rgb, text=text, return_tensors='pt')
            for k, item in receipt['inputs'].items():
                assert tensor_hash(inputs[k]) == item['sha256'], ('cache_input', k, pos)
            obs[(name, pos)] = copy.deepcopy(cached)
            reuse.append(dict(view=name, position=pos, rgb_sha256=rgbsha, exact_processor=True))
            continue
        inp = {}
        def hook(m, args, kw):
            for k in ('pixel_values', 'pixel_mask', 'input_ids', 'attention_mask'):
                if k in kw:
                    inp[k] = dict(sha256=tensor_hash(kw[k]), shape=list(kw[k].shape), dtype=str(kw[k].dtype))
        h = expert.model.register_forward_pre_hook(hook, with_kwargs=True)
        torch.cuda.synchronize(); tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        budget.charge('new_DINO', 1)
        try:
            with torch.no_grad():
                d = ContextView(expert, context)(rgb, context['context'], context['entity']) if eligible else expert(rgb, s1['phrase'], s1['entity'])
        finally:
            h.remove()
        torch.cuda.synchronize(); seconds = time.perf_counter()-tick; calls += 1
        if eligible:
            probe = probe_from_detection(d, pos, ids[pos])
        else:
            probe = dict(position=pos, frame_id=ids[pos], accepted=d['accepted'], reason=d['reason'], margin=d['margin'],
                boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0, 4), target_scores=[d['score']] if d['box'] else [], candidate_ids=[0] if d['box'] else [])
        receipt = dict(position=pos, frame_id=ids[pos], view=name, scale=scale, rgb_sha256=rgbsha,
            rgb_shape=list(rgb.shape), inputs=inp, seconds=seconds, forward_seconds=d['seconds'], peak_bytes=torch.cuda.max_memory_allocated(),
            inverse_geometry='identity full canvas downup', text=d['text'], target_span=context['span'] if eligible else None,
            context_active=eligible, fallback=context['reason'] if not eligible else None,
            expert_snapshot=read(ROOT/'artifacts/decota_refine_v1/lock.json')['expert_sha256'])
        obs[(name, pos)] = dict(probe=probe, receipt=receipt, detection={k: v for k, v in d.items() if k != 'raw_token_logits'})
    def single(positions):
        aa = []
        for pos in positions:
            z = obs.get(('original', pos), {}).get('probe')
            if z and z['accepted']:
                j = int(np.argmax(z['target_scores']))
                aa.append(dict(position=pos, frame_id=ids[pos], box=torch.as_tensor(z['boxes'][j]).tolist(), score=float(z['target_scores'][j]), weight=1.))
        return aa
    anchors = dict(single4=single(p4), single8=single(p8) if panel else [])
    details = {}
    for name in ('weak', 'duplicate'):
        aa, mm = [], []
        for pos in p4:
            a = obs.get(('original', pos)); b = a if name == 'duplicate' else obs.get(('weak', pos))
            if a is None or b is None:
                continue
            m = match_observations(a['probe'], b['probe'])
            mm.append(dict(position=pos, **m))
            if m['anchor'] is not None:
                aa.append(m['anchor'])
        anchors[name], details[name] = aa, mm
    aa = copy.deepcopy(anchors['weak'])
    for a in aa:
        m = next(m for m in details['weak'] if m['position'] == a['position'])
        chosen = max(m['matches'], key=lambda v: v['score'])
        a['box'] = torch.as_tensor(obs[('original', a['position'])]['probe']['boxes'][chosen['first_index']]).tolist()
    anchors['original_member'] = aa
    if old:
        assert anchors['weak'] == old['anchors']['weak'], 'parent_weak_observation_changed'
    observed4 = sorted({p for n, p in obs if p in p4})
    observed8 = sorted({p for n, p in obs if n == 'original' and p in p8})
    return dict(anchors=anchors, details=details, observations=obs, reused=reuse, new_DINO=calls,
        positions4=p4, positions8=p8, observed4=observed4, observed8=observed8,
        common_observed=sorted(set(observed4+observed8)), duplicate_exact_cache=True,
        no_phrase=not eligible and not s1['phrase'])


def capture_timed(model, r):
    import numpy as np
    import torch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.shared_state_v1 import capture_shared
    from vg_tta.parametric_observation_v1 import tensor_hash
    assert sha(r['input']['video_path']) == r['input']['video_sha256']
    tick = time.perf_counter(); frames, ids = decode(r['input']); decode_s = time.perf_counter()-tick
    tick = time.perf_counter(); batch = make_batch(frames, ids, r['input'], r['subject'], model)
    qa = dict(rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(), input_tensor_sha256=tensor_hash(batch['videos'].tensors),
              actionness_placeholder_zero=not bool(batch['targets'][0]['actioness'].any()))
    assert qa['actionness_placeholder_zero']
    base, _, records, _, _, views = capture_shared(model, batch)
    torch.cuda.synchronize(); freeze_s = time.perf_counter()-tick
    return frames, base, records, views, qa, dict(decode_seconds=decode_s, frozen_backbone_seconds=freeze_s)


def specifications(c, ex, full):
    sp = read(OUT/'LOCK.json')['spatial']; kap = sp['kappa'][c]
    aa = ex['anchors']; out = [('C0', aa['weak'], dict(gradients=full))]
    if not full:
        return out
    from vg_tta.spatial10_components_v1 import anchor_variant
    out += [(n, aa[a], {}) for n, a in [('B0', 'single4'), ('B1', 'duplicate'), ('B3', 'single8'), ('BF', 'original_member')]]
    out += [('C1', anchor_variant(aa['weak'], 'confidence'), dict(gradients=True)),
            ('C2', anchor_variant(aa['weak'], 'uniform'), dict(gradients=True))]
    out += [(f'C3_{s}', anchor_variant(aa['weak'], 'shuffle', s), dict(gradients=True)) for s in SEEDS]
    out += [('D_rho', aa['weak'], dict(kappa=2. if kap is None else None)),
            ('D_gamma0', aa['weak'], dict(gamma=0., gradients=True))]
    if c == 'vidstg_test':
        out.append(('D_initial', aa['weak'], dict(frozen_influence=True)))
    out += [('E_query', aa['weak'], dict(scope='query')), ('E_ln', aa['weak'], dict(scope='ln'))]
    return out


def spatial_one(model, expert, r, c, budget, model_digest):
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.spatial10_components_v1 import fit, scope_mask
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    from vg_tta.foreground_runtime import state_digest
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from methods.decota_refine_uniform_v1.api import reconstruct
    full = 'panel' in r['roles'] or 'guard' in r['roles']
    budget.reserve(fits=(15 if c == 'vidstg_test' else 14) if full else 1,
                   backward=(220 if c == 'vidstg_test' else 210) if full else 10, new_DINO=16 if full else 8)
    tick = time.perf_counter()
    frames, base, records, views, qa, cost = capture_timed(model, r)
    ids = r['input']['frame_ids']; it = ObservationReplay(model, views, len(ids), 'spatial')
    with torch.no_grad():
        zero = it.values()
    native = list(fitted_merge(zero['logits'], records, ids)); native_boxes = zero['boxes'].cpu()
    old, oldpath, oldsha = parent(r) if r['in_F34'] else (None, None, None)
    if old:
        assert qa['input_tensor_sha256'] == old['qa'][0]['input_tensor_sha256']
        assert qa['rgb_sha256'] == old['qa'][0]['rgb_sha256']
        assert torch.equal(native_boxes, old['native_boxes']) and native == old['native_indices']
        assert all(torch.equal(z.cpu(), q) for z, q in zip(zero['logits'], old['fits']['S_PRIVATE10']['final']['logits'][:2]))
    # Guard original8 observations live in observe, not joint (but both use same weak).
    oldex = parent(r, 'observe')[0]['expert'] if old and 'guard' in r['roles'] else old['expert'] if old else None
    ex = observations(expert, r, frames, ids, native, oldex, full, budget)
    spec = specifications(c, ex, full)
    fits = {}
    for name, anchors, args in spec:
        pp = dict(lr=.01 if c == 'hcstvg1_test' else .1, kappa=None if c == 'hcstvg1_test' else 2., gamma=1e-4, steps=10)
        pp.update(args); scope = pp.pop('scope', 'both')
        del it
        it = ObservationReplay(model, views, len(ids), 'spatial'); scope_mask(it, scope)
        torch.cuda.reset_peak_memory_stats()
        z = fit(it, anchors, records, ids, charge=budget.charge, **pp)
        z['peak_bytes'] = torch.cuda.max_memory_allocated()
        assert all(torch.equal(a, b.cpu()) for a, b in zip(z['final']['logits'], zero['logits']))
        assert z['final']['indices'] == native
        if name == 'C0' and old:
            pfit = old['fits']['S_PRIVATE10']
            assert torch.equal(z['final']['boxes'], pfit['final']['boxes'])
            assert all(torch.equal(z['state'][n], pfit['state'][n]) for n in z['state'])
            assert z['best_step'] == pfit['best_step']
            assert [a['loss'] for a in z['path']] == [a['loss'] for a in pfit['path']]
            z['parent_exact'] = True
        clock = time.perf_counter()
        replay = full_prediction(model, frames, ids, r['input'], r['subject'], z['state'], z['final'])
        z['full_replay_seconds'] = time.perf_counter()-clock; z['full_replay_audit'] = replay['audit']
        fits[name] = z
        print('SPATIAL', r['key'], name, 'best', z['best_step'], 'delta', round(z['state_delta'], 6), 'seconds', round(z['seconds'], 2), flush=True)
    controls = {'A0': dict(boxes=native_boxes, indices=native)}
    for name, mode in [('A1', 'direct'), ('A2', 'absolute'), ('A4', 'residual')]:
        bb, a = reconstruct(native_boxes, ex['anchors']['weak'], ids, mode)
        controls[name] = dict(boxes=bb, indices=native, reconstruction_audit=a)
    assert state_digest(model) == model_digest
    result = dict(key=r['key'], source=r['source'], group=r['group'], roles=r['roles'], frame_ids=ids,
        native_boxes=native_boxes, native_indices=native, native_logits=[z.cpu() for z in zero['logits']],
        fits=fits, controls=controls, expert=ex, qa=qa, costs=cost, seconds=time.perf_counter()-tick,
        source_restored=True, GT_online=False, parent_path=oldpath, parent_sha256=oldsha,
        old_matches=bool(old), actual_temporal_logits_unchanged=True)
    record(dest('spatial', r['key']), result)
    del it, views, frames, zero, old, oldex, fits
    gc.collect(); torch.cuda.empty_cache()


def temporal_one(model, r, c, budget, model_digest):
    return temporal_runtime_one(model, r, c, budget, model_digest)


def temporal_runtime_one(model, r, c, budget, model_digest):
    import torch
    from vg_tta.parametric_observation_v1 import ObservationReplay, build_teacher, move_mass, posterior_decode
    from vg_tta.spatial10_components_v1 import fit, cross_gradients
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    dev = 'development' in r['roles']; joint = 'coordination' in r['roles']
    budget.reserve(fits=2+(0 if dev else 2)+(4 if joint else 0),
                   backward=10+(0 if dev else 10)+(22 if joint else 0), new_DINO=0)
    tick = time.perf_counter()
    frames, base, records, views, zero, qa, _ = capture_pair(model, r)
    ids = r['input']['frame_ids']; native = list(fitted_merge(zero['logits'][:2], records[:2], ids))
    old, oldpath, oldsha = parent(r)
    assert qa == old['qa'] and torch.equal(zero['boxes'].cpu(), old['native_boxes'])
    rebuilt = build_teacher(zero['logits'], records, ids, .6, qa[1]['grid']['active'])
    assert all(torch.equal(q.cpu(), p) for q, p in zip(rebuilt['probabilities'], old['teacher']['probabilities']))
    map_roundoff = max(float((q.cpu()-p).abs().max()) for q, p in zip(rebuilt['targets'], old['teacher']['targets']))
    assert map_roundoff < 1e-14, ('mapped_teacher_changed', map_roundoff)
    # CUDA scatter_add has order-dependent double roundoff (~1e-18). Freeze
    # the EXACT previously consumed targets; do not quietly claim bit equality
    # for rebuilt sums or let numerical rebuilding change a teacher arm.
    ensemble = copy.deepcopy(old['teacher'])
    teachers = dict(ensemble=ensemble)
    for name in ('original', 'mapped'):
        q = ([x.detach().clone() for x in ensemble['probabilities'][:2]] if name == 'original' else
             [move_mass(ensemble['probabilities'][i+2], ensemble['maps'][i]['inverse']).detach() for i in range(2)])
        t = copy.deepcopy(ensemble); t['q'] = q
        t['targets'] = q+[move_mass(q[i], t['maps'][i]['forward']).detach() for i in range(2)]
        t['source'] = name; teachers[name] = t
    tlr = read(F34/'AB_SELECTION.json')[c]['temporal']['lr']
    slr = .01 if c == 'hcstvg1_test' else .1; kappa = None if c == 'hcstvg1_test' else 2.
    fits = {}; reused = {}
    fits['ensemble'] = copy.deepcopy(old['fits']['T_PRIVATE'])
    reused['ensemble'] = dict(path=oldpath, sha256=oldsha, arm='T_PRIVATE', pixel_and_graph_exact=True)
    observations_old = parent(r, 'observe')[0] if dev else None
    if dev:
        for name, oldname in [('duplicate', 'T_MATCH_DUP'), ('wrong', 'T_MATCH_WRONG')]:
            fits[name] = copy.deepcopy(old['fits'][oldname])
            teachers[name] = copy.deepcopy(observations_old['duplicate_teacher' if name == 'duplicate' else 'wrong_teacher'])
            reused[name] = dict(path=oldpath, sha256=oldsha, arm=oldname, pixel_and_graph_exact=True)
    else:
        teachers['duplicate'] = build_teacher(zero['logits'][:2]*2, records[:2]*2, ids, .5, True)
        wrong_logits, wrongqa = [], []
        for mid in (.5, .6):
            _, _, wr, wv, wq = capture_input(model, r, mid, True)
            it = ObservationReplay(model, wv, len(ids), 'head')
            with torch.no_grad():
                wz = it.values()
            wrong_logits += [z.detach().cpu() for z in wz['logits']]; wrongqa.append(wq)
            del it, wv, wz
        teachers['wrong'] = build_teacher(wrong_logits, records, ids, .6, wrongqa[1]['grid']['active'])
        teachers['wrong']['wrong_query'] = r['wrong_query']; teachers['wrong']['wrong_qa'] = wrongqa
    for name in ('original', 'mapped')+(() if dev else ('duplicate', 'wrong')):
        vv = views[:2]*2 if name == 'duplicate' else views
        rr = records[:2]*2 if name == 'duplicate' else records
        it = ObservationReplay(model, vv, len(ids), 'head')
        z = fit(it, [], rr, ids, lr=slr, steps=5, kind='temporal', head_lr=tlr,
                teacher=teachers[name], charge=budget.charge)
        assert torch.equal(z['final']['boxes'], zero['boxes'].cpu())
        fits[name] = z
        print('TEMPORAL', r['key'], name, 'best', z['best_step'], 'delta', round(z['state_delta'], 6), flush=True)
        del it
    teacher_controls = {n: dict(boxes=zero['boxes'].cpu(), indices=posterior_decode(t['q'], records, ids)) for n, t in teachers.items()}
    coordination = None
    if joint:
        aa = old['expert']['anchors']['weak']
        it = ObservationReplay(model, views, len(ids), 'spatial_head')
        cross = cross_gradients(it, aa, ensemble, kappa, budget.charge)
        assert cross['spatial_to_head'] == cross['temporal_to_spatial'] == 0, ('nonseparable_graph', cross)
        del it
        jj = {}
        for name, scope, kind, vv, independent in [
            ('private_T', 'head', 'temporal', views, False),
            ('private_S', 'spatial', 'spatial', views[:2], False),
            ('sum', 'spatial_head', 'joint', views, False),
            ('grouped', 'spatial_head', 'joint', views, True)]:
            it = ObservationReplay(model, vv, len(ids), scope)
            z = fit(it, aa if kind != 'temporal' else [], records, ids, lr=slr, steps=5, kind=kind,
                    kappa=kappa, head_lr=tlr, teacher=ensemble, independent_accept=independent, charge=budget.charge)
            jj[name] = z
            if name in ('private_T', 'private_S'):
                oldfit = old['fits']['T_PRIVATE' if name == 'private_T' else 'S_PRIVATE5']
                assert torch.equal(z['final']['boxes'], oldfit['final']['boxes'])
                assert all(torch.equal(z['state'][n], oldfit['state'][n]) for n in z['state'])
            del it
        # A zero gradient alone does not certify finite-step forward independence.
        it = ObservationReplay(model, views, len(ids), 'spatial_head')
        st = it.state(); st.update(jj['private_S']['state']); it.restore(st)
        with torch.no_grad():
            s_only = it.values()
        assert all(torch.equal(z.cpu(), q.cpu()) for z, q in zip(s_only['logits'], zero['logits']))
        st = it.initial.copy(); st.update(jj['private_T']['state']); it.restore(st)
        with torch.no_grad():
            t_only = it.values()
        assert torch.equal(t_only['boxes'].cpu(), zero['boxes'].cpu())
        st.update(jj['private_S']['state']); it.restore(st)
        with torch.no_grad():
            both = it.values()
        independent = dict(boxes=both['boxes'].cpu(), logits=[z.cpu() for z in both['logits']],
                           indices=list(fitted_merge(both['logits'][:2], records[:2], ids)))
        assert torch.equal(independent['boxes'], jj['private_S']['final']['boxes'])
        assert all(torch.equal(a, b) for a, b in zip(independent['logits'], jj['private_T']['final']['logits']))
        grouped_state_exact = all(torch.equal(jj['grouped']['state'][n], p.cpu()) for n, p in st.items())
        for z in jj.values():
            z['reinsertion'] = full_prediction(model, frames, ids, r['input'], r['subject'], z['state'], z['final'])['audit']
        coordination = dict(fits=jj, cross_gradients=cross, finite_forward_cross_zero=True,
                            private_composed=independent, grouped_state_equals_private=grouped_state_exact)
        del it
    # Fix Spatial10's original support and coordinates, then ACTUALLY reinsert
    # each time state plus that SAME spatial state in the full source backbone.
    spatial_parent = old['fits']['S_PRIVATE10']
    systems, replay_costs = {}, {}
    for name, z in fits.items():
        st = dict(spatial_parent['state']); st.update(z['state'])
        expected = dict(boxes=spatial_parent['final']['boxes'], logits=z['final']['logits'], indices=z['final']['indices'])
        t = time.perf_counter()
        live = full_prediction(model, frames, ids, r['input'], r['subject'], st, expected)
        systems[name] = dict(boxes=live['boxes'], indices=live['indices'], audit=live['audit'])
        replay_costs[name] = time.perf_counter()-t
    assert state_digest(model) == model_digest
    result = dict(key=r['key'], source=r['source'], group=r['group'], roles=r['roles'], frame_ids=ids,
        fits=fits, teachers=cpu_tree(teachers), teacher_controls=teacher_controls, reused=reused,
        native_boxes=zero['boxes'].cpu(), native_indices=native, native_logits=[z.cpu() for z in zero['logits']],
        spatial_boxes=spatial_parent['final']['boxes'], systems=systems, original_spatial_anchors=old['expert']['anchors']['weak'],
        I_seed=native, unchanged_spatial_supervision=True, spatial_state_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in spatial_parent['state'].values())).hexdigest(),
        coordination=coordination, qa=qa, map_rebuild_max_roundoff=map_roundoff,
        teacher_original_probabilities_exact=True, exact_parent_ensemble_consumed=True,
        costs=dict(final_replay_seconds=replay_costs),
        seconds=time.perf_counter()-tick, GT_online=False, source_restored=True, new_DINO=0,
        parent_path=oldpath, parent_sha256=oldsha)
    record(dest('temporal', r['key']), result)
    del frames, base, records, views, zero, old, fits, teachers, spatial_parent
    gc.collect(); torch.cuda.empty_cache()


def runtime(stage, cohort=None, limit=0):
    import torch
    from scripts.run_closure_v1 import model_for
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    p = plan(); budget = Budget(p); n = 0
    expert = None
    if stage == 'spatial':
        config = read(ROOT/'artifacts/decota_refine_v1/lock.json')
        assert sha(Path(config['expert_snapshot'])/'model.safetensors') == config['expert_sha256']
        tick = time.perf_counter(); expert = SpatialExpert(config['expert_snapshot']); expert_digest = state_digest(expert.model)
        write(OUT/'loads'/f'expert_{time.time_ns()}.json', dict(seconds=time.perf_counter()-tick, cold_model_load=True))
    for c, rr in p['rows'].items():
        if cohort and cohort != c:
            continue
        pending = [r for r in rr if (('core' in r['roles'] or 'guard' in r['roles']) if stage == 'spatial' else 'temporal' in r['roles'])
                   and not existing(dest(stage, r['key']))]
        # Temporal development always precedes panel; downstream selection uses only dev labels.
        pending.sort(key=lambda r: (0 if 'development' in r['roles'] else 1, rank(r['group'])))
        if not pending or (limit and n >= limit):
            continue
        tick = time.perf_counter(); model = model_for(c); md = state_digest(model)
        write(OUT/'loads'/f'{c}_{time.time_ns()}.json', dict(cohort=c, stage=stage, seconds=time.perf_counter()-tick))
        for r in pending:
            if limit and n >= limit:
                break
            if stage == 'spatial':
                spatial_one(model, expert, r, c, budget, md)
                assert state_digest(expert.model) == expert_digest
            else:
                temporal_one(model, r, c, budget, md)
            n += 1
            status(OUT/'STATUS.json', dict(stage=stage, last=r['key'], completed_this_invocation=n, counts=budget.counts, time=time.time()))
        del model; gc.collect(); torch.cuda.empty_cache()
    if expert:
        del expert; gc.collect(); torch.cuda.empty_cache()
    write(OUT/'invocations'/f'{stage}_{time.time_ns()}.json', dict(stage=stage, completed=n, cumulative_counts=budget.counts, completed_unix=time.time()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['prepare', 'spatial', 'temporal'])
    ap.add_argument('--cohort'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    if a.stage == 'prepare':
        return prepare()
    from scripts.run_decota_refine_v1 import configure
    configure()
    lease = open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock', 'a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, failure = time.time(), None
    try:
        runtime(a.stage, a.cohort, a.limit)
    except BaseException as e:
        failure = repr(e)
        raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json', dict(stage=a.stage, seconds=time.time()-tick, failure=failure))
        fcntl.flock(lease, fcntl.LOCK_UN); lease.close()


if __name__ == '__main__':
    main()

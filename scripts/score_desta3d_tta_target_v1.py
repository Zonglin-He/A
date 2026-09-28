"""Post-192-pair-seal CPU scoring for fixed E5; never called by inference."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_tta_target_readback_v1 import ART, TARGET, OLD, read, sha, write_once
from scripts.score_ptd_corruption_coupling_v1 import metric as original_metric
from scripts.score_ptd_spatial_adapter_ab_v1 import truth
from vg_tta.ptd_spatial_adapter_ab_v1 import boxes_from_tokens

SCORE = TARGET / 'scoring_rev2'


def metric_boxes(pred, row, gt):
    """Independent scalar geometry with exactly P3's temporal/support convention."""
    ids = row['input']['frame_ids']
    n = len(ids)
    present = np.zeros(n, dtype=bool)
    ious = np.zeros(n, dtype=float)
    if pred['format_ok']:
        for pos, b in zip(pred['positions'], np.asarray(pred['boxes_cxcywh'], dtype=float)):
            present[pos] = True
            g = gt['boxes'][pos]
            x1, y1 = b[:2] - b[2:] / 2
            x2, y2 = b[:2] + b[2:] / 2
            a1, b1 = g[:2] - g[2:] / 2
            a2, b2 = g[:2] + g[2:] / 2
            area = max(0., min(x2, a2) - max(x1, a1)) * max(0., min(y2, b2) - max(y1, b1))
            union = max(0., x2-x1) * max(0., y2-y1) + (a2-a1)*(b2-b1) - area
            ious[pos] = area / max(union, 1e-12)
    inside = np.zeros(n, dtype=bool)
    denom = 0
    v, t = 0., 0.
    if pred.get('interval') is not None:
        s, e = pred['interval']
        a, b = ids[s], ids[e] + 1
        c, d = gt['interval']
        inside = np.asarray([a <= fid < b for fid in ids])
        denom = sum(min(a,c) <= fid < max(b,d) for fid in ids)
        v = sum(ious[j] for j in range(n) if gt['valid'][j] and inside[j]) / max(1, denom)
        t = max(0., min(b,d)-max(a,c)) / max(1., max(b,d)-min(a,c))
    return {'v':float(v), 't':float(t),
            's':float(ious[gt['valid']].mean()) if gt['valid'].any() else None,
            'ious':ious.tolist(), 'GT_valid_positions':int(gt['valid'].sum()),
            'v_union_sampled_positions':denom,
            'common_positions':int((present & inside & gt['valid']).sum()),
            'format_ok':bool(pred['format_ok']), 'interval':pred.get('interval'),
            'positions':pred.get('positions') or []}


def recovered_tokens(pred):
    # Used only to verify metric equivalence, never changes the evaluated boxes.
    b = pred['boxes_cxcywh'].float()
    tok = torch.cat((b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2), -1).mul(1000).round().long()
    rebuilt, valid = boxes_from_tokens(tok)
    assert torch.allclose(rebuilt, b, rtol=0, atol=2e-7)
    assert torch.equal(valid, pred['geometry_valid'])
    return tok


def stats(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    rng = np.random.default_rng(20260927)
    boot = a[rng.integers(0, len(a), (10000, len(a)))].mean(1)
    return {'n':len(a), 'mean':float(a.mean()), 'ci95':np.quantile(boot, [.025,.975]).tolist(),
            'positive':int((a > 1e-12).sum()), 'negative':int((a < -1e-12).sum()),
            'zero':int((np.abs(a) <= 1e-12).sum()), 'minimum':float(a.min()),
            'maximum':float(a.max()), 'severe_loss_gt5pp':int((a < -.05).sum())}


def crosscheck(actual, expected):
    errors = [abs(actual[k]-expected[k]) for k in ['v','t','s'] if actual[k] is not None]
    errors.append(float(np.max(np.abs(np.asarray(actual['ious'])-np.asarray(expected['ious'])))))
    assert max(errors) < 1e-6, errors
    return max(errors)


def synthetic():
    r = {'input':{'frame_ids':[0,10,20,30]}}
    g = {'boxes':np.tile([.5,.5,.2,.2], (4,1)), 'valid':np.array([False,True,True,False]), 'interval':[10,21]}
    p = {'format_ok':True, 'positions':[1,2], 'interval':[1,2],
         'boxes_cxcywh':torch.tensor([[.5,.5,.2,.2],[.5,.5,.2,.2]]), 'geometry_valid':torch.ones(2,dtype=torch.bool)}
    m = metric_boxes(p,r,g)
    assert abs(m['v']-1)<1e-6 and abs(m['s']-1)<1e-6 and m['t']==1
    crosscheck(m, original_metric(p,recovered_tokens(p),r,g))
    missing = dict(p, positions=[1], boxes_cxcywh=p['boxes_cxcywh'][:1], geometry_valid=p['geometry_valid'][:1])
    assert abs(metric_boxes(missing,r,g)['s']-.5)<1e-6
    bad = dict(p, format_ok=False)
    assert metric_boxes(bad,r,g)['v']==0 and metric_boxes(bad,r,g)['t']==1
    g0 = dict(g, valid=np.zeros(4,dtype=bool))
    assert metric_boxes(p,r,g0)['s'] is None and metric_boxes(p,r,g0)['v']==0
    assert metric_boxes(dict(p, interval=None, format_ok=False),r,g)['t']==0
    absent_fields = {'format_ok':False, 'interval':None}
    assert metric_boxes(absent_fields,r,g)['v']==0 and metric_boxes(absent_fields,r,g)['positions']==[]
    crosscheck(metric_boxes(absent_fields,r,g), original_metric(absent_fields,[],r,g))
    return {'status':'passed', 'checks':['fixed_support_missing_zero','valid_time_with_spatial_format_failure',
            'no_legal_GT_support_retained','absent_interval_zero','format_failure_missing_positions_retained',
            'P3_scalar_equivalence'], 'target_GT_read':False}


def run():
    torch.set_num_threads(2)
    SCORE.mkdir(exist_ok=True)
    reg = read(TARGET / 'REGISTRATION.json')
    bar_path = TARGET / 'ALL_192_PAIRS_BARRIER.json'
    bar = read(bar_path)
    lock = read(SCORE / 'SCORING_LOCK.json')
    assert lock['initial_target_GT_read_after_all_predictions_sealed'] is True
    for p, h in lock['pins'].items():
        assert sha(p)==h, p
    assert lock['pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert bar['status']=='all_192_pairs_structurally_audited_before_target_GT_read'
    assert bar['pairs']==192 and bar['individual_predictions']==384 and bar['GT_read'] is False
    assert bar['conditions']==reg['conditions'] and len(bar['conditions'])==3
    assert bar['audit_script_sha256']==sha(ROOT / 'scripts/desta3d_tta_target_readback_v1.py')
    assert bar['registration_sha256']==sha(TARGET / 'REGISTRATION.json')
    for condition in reg['conditions']:
        structure_path = TARGET / ('STRUCTURE_'+condition+'.json')
        structure = read(structure_path)
        assert structure['status']=='condition_structural_readback_passed' and structure['GT_read'] is False
        assert structure['condition']==condition and structure['queries']==structure['parents']==structure['updates']==structure['prediction_pairs']==64
        assert structure['script_sha256']==bar['audit_script_sha256']
        assert bar['files'][str(structure_path)]==sha(structure_path)
        expected_seal = ART / 'runs' / reg['run_ids'][condition] / 'PREDICTION_SEAL.json'
        assert structure['seal_path']==str(expected_seal)
        seal = read(expected_seal)
        assert seal['status']=='complete_gt_free_e5_tta_predictions' and seal['target_GT_read'] is False
        assert seal['count']==len(seal['files'])==64
        assert structure['seal_sha256']==bar['files'][str(expected_seal)]==sha(expected_seal)
        for f in seal['files']:
            assert bar['files'][f['path']]==f['prediction_sha256']
    for p,h in {**bar['files'], **reg['pins']}.items():
        assert sha(p)==h,p
    # Bind each seal identity to its payload before any label is opened.
    expected = {r['key']:r for r in read(TARGET / 'INPUTS.json')}
    payloads, identity_rows = {}, []
    for condition in reg['conditions']:
        seal = read(ART / 'runs' / reg['run_ids'][condition] / 'PREDICTION_SEAL.json')
        seen = set()
        for f in seal['files']:
            d = torch.load(f['path'], map_location='cpu', weights_only=False)
            assert d['key']==f['key'] and d['key'] in expected and d['key'] not in seen
            e = expected[d['key']]
            assert d['source']==e['source'] and d['cohort']==e['cohort']
            assert d['condition']==condition and d['split']=='development'
            assert d['GT_used'] is False and d['target_GT_read'] is False
            assert d['frame_ids']==e['input']['frame_ids'] and d['fps']==e['input']['fps']
            seen.add(d['key'])
            payloads[(d['key'],condition)] = d
            identity_rows.append({'key':d['key'],'source':d['source'],'condition':condition,
                                  'path':f['path'],'sha256':f['prediction_sha256']})
        assert seen==set(expected) and len({expected[k]['source'] for k in seen})==64
    write_once(SCORE / 'PAYLOAD_IDENTITY_BARRIER.json', {
        'time':time.time(),'pairs':len(identity_rows),'GT_read':False,
        'status':'all_192_seal_keys_equal_payload_keys_unique_and_match_registered_inputs',
        'scoring_script_sha256':sha(__file__),'rows':identity_rows})
    old_inputs = {r['key']:r for r in read(OLD / 'INPUTS.json')}
    native_barrier = read(OLD / 'CAPTURE_DEVELOPMENT_BARRIER.json')
    native_paths, native_payloads = {}, {}
    for condition in reg['conditions']:
        for key in expected:
            p = OLD / 'capture' / condition / (hashlib.sha256(key.encode()).hexdigest()+'_F.pt')
            assert sha(p)==native_barrier['files'][str(p)]
            native_paths[str(p)] = native_barrier['files'][str(p)]
    for condition in reg['conditions']:
        for key in expected:
            p = OLD / 'capture' / condition / (hashlib.sha256(key.encode()).hexdigest()+'_F.pt')
            d = torch.load(p, map_location='cpu', weights_only=False)
            assert d['key']==key and d['condition']==condition and d['GT_used'] is False
            z = d['z']
            assert z['frame_ids']==expected[key]['input']['frame_ids']
            assert z['preprocess']==payloads[(key,condition)]['views']['base']['preprocess']
            native_payloads[(key,condition)] = z
    write_once(SCORE / 'NATIVE_FROZEN_PRE_GT_BARRIER.json', {
        'time':time.time(),'count':192,'GT_read':False,'files':native_paths,
        'old_capture_barrier_sha256':sha(OLD / 'CAPTURE_DEVELOPMENT_BARRIER.json'),
        'status':'192_Frozen_payloads_hash_identity_pixel_checked_and_loaded_before_GT',
        'scoring_script_sha256':sha(__file__)})
    # GT becomes accessible only after all 384 outputs and structural audits verify.
    label_path = ROOT / 'artifacts/ptd_joint_box_opd_v1/LABELS_SCORER_ONLY.json'
    label_hash = 'f9c766be993784b2b5f987cc531477e83ba73a994a249eabc21b6931c0dc4b09'
    assert sha(label_path)==label_hash
    write_once(SCORE / 'OFFLINE_GT_READ_STARTED.json', {
        'time':time.time(), 'barrier_sha256':sha(bar_path), 'scoring_script_sha256':sha(__file__),
        'labels_path':str(label_path), 'labels_sha256':label_hash,
        'scope':'offline mechanism/efficacy scoring of historically exposed 64 development parents only',
        'used_for_update_state_selection_or_source_filtering':False})
    labels = read(label_path)
    records = []
    max_error = 0.
    for condition in reg['conditions']:
        seal = read(ART / 'runs' / reg['run_ids'][condition] / 'PREDICTION_SEAL.json')
        for f in seal['files']:
            d = payloads[(f['key'],condition)]
            r = old_inputs[d['key']]
            g = truth(r, labels)
            z = native_payloads[(d['key'],condition)]
            native = dict(z, boxes_cxcywh=z.get('boxes',torch.empty((0,4))))
            arms = {'Frozen':native, 'Sourcefit':d['baseline_source_fit_frozen_no_tta'], 'TTA':d['adapted_fixed_terminal']}
            metrics = {}
            for name,p in arms.items():
                metrics[name] = metric_boxes(p,r,g)
                tokens = z.get('base_tokens',[]) if name=='Frozen' else recovered_tokens(p)
                max_error = max(max_error, crosscheck(metrics[name],original_metric(p,tokens,r,g)))
            good = g['valid'] & (np.asarray(metrics['Frozen']['ious'])>=.5)
            good_counts = {name:int((good & (np.asarray(m['ious'])>=.5)).sum()) for name,m in metrics.items()}
            records.append({'key':r['key'],'source':r['source'],'domain':r['domain'],'condition':condition,
                            'arms':metrics,'native_good_frames':int(good.sum()),'native_good_retained':good_counts})
    assert len(records)==192
    write_once(SCORE / 'SCORED_QUERY_CONDITIONS.json', records)
    by = {(x['key'],x['condition']):x for x in records}
    summaries, parent_tables = {}, []
    for domain in ['all','HC','Vid']:
        keys = sorted({x['key'] for x in records if domain=='all' or x['domain']==domain})
        summaries[domain] = {}
        for condition in reg['conditions'] + ['corruption']:
            cc = ['noise_medium','defocus_extreme'] if condition=='corruption' else [condition]
            table = []
            for k in keys:
                rr = [by[(k,c)] for c in cc]
                arms = {a:{m:float(np.mean([r['arms'][a][m] for r in rr]))
                           if all(r['arms'][a][m] is not None for r in rr) else None
                           for m in ['v','s','t']} for a in ['Frozen','Sourcefit','TTA']}
                table.append({'key':k,'source':rr[0]['source'],'domain':rr[0]['domain'],
                              'condition':condition,'arms':arms})
            result = {'parents':len(table),'absolute':{},'contrasts':{}}
            for arm in ['Frozen','Sourcefit','TTA']:
                result['absolute'][arm] = {m:stats([x['arms'][arm][m] for x in table if x['arms'][arm][m] is not None])
                                            if any(x['arms'][arm][m] is not None for x in table) else None for m in ['v','s','t']}
            for a,b in [('TTA','Sourcefit'),('TTA','Frozen'),('Sourcefit','Frozen')]:
                result['contrasts'][a+'-'+b] = {m:stats([x['arms'][a][m]-x['arms'][b][m] for x in table
                                                    if x['arms'][a][m] is not None and x['arms'][b][m] is not None])
                                                  if any(x['arms'][a][m] is not None for x in table) else None for m in ['v','s','t']}
            relevant = [by[(k,c)] for k in keys for c in cc]
            result['format_ok'] = {a:sum(r['arms'][a]['format_ok'] for r in relevant) for a in ['Frozen','Sourcefit','TTA']}
            result['condition_query_count'] = len(relevant)
            result['native_good_frames'] = sum(r['native_good_frames'] for r in relevant)
            result['native_good_retained'] = {a:sum(r['native_good_retained'][a] for r in relevant) for a in ['Frozen','Sourcefit','TTA']}
            result['GT_valid_frames'] = sum(r['arms']['Frozen']['GT_valid_positions'] for r in relevant)
            result['valid_spatial_parents'] = sum(x['arms']['Frozen']['s'] is not None for x in table)
            summaries[domain][condition] = result
            if domain=='all':parent_tables.extend(table)
    write_once(SCORE / 'PARENT_TABLES.json',parent_tables)
    write_once(SCORE / 'SUMMARY.json', {
        'time':time.time(), 'status':'completed_offline_target_development_readout', 'groups':summaries,
        'primary':'TTA-Sourcefit vIoU; noise/blur averaged within parent then all64 parent macro',
        'bootstrap':{'n':10000,'seed':20260927,'unit':'parent','reset_per_statistic':True},
        'optimization_seeds':1,'fresh_test':False,'all_GT_read_after_192_pair_barrier':True,
        'independent_scalar_vs_original_P3_metric_max_error':max_error,
        'scorer_sha256':sha(__file__),'production_promoted':False})
    print(json.dumps({'primary':summaries['all']['corruption']['contrasts'],
                      'clean':summaries['all']['clean']['contrasts'],'metric_crosscheck_max_error':max_error},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['self-check','score'])
    args=parser.parse_args()
    print(json.dumps(synthetic())) if args.action=='self-check' else run()

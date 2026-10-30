"""Post-seal descriptive decomposition of already-scored R8 predictions.

Does not read target data/GT or change a prediction. Original row identities and
tail lists stay private; public tables may alias identities without the mapping.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


RGB = 'rgb_fulltext__'
CONTRASTS = {
    'association_minus_source': (RGB+'association_only', 'source'),
    'direct_minus_association': (RGB+'direct_G', RGB+'association_only'),
    'param1_minus_association': (RGB+'param1', RGB+'association_only'),
    'param3_minus_association': (RGB+'param3', RGB+'association_only'),
    'param1_minus_direct': (RGB+'param1', RGB+'direct_G'),
    'param3_minus_direct': (RGB+'param3', RGB+'direct_G'),
    'param1_minus_outputspace': (RGB+'param1', RGB+'outputspace1'),
    'param3_minus_outputspace': (RGB+'param3', RGB+'outputspace3'),
    'RGB_association_minus_PC': (RGB+'association_only', 'pc_only__association_only'),
    'RGB_association_minus_shuffle': (RGB+'association_only', 'semantic_shuffle__association_only'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(rows, names):
    a, b = names
    av = np.array([r['arms'][a]['iou'] for r in rows])
    bv = np.array([r['arms'][b]['iou'] for r in rows])
    d = av-bv
    scene_values = collections.defaultdict(list)
    for r, value in zip(rows, d):
        scene_values[r['scene']].append(float(value))
    return {
        'n': len(rows),
        'defined_scene_clusters': len(scene_values),
        'delta_mean': float(d.mean()) if len(d) else None,
        'equal_scene_delta_mean': float(np.mean([np.mean(v) for v in scene_values.values()])) if scene_values else None,
        'gain_atol1e_minus8': int((d>1e-8).sum()),
        'harm_atol1e_minus8': int((d<-1e-8).sum()),
        'tie_atol1e_minus8': int((np.abs(d)<=1e-8).sum()),
        'delta_quantiles': np.quantile(d, [0, .01, .1, .5, .9, .99, 1]).tolist() if len(d) else None,
        'strict_Acc_transitions': {
            str(t): {'gained': int(((av>t)&(bv<=t)).sum()),
                     'lost': int(((av<=t)&(bv>t)).sum()),
                     'retained_success': int(((av>t)&(bv>t)).sum()),
                     'retained_failure': int(((av<=t)&(bv<=t)).sum())}
            for t in (.25, .5)
        },
    }


def main(args):
    scored = args.scored.resolve()
    seal = json.loads((scored/'SEALED.json').read_text())
    for name, digest in seal['files'].items():
        path = (scored/name).resolve()
        assert path.is_relative_to(scored) and sha(path)==digest
    data = json.loads((scored/'RESULTS_PRIVATE.json').read_text())
    assert data['state']=='R8_scored_after_fullstage_prediction_seal'
    plan = json.loads(args.plan.read_text())
    assert plan['no_changes_to_G_or_predictions'] and plan['GT_is_offline_only']
    rows = data['rows_PRIVATE']
    out = {'state': 'descriptive_postseal_diagnostic_no_configuration_selection',
           'scored_seal_sha256': sha(scored/'SEALED.json'), 'plan_sha256': sha(args.plan),
           'program_sha256': sha(__file__), 'n': len(rows), 'platforms': {},
           'limits': 'IoU coverage categories do not prove instance identity; 2 Drone/3 Quad scenes, target research-exposed.'}
    tails = {}
    for platform in ('drone', 'quad'):
        rr = [r for r in rows if r['platform']==platform]
        scenes = sorted({r['scene'] for r in rr})
        groups = {'all': rr,
                  'q_changed': [r for r in rr if r['G']['rgb_fulltext']['q_changed']],
                  'q_unchanged': [r for r in rr if not r['G']['rgb_fulltext']['q_changed']],
                  'actual_pseudo_nonzero': [r for r in rr if r['G']['rgb_fulltext']['actual_float32_pseudo_delta_nonzero']],
                  'actual_pseudo_zero': [r for r in rr if not r['G']['rgb_fulltext']['actual_float32_pseudo_delta_nonzero']],
                  'common_eligible': [r for r in rr if all(g['accepted_axis'] for g in r['G'].values())]}
        for lo, hi in ((0, 0), (0, .05), (.05, .2), (.2, .5), (.5, 1)):
            groups['r_global_'+str(lo)+'_'+str(hi)] = [r for r in rr if
                (r['G']['rgb_fulltext']['r_global']==0 if hi==0 else lo<r['G']['rgb_fulltext']['r_global']<=hi)]
        result = {'n': len(rr), 'scene_clusters': len(scenes),
                  'subgroups': {k: {name: stats(v, ab) for name, ab in CONTRASTS.items()} for k, v in groups.items()},
                  'scene_rows_PRIVATE': {s: {name: stats([r for r in rr if r['scene']==s], ab) for name, ab in CONTRASTS.items()} for s in scenes},
                  'GT_assisted_coverage_categories': {},
                  'coverage_scope': 'G-path coverage staircase, not actual failure or instance identity. A source-q0 fallback may already hit despite a prepared/ROI-path miss.'}
        for threshold in (.25, .5):
            c = collections.Counter()
            for r in rr:
                o = r['cluster_oracles_PRIVATE']['rgb_fulltext']
                def hit(key):
                    v = o[key]
                    return v is not None and v>threshold
                if r['source_coverage']['256']['best_iou']<=threshold:
                    category='no_source256_hit'
                elif not hit('all_prepared_group_pool_best_iou_GTassisted'):
                    category='source_hit_but_prepared_pool_miss'
                elif not hit('semantic_ROI_pool_best_iou_GTassisted'):
                    category='prepared_hit_but_ROI_pool_miss'
                elif not hit('selected_group_best_iou_GTassisted'):
                    category='ROI_hit_but_selected_group_miss'
                elif r['arms'][RGB+'association_only']['iou']<=threshold:
                    category='selected_group_hit_but_q1_anchor_miss'
                else:
                    category='q1_anchor_hit'
                c[category]+=1
            result['GT_assisted_coverage_categories'][str(threshold)] = dict(c)
        out['platforms'][platform] = result
        tails[platform] = {}
        for name in ('association_minus_source', 'direct_minus_association', 'param3_minus_association', 'param3_minus_direct'):
            a, b = CONTRASTS[name]
            delta = lambda r: r['arms'][a]['iou']-r['arms'][b]['iou']
            negative = sorted([r for r in rr if delta(r)<-1e-8], key=lambda r: (delta(r), r['id']))
            positive = sorted([r for r in rr if delta(r)>1e-8], key=lambda r: (-delta(r), r['id']))
            def compact(r):
                return {'id': r['id'], 'scene': r['scene'], 'arm_a': a, 'arm_b': b,
                        'iou_a': r['arms'][a]['iou'], 'iou_b': r['arms'][b]['iou'],
                        'delta': r['arms'][a]['iou']-r['arms'][b]['iou'], 'G': r['G']['rgb_fulltext']}
            tails[platform][name]={'negative_available': len(negative), 'positive_available': len(positive),
                                   'negative': [compact(r) for r in negative[:10]],
                                   'positive': [compact(r) for r in positive[:10]]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'DIAGNOSTIC_PRIVATE.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    (args.output/'SIGNED_TAILS_PRIVATE.json').write_text(json.dumps(tails, indent=2, allow_nan=False)+'\n')
    public = dict(out)
    public['platforms'] = {p: {k: v for k, v in val.items() if k!='scene_rows_PRIVATE'} for p, val in out['platforms'].items()}
    (args.output/'SCALAR_DIAGNOSTIC.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    final = {'files': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}, 'n': len(rows)}
    (args.output/'SEALED.json').write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps({'n': len(rows), 'diagnostic_seal_sha256': sha(args.output/'SEALED.json')}))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    for arg in ('scored', 'plan', 'output'):
        ap.add_argument('--'+arg, type=Path, required=True)
    main(ap.parse_args())

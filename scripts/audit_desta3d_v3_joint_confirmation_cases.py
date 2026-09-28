"""Read every sealed confirmation case; no new labels or GPU evaluation."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crosscheck_desta3d_v3_joint_confirmation import D, E, ARMS, METRICS, read, sha, write


def audit():
    import torch
    torch.set_num_threads(4)
    start = time.monotonic()
    dest = E / 'independent_readback_v1'
    out = dest / 'ROOT_NATIVE_CASE_READBACK.json'
    assert not out.exists()
    assert read(dest/'ROOT_SUMMARY_CROSSCHECK.json')['status'] == 'passed'
    report = read(dest/'REPORT.json')
    assert read(dest/'COMPLETE.json')['report_sha'] == sha(dest/'REPORT.json')
    manifest = read(E/'PREDICTIONS_SEAL.json')['files']
    cases, counts = {a: [] for a in ARMS[1:]}, {}
    format_failures, invalid_geometry_frames = {a: 0 for a in ARMS}, {a: 0 for a in ARMS}
    norms = {a: [] for a in ARMS}
    for i, row in enumerate(read(D/'VALIDATION_INPUTS.json')):
        predictions = {}
        for a in ARMS:
            path = E/'episodes'/f'{i:04}'/(a+'.pt')
            assert sha(path) == manifest[str(path.relative_to(E))]
            p = torch.load(path, map_location='cpu', weights_only=False)
            assert p['key'] == row['key'] and p['source'] == row['source']
            format_failures[a] += not p['format_ok']
            invalid_geometry_frames[a] += int((~p['geometry_valid'].bool()).sum())
            norms[a].append(p['injection']['relative_norm'])
            predictions[a] = p
        b = predictions['B1']
        for a in ARMS[1:]:
            p = predictions[a]
            br, pr = report['arms']['B1']['rows'][i], report['arms'][a]['rows'][i]
            assert br['key'] == pr['key'] == row['key']
            bp = {int(pos): box for pos, box in zip(b['positions'], b['boxes_cxcywh'])}
            pp = {int(pos): box for pos, box in zip(p['positions'], p['boxes_cxcywh'])}
            common = sorted(bp.keys() & pp.keys())
            c = dict(index=i, key=row['key'], source=row['source'],
                     base_interval=b['interval'], candidate_interval=p['interval'],
                     interval_changed=b['interval'] != p['interval'],
                     stored_reference_token_support_equal=b['readout']['spatial_reference_token_ids'] == p['readout']['spatial_reference_token_ids'],
                     both_spatial_passes_present=b['readout']['spatial_injection'] is not None and p['readout']['spatial_injection'] is not None,
                     box_positions_changed=b['positions'] != p['positions'],
                     common_box_positions=len(common),
                     changed_boxes_on_common_positions=sum(not torch.equal(bp[k], pp[k]) for k in common),
                     base_format_ok=b['format_ok'], candidate_format_ok=p['format_ok'],
                     delta_pp={m: 100*(pr['metrics'][m]-br['metrics'][m]) for m in METRICS})
            cases[a].append(c)
    for a, cs in cases.items():
        counts[a] = dict(queries=len(cs), interval_changed=sum(c['interval_changed'] for c in cs),
                         comparable_reference_pairs=sum(c['both_spatial_passes_present'] for c in cs),
                         reference_tokens_changed_on_comparable_pairs=sum(c['both_spatial_passes_present'] and not c['stored_reference_token_support_equal'] for c in cs),
                         box_support_changed=sum(c['box_positions_changed'] for c in cs),
                         unchanged_interval_with_changed_common_boxes=sum(not c['interval_changed'] and c['changed_boxes_on_common_positions'] > 0 for c in cs),
                         query_v_delta_below_minus5pp=sum(c['delta_pp']['vIoU'] < -5 for c in cs),
                         query_v_delta_above5pp=sum(c['delta_pp']['vIoU'] > 5 for c in cs))
    write(out, dict(status='passed', all_queries_retained=447, counts=counts, cases=cases,
                    format_failures=format_failures, invalid_geometry_frames=invalid_geometry_frames,
                    relative_norm_range={a:[min(x), max(x)] for a,x in norms.items()},
                    scope='Saved native interval/reference/box comparisons across all queries; query tails are not parent tails. No new GT. Same-interval changes alone do not identify a causal branch.',
                    report_sha=sha(dest/'REPORT.json'), auditor_sha=sha(Path(__file__)), CPU_seconds=time.monotonic()-start))
    print(json.dumps({k:v for k,v in read(out).items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    audit()

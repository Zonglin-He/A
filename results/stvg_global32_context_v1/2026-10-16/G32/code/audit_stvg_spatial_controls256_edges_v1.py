"""Actual B/C retained graph accounting, fixed256 parents, no GT or model."""
from pathlib import Path
import collections, hashlib, json, math, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE, read, put, sha

def measures(records, available):
    out = []
    for i, r in enumerate(records):
        edges = [(int(t)*256+int(u), (int(t)+1)*256+int(v), float(w))
                 for t, u, v, w, _ in r['edges'] if float(w) > 0]
        assert all(math.isfinite(w) for _, _, w in edges)
        links = collections.defaultdict(list)
        for u, v, w in edges:
            links[u].append((v, w))
        paths = [(u, v, z, min(w, q)) for u, v, w in edges for z, q in links[v]] if i == 1 else []
        denominator = math.fsum(w for _, _, w in edges)
        pathden = math.fsum(w for _, _, _, w in paths)
        for arm in ('B', 'C'):
            keep = set(r['indices'][arm]) if available else set()
            numer = math.fsum(w for u, v, w in edges if u in keep and v in keep)
            pathnum = math.fsum(w for u, v, z, w in paths if u in keep and v in keep and z in keep)
            out.append(dict(view='context' if i == 0 else 'local', arm=arm,
                actual_used_evidence=available, edge_count=len(edges), total_edge_weight=denominator,
                retained_edge_count=sum(u in keep and v in keep for u, v, _ in edges) if available else None,
                retained_edge_ratio=numer/denominator if available and denominator else None,
                dangling_edge_count=sum((u in keep) != (v in keep) for u, v, _ in edges) if available else None,
                two_edge_path_count=len(paths) if i == 1 else None,
                retained_path_weight_ratio=pathnum/pathden if available and pathden else None,
                GT_read=False))
    return out

def lock():
    put(BASE/'EDGE_AUDIT_LOCK.json', dict(status='locked', count=256, GT_read=False,
        pins={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), BASE/'ROSTER_LOCK.json', BASE/'DESIGN_LOCK.json')},
        edge='original positive same-candidate reciprocal projected-feature graph; both actual kept endpoints',
        path='two consecutive edges sharing the same middle vertex; weight min of two edges; all three retained',
        missing='no job, no caption, or no graph means unknown; never evaluate proxy-caption B/C as used evidence',
        aggregation='mean known windows within parent, then equal-parent mean by domain; support counts and nulls explicit',
        construction_search=False, paired_C_extra_run=False, bootstrap=False))
    print('EDGE_AUDIT_LOCKED_256_NO_GT')

def main():
    for p, h in read(BASE/'EDGE_AUDIT_LOCK.json')['pins'].items():
        assert sha(ROOT/p) == h, p
    roster = read(BASE/'ROSTER_LOCK.json')['records']
    for arm in ('B', 'C'):
        root = read(BASE/'teacher'/arm/'ROOT_READBACK.json')
        bar = read(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json')
        assert root['status'] == 'pass' and root['count'] == 256 and not root['GT_read']
        assert bar['status'] == 'complete' and bar['count'] == 256 and not bar['GT_read']
        assert root['barrier_sha256'] == sha(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json')
        for f in bar['files']:
            assert sha(ROOT/f['path']) == f['sha256']
    rows = []
    for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
        p = BASE/'native_cache'/j['job_id']/'POLICIES.json'
        policy = read(p)
        available = policy['caption_available']
        for arm in ('B', 'C'):
            d = read(BASE/'teacher'/arm/(j['job_id']+'.json'))
            assert d['native_policy_sha256'] == sha(p)
            assert d['new_teacher_call'] is available
            ix = [r['indices'][arm] for r in policy['records']] if available else []
            assert hashlib.sha256(json.dumps(ix).encode()).hexdigest() == d['visual_indices_sha256']
        rows.extend(dict(r, case_id=j['case_id'], dataset=j['dataset'], slot=j['center_slot'])
                    for r in measures(policy['records'], available))
    source_rows, summary, paired = [], [], []
    for ds in ('hc2', 'vidstg'):
        ids = [r['probe_alias'] for r in roster if r['dataset'] == ds]
        assert len(ids) == 128
        for view in ('context', 'local'):
            for arm in ('B', 'C'):
                for cid in ids:
                    rr = [x for x in rows if x['case_id'] == cid and x['view'] == view and x['arm'] == arm]
                    assert len(rr) <= 4
                    values = [x['retained_edge_ratio'] for x in rr if x['retained_edge_ratio'] is not None]
                    paths = [x['retained_path_weight_ratio'] for x in rr if x['retained_path_weight_ratio'] is not None]
                    source_rows.append(dict(case_id=cid, dataset=ds, view=view, arm=arm,
                        known_windows=len(values), unknown_windows=4-len(values),
                        absent_windows=4-len(rr), unavailable_caption_windows=sum(not x['actual_used_evidence'] for x in rr),
                        retained_edge_ratio=math.fsum(values)/len(values) if values else None,
                        retained_path_weight_ratio=math.fsum(paths)/len(paths) if paths else None))
                sr = [x for x in source_rows if x['dataset'] == ds and x['view'] == view and x['arm'] == arm]
                for key in ('retained_edge_ratio', 'retained_path_weight_ratio'):
                    v = [x[key] for x in sr if x[key] is not None]
                    summary.append(dict(dataset=ds, view=view, arm=arm, metric=key, known_sources=len(v),
                        unknown_sources=128-len(v), mean=math.fsum(v)/len(v) if v else None))
            idx = {(x['case_id'], x['arm']): x for x in source_rows if x['dataset'] == ds and x['view'] == view}
            for key in ('retained_edge_ratio', 'retained_path_weight_ratio'):
                v = {cid: idx[cid, 'C'][key]-idx[cid, 'B'][key] for cid in ids
                     if idx[cid, 'C'][key] is not None and idx[cid, 'B'][key] is not None}
                paired.append(dict(dataset=ds, view=view, metric=key, known_sources=len(v), unknown_sources=128-len(v),
                    C_minus_B=math.fsum(v.values())/len(v) if v else None, per_source_delta=v))
    put(BASE/'EDGE_AUDIT_CPU_COMPLETION.json', dict(status='complete', count=256, window_count=len(rows)//4,
        rows=rows, source_rows=source_rows, summary=summary, paired=paired, GT_read=False, CPU_only=True,
        paired_C_extra_run=False, limitation='feature graph preservation is not object identity, role correctness or Student benefit'))
    print('EDGE_AUDIT_COMPLETE_256_NO_GT')

def contracts():
    r = dict(edges=[[0, 0, 1, 2., 0], [1, 1, 2, 3., 0]], indices={'B': [0, 257], 'C': [0, 257, 514]})
    v = measures([r, r], True)
    assert v[2]['retained_edge_ratio'] == .4 and v[3]['retained_edge_ratio'] == 1
    assert v[2]['retained_path_weight_ratio'] == 0 and v[3]['retained_path_weight_ratio'] == 1
    assert all(x['retained_edge_ratio'] is None for x in measures([r, r], False))
    print('EDGE_CONTRACTS_PASS_ACTUAL_ENDPOINT_PATH_AND_NO_CAPTION_UNKNOWN')

if __name__ == '__main__':
    {'lock': lock, 'contracts': contracts, 'run': main}[sys.argv[1]]()

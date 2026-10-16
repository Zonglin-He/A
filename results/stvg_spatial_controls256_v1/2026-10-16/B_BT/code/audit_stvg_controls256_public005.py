"""Independent post-seal scalar, anonymity and public-manifest audit; no GT reader."""
from pathlib import Path
import collections,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_soft_ep_v1 as privacy
def main(arm):
    out=ROOT/'results/stvg_spatial_controls256_v1/2026-10-16'/arm
    d=json.loads((out/'RESULTS.json').read_text());cells=0;pairs=0
    roster=json.loads((ROOT/'artifacts/stvg_spatial_controls256_v1/ROSTER_LOCK.json').read_text())['records']
    original_id={f'R{i:03d}':r['probe_alias'] for i,r in enumerate(roster,1)}
    for prefix in ('','actual_support_'):
        rows=d[prefix+'rows'];ix={(r['dataset'],r['case_id'],r['arm']):r['metrics'] for r in rows}
        assert len(ix)==len(rows)
        assert {r['case_id'] for r in rows}=={f'R{i:03d}' for i in range(1,257)}
        for ds in ('hc2','vidstg'):
            assert len({r['case_id'] for r in rows if r['dataset']==ds})==128
        for s in d[prefix+'summary']:
            vals=[r['metrics'][s['metric']] for r in rows if r['dataset']==s['dataset'] and r['arm']==s['arm'] and r['metrics'][s['metric']] is not None]
            assert s['valid']==len(vals) and s['unknown']==128-len(vals)
            assert (s['mean'] is None and not vals) or abs(s['mean']-math.fsum(vals)/len(vals))<1e-13
            cells+=1
        for p in d[prefix+'paired']:
            a,b=p['comparison'].split('_minus_');ds=p['dataset'];m=p['metric']
            ids={r['case_id'] for r in rows if r['dataset']==ds}
            delta={cid:100*(ix[ds,cid,a][m]-ix[ds,cid,b][m]) for cid in ids if ix[ds,cid,a][m] is not None and ix[ds,cid,b][m] is not None}
            assert delta==p['per_source_delta_pp'] and p['valid']==len(delta) and p['unknown']==128-len(delta)
            assert p['positive']==sum(x>1e-6 for x in delta.values()) and p['negative']==sum(x< -1e-6 for x in delta.values()) and p['neutral']==sum(abs(x)<=1e-6 for x in delta.values())
            assert (p['mean_delta_pp'] is None and not delta) or abs(p['mean_delta_pp']-math.fsum(delta.values())/len(delta))<1e-11
            assert p['worst_sources']==sorted(delta,key=lambda c:(delta[c],original_id[c]))[:8]
            pairs+=1
    manifest=ROOT/'artifacts/stvg_spatial_controls256_v1/publication'/arm/'EXPORT_MANIFEST.json'
    md=json.loads(manifest.read_text());paths=set(md['paths']);privacy.verify_privacy(paths)
    roster=json.loads((ROOT/'artifacts/stvg_spatial_controls256_v1/ROSTER_LOCK.json').read_text())['records']
    for rel in paths:
        p=ROOT/rel
        if p.suffix in ('.json','.md'):
            raw=p.read_text()
            for rec in roster:assert ('"'+rec['probe_alias']+'"') not in raw,rel
    receipt=dict(status='pass',count=256,CPU_only=True,GT_read=False,new_GT_reader_calls=0,new_model_calls=0,summary_cells_checked=cells,paired_cells_checked=pairs,all_signs_and_unknowns_preserved=True,strict_roster_anonymity=True,actual_figures_viewed_separately=True,export_recovery='005 only accepts already-anonymous identities bound to the same roster; unknown identities still fail; original partial exports retained')
    dest=out/'ROOT_SCALAR_PRIVACY_AUDIT005.json';dest.write_text(json.dumps(receipt,indent=2)+'\n');paths.add(str(dest.relative_to(ROOT)))
    for name in ('audit_stvg_controls256_public005.py',):
        p=out/'code'/name;p.write_bytes((ROOT/'scripts'/name).read_bytes());paths.add(str(p.relative_to(ROOT)))
    md.update(paths=sorted(paths),files=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)])
    manifest.write_text(json.dumps(md,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main(sys.argv[1])

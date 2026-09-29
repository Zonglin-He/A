"""Recompute public Round1 counts and mean preservation from anonymous scalar rows."""
import argparse,json
from pathlib import Path

def audit(folder):
    read=lambda p:json.loads(p.read_text())
    index=read(folder/'ROWS_INDEX.json');rows=[]
    for part in index['files']:
        x=read(folder/part['path']);assert len(x)==part['rows'];rows+=x
    assert len(rows)==192 and len({r['key'] for r in rows})==64
    assert len({(r['key'],r['rho']) for r in rows})==192
    summary=read(folder/'SUMMARY.json');decision=read(folder/'DECISION.json')
    unique=set();better=set()
    for r in rows:
        strong=any(v is not None and ((k.endswith('_JSD') and v>=.01) or ('cosine_drift' in k and v>=.10) or (k.endswith('_jaccard') and v<=.5)) for m in r['metrics'] for k,v in m.items())
        strong=bool(strong and r['preservation']['strict'] and r['preservation']['self_vIoU']>.95)
        assert strong==r['strong_preserved']
        if strong:unique.add(r['key'])
        if strong and r['optimized_over_feasible_random']:better.add(r['key'])
    assert len(unique)==decision['strong_preserved_queries']==25
    assert len(better)==decision['strong_optimized_over_random']==25
    for cohort in ('all','hcstvg1_test','vidstg_test'):
        for rho in (.005,.01,.02):
            rr=[r for r in rows if r['rho']==rho and (cohort=='all' or r['cohort']==cohort)]
            s=summary[f'{cohort}/rho{rho:g}'];assert len(rr)==s['queries']
            assert sum(r['strong_preserved'] for r in rr)==s['strong_preserved']
            assert sum(r['terminal_preservation']['strict'] for r in rr)==s['terminal_strict']
            assert sum(r['random']['preservation']['strict'] for r in rr)==s['random_strict']
            assert abs(sum(r['preservation']['self_vIoU'] for r in rr)/len(rr)-s['self_vIoU']['mean'])<1e-12
    return dict(status='pass',queries=64,arms=192,strong_unique=25,raw_tensors_read=False,scope='anonymous scalar aggregation, not model inference or independent tensor readback')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();print(json.dumps(audit(args.folder),indent=2))

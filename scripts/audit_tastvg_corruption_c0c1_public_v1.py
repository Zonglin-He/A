"""Rebuild public C0/C1 aggregates and resource gates from anonymous rows."""
import argparse,json,collections
from pathlib import Path
import numpy as np


def audit(folder):
    read=lambda f:json.loads((folder/f).read_text());index=read('ROWS_INDEX.json');groups={};checks=0
    for name,parts in index['groups'].items():
        rr=[]
        for p in parts['files']:
            z=read(p['path']);assert len(z)==p['rows'];rr+=z
        assert len(rr)==parts['rows'];groups[name]=rr
    assert len(groups['C0_ROWS'])==224 and len(groups['C1_ROWS'])==64
    def eq(a,b):
        nonlocal checks
        assert abs(a-b)<1e-11,(a,b);checks+=1
    c0=read('C0_SUMMARY.json');c1=read('C1_SUMMARY.json');decision=read('DECISION.json')
    for cond,z in c0.items():
        rr=[r for r in groups['C0_ROWS'] if r['condition']==cond];assert len(rr)==32
        assert dict(collections.Counter(r['failure_group'] for r in rr))==z['failure_groups']
        for metric in ['sIoU','tIoU','vIoU_corrected']:
            eq(np.mean([r['native'][metric] for r in rr]),z['absolute'][metric]['mean']);eq(np.mean([r['delta'][metric] for r in rr]),z['delta'][metric]['mean'])
    for cond,z in c1.items():
        rr=[r for r in groups['C1_ROWS'] if (r['condition']!='clean' if cond=='corrupted_parent_macro' else r['condition']==cond)]
        for branch,metric,cs in [('S','sIoU','spatial_candidates'),('T','tIoU','temporal_candidates')]:
            parent=collections.defaultdict(list)
            for r in rr:
                best=max(x['metrics'][metric] for x in r[cs]);eq(best,r['arms'][branch+'_oracle'][metric]);eq(best-r['arms']['native'][metric],r['gain_'+branch]);parent[r['key']].append(r['gain_'+branch])
            values=np.array([np.mean(parent[k]) for k in index['C1_parent_bootstrap_order'] if k in parent]);eq(values.mean(),z['gain_'+branch]['mean']);assert sum(values>.05)==z['opportunity_gt5pp'][branch]
            if cond=='corrupted_parent_macro':
                rng=np.random.default_rng(20260929);ci=np.quantile(values[rng.integers(16,size=(10000,16))].mean(1),[.025,.975]);assert np.max(np.abs(ci-z['gain_'+branch]['ci95']))<1e-11
                passed=values.mean()>=.02 and ci[0]>0 and sum(values>.05)>=4;assert bool(passed)==decision['branches'][branch]['candidate_support_gate']
    return dict(status='pass',C0_cells=224,C1_cells=64,aggregate_checks=checks,scope='anonymous scalar reconstruction; no raw predictions or inference')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();print(json.dumps(audit(args.folder),indent=2))

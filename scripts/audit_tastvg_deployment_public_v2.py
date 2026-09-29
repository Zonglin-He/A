"""Reconstruct published means/CIs/decisions from anonymous scalar rows."""
import argparse,collections,itertools,json
from pathlib import Path
import numpy as np

def audit(root):
    read=lambda n:json.loads((root/n).read_text())
    checks=0
    def stat(values,z):
        nonlocal checks
        a=np.array(values,float);assert len(a)==z['n']
        if not len(a):assert z['mean'] is None;checks+=1;return
        assert abs(a.mean()-z['mean'])<1e-12
        rng=np.random.default_rng(20260929);ci=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975])
        assert np.allclose(ci,z['ci95'],atol=1e-12,rtol=0);checks+=1
    def vals(rr,fn):
        d=collections.defaultdict(list)
        for r in rr:
            v=fn(r)
            if v is not None:d[r['parent']].append(v)
        return [np.mean(d[k]) for k in sorted(d)]
    metrics=['sIoU','tIoU','vIoU_corrected'];c0=read('C05_ROWS.json');s0=read('C05_SUMMARY.json');assert len(c0)==480
    for name,z in s0.items():
        rr=c0 if name=='panel' else [r for r in c0 if name in [r['family'],r['condition']]]
        for m in metrics:stat(vals(rr,lambda r:r['delta'][m]),z['delta'][m])
        assert sum(r['observed_burst_frames']==0 for r in rr)==z['zero_observed_hit']
    c2=read('C2_ROWS.json');s2=read('C2_SUMMARY.json');decision=read('C2_DECISION.json');assert len(c2)==64
    gaps=[]
    for r in c2:
        scores=r['candidate_scores'];values=r['candidate_metrics'];i=r['selected'];assert i==int(np.argmax(scores));assert r['oracle']==int(np.argmax([v['tIoU'] for v in values]))
        assert r['arms']['Native']==values[0] and r['arms']['Expert']==values[i] and r['arms']['Oracle']==values[r['oracle']]
        for m in metrics:
            assert abs(r['arms']['Uniform'][m]-np.mean([v[m] for v in values]))<1e-12
            for a in ['Expert','Oracle','Uniform']:assert abs(r['delta'][a][m]-(r['arms'][a][m]-r['arms']['Native'][m]))<1e-12
        gaps.extend(abs(scores[i]-scores[j]) for i,j in itertools.combinations(range(len(scores)),2))
    cuts=np.quantile(gaps,[1/3,2/3]);assert np.allclose(cuts,decision['margin_terciles'],atol=1e-12,rtol=0)
    for r in c2:
        d=collections.defaultdict(list);scores=r['candidate_scores'];values=r['candidate_metrics'];ties=0
        for i,j in itertools.combinations(range(len(scores)),2):
            dg=values[i]['tIoU']-values[j]['tIoU'];de=scores[i]-scores[j];gap=abs(de)
            if abs(dg)<=1e-12:ties+=1;continue
            acc=.5 if gap<=1e-12 else float((dg>0)==(de>0));name='low' if gap<=cuts[0] else 'middle' if gap<=cuts[1] else 'high';d[name].append(acc);d['all'].append(acc)
        for b in ['all','low','middle','high']:
            assert len(d[b])==r['pair_count'][b]
            if d[b]:assert abs(np.mean(d[b])-r['pair_accuracy'][b])<1e-12
            else:assert r['pair_accuracy'][b] is None
        assert ties==r['GT_tied_pairs_excluded']
    for name,z in s2.items():
        rr=[r for r in c2 if r['condition']!='clean'] if name=='corrupted_parent_macro' else [r for r in c2 if r['condition']==name]
        for a in ['Native','Expert','Oracle','Uniform']:
            for m in metrics:stat(vals(rr,lambda r:r['arms'][a][m]),z['arms'][a][m])
        for a in ['Expert','Oracle','Uniform']:
            for m in metrics:stat(vals(rr,lambda r:r['delta'][a][m]),z['delta'][a][m])
        for b in ['all','low','middle','high']:stat(vals(rr,lambda r:r['pair_accuracy'][b]),z['pair_accuracy'][b])
        for m in metrics:stat(vals(rr,lambda r:r['arms']['Expert'][m]-r['arms']['Uniform'][m]),z['expert_minus_uniform'][m])
    clean={r['parent']:r for r in c2 if r['condition']=='clean'};rr=[r for r in c2 if r['condition']!='clean'];excess=read('C2_EXCESS.json')
    for m in metrics:stat(vals(rr,lambda r:r['delta']['Expert'][m]-clean[r['parent']]['delta']['Expert'][m]),excess[m])
    gate0=all(s0['panel']['delta'][m]['ci95'][1]<0 for m in ['tIoU','vIoU_corrected']);assert gate0==read('C05_DECISION.json')['benchmark_development_gate']
    p=s2['corrupted_parent_macro'];hi=p['pair_accuracy']['high']['ci95'];gate2=bool(p['delta']['Expert']['tIoU']['ci95'][0]>0 and hi and hi[0]>.5);assert gate2==decision['OPD_resource_gate']
    return dict(status='pass',bootstrap_scalar_checks=checks,c05_cells=len(c0),c2_cells=len(c2),candidate_selection_pairwise_rebuilt=True,C05_gate=gate0,C2_gate=gate2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();print(json.dumps(audit(args.directory),indent=2))

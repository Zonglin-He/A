"""NumPy-only reconstruction of all fixed-scale choices and public summaries."""
import argparse,json
from pathlib import Path
import numpy as np

def audit(root):
    read=lambda f:json.loads((root/f).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');reference=read('REFERENCE.json');checks=0;selections=0
    assert len(rows)==len({r['parent'] for r in rows})==24 and [r['position'] for r in rows]==sorted(r['position'] for r in rows)
    def stat(values,expected):
        nonlocal checks
        v=np.array(values,float);assert len(v)==expected['n'];assert abs(v.mean()-expected['mean'])<1e-12
        rng=np.random.default_rng(20260929);ci=np.quantile(v[rng.integers(len(v),size=(10000,len(v)))].mean(1),[.025,.975]);assert np.allclose(ci,expected['ci95'],atol=1e-12,rtol=0);checks+=1
    for r in rows:
        assert int(np.argmax(r['native_score']))==0 and int(np.argmax(r['full_scores']))==r['full_selected']
        assert r['full']==r['candidate_metrics'][r['full_selected']] and r['frozen']==r['candidate_metrics'][0]
        for a in ['1','8','16','32']:
            v=r['variants'][a];score=np.asarray(r['native_score'])+int(a)*np.asarray(r['residual']);assert np.allclose(score,v['scores'],atol=1e-12,rtol=0)
            k=int(score.argmax());assert k==v['selected'];assert v['metrics']==r['candidate_metrics'][k];assert v['agrees']==(k==r['full_selected']);assert v['changed_from_frozen']==(k!=0);selections+=1
    metrics=['sIoU','tIoU','vIoU_corrected']
    for a,s in summary.items():
        assert s['n']==24
        for key,fn in [('changed_from_frozen',lambda r:r['variants'][a]['selected']!=0),('changed_from_alpha1',lambda r:r['variants'][a]['selected']!=r['variants']['1']['selected']),('full_agreement',lambda r:r['variants'][a]['selected']==r['full_selected']),('toward_full',lambda r:r['variants'][a]['selected']==r['full_selected'] and r['variants']['1']['selected']!=r['full_selected']),('away_full',lambda r:r['variants'][a]['selected']!=r['full_selected'] and r['variants']['1']['selected']==r['full_selected'])]:assert sum(fn(r) for r in rows)==s[key]
        for m in metrics:
            stat([r['variants'][a]['metrics'][m] for r in rows],s['metrics'][m]);delta=[r['variants'][a]['metrics'][m]-r['variants']['1']['metrics'][m] for r in rows];stat(delta,s['delta_alpha1'][m])
            assert sum(d>1e-12 for d in delta)==s['positive'][m] and sum(d<-1e-12 for d in delta)==s['negative'][m] and sum(d<-.05 for d in delta)==s['harms_gt5pp'][m]
    for name,field in [('Frozen','frozen'),('Full Rerank','full')]:
        for m in metrics:stat([r[field][m] for r in rows],reference[name]['metrics'][m])
    return dict(status='pass',arrivals=24,selections=selections,bootstrap_scalar_checks=checks,all_agreements_and_harm_counts=True,conditional_fixed_trajectory=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);x=a.parse_args();print(json.dumps(audit(x.directory),indent=2))

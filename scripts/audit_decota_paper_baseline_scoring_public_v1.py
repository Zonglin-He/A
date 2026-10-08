"""Recompute all published baseline scalars from anonymous rows only, on CPU."""
import os,sys,gzip,json,time,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_decota_paper_baseline_scoring_v1 import independent_statistics

def run(directory):
    directory=Path(directory);summary=json.loads((directory/'SUMMARY.json').read_text());checks=0;logical=0
    for ds,methods in summary['datasets'].items():
        pairing=None
        for method,s in methods.items():
            with gzip.open(directory/ds/s['row_file'],'rt') as f:rows=[json.loads(line) for line in f]
            logical+=len(rows)
            keys={(r['query_id'],r['source_id'],r['order'],r['arrival']) for r in rows}
            assert len(keys)==len(rows) and len(rows)==s['logical_rows']
            if pairing is None:pairing=keys
            else:assert keys==pairing
            fields=list(s['metrics']);z=independent_statistics(rows,fields)
            q=collections.defaultdict(list)
            for r in rows:q[r['query_id']].append(r)
            data=np.array([[np.mean([r[f] for r in qr]) for f in fields] for qr in q.values()])
            for j,f in enumerate(fields):
                m=s['metrics'][f]
                assert abs(z['source_macro'][j]-m['source_macro'])<2e-12
                assert abs(z['query_macro'][j]-m['query_macro'])<2e-12
                assert np.allclose(z['ci'][:,j],m['ci95_source'],rtol=0,atol=2e-12)
                assert np.allclose(z['query_ci'][:,j],m['ci95_query_clustered'],rtol=0,atol=2e-12);checks+=6
                for order in ['order1','order2','order3']:
                    rr=[r for r in rows if r['order']==order];ss=collections.defaultdict(list)
                    for r in rr:ss[r['source_id']].append(r[f])
                    assert abs(np.mean([r[f] for r in rr])-m['orders'][order])<2e-12
                    assert abs(np.mean([np.mean(v) for v in ss.values()])-m['order_source_macro'][order])<2e-12;checks+=2
                if f.startswith('delta_'):
                    a=z['source_matrix'][:,j]
                    for threshold,suffix in [(-.05,'5pp'),(-.20,'20pp')]:
                        assert m['harm_gt'+suffix+'_sources']==int((a<threshold).sum())
                        assert m['harm_gt'+suffix+'_queries']==int((data[:,j]<threshold).sum());checks+=2
                    assert abs(m['gross_gain_pp']-np.maximum(a,0).mean()*100)<1e-10
                    assert abs(m['gross_loss_pp']+np.minimum(a,0).mean()*100)<1e-10;checks+=2
            for r in rows:
                assert abs(r['delta_total_v']-r['delta_inherited_v']-r['delta_current_v'])<2e-12
                assert abs(r['delta_total_v']-sum(r[k] for k in ['inherited_boxes_v','inherited_time_v','current_boxes_v','current_time_v']))<2e-12
                for t in [.3,.5]:
                    assert r['After_R'+str(t)]==float(r['After_v']>t)
                    assert r['correct_to_wrong_'+str(t)]==float(r['Frozen_v']>t and r['After_v']<=t)
                    assert r['wrong_to_correct_'+str(t)]==float(r['Frozen_v']<=t and r['After_v']>t)
                checks+=8
            print('PUBLIC_SCALARS_VERIFIED',ds,method,len(rows),flush=True)
    assert logical==summary['logical_rows']==217221
    return dict(status='pass',checks=checks,logical_rows=logical,GT_or_model_access=False,
        complete_order_and_query_pairing=True,source_query_means_CIs_tails_thresholds_verified=True)

if __name__=='__main__':
    result=run(sys.argv[1] if len(sys.argv)>1 else ROOT/'results/decota_paper_baseline_scoring/2026-10-08')
    print(json.dumps(result))

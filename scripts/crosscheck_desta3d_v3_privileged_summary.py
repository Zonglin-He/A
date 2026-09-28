"""Root independent reductions from scored per-query rows; no model or new labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import OUT,write,sha

def run(name):
    path=OUT/name/'independent_readback_v1';report=json.loads((path/'REPORT.json').read_text());checks=[]
    arms=report['arms'];metrics=('vIoU','sIoU','tIoU');parent={}
    def check(a,b):
        assert np.allclose(a,b,rtol=0,atol=1e-10),(a,b)
        checks.append(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    for arm,data in arms.items():
        parents=sorted({r['source'] for r in data['rows']});parent[arm]={}
        for metric in metrics:
            values=np.array([sum(r['metrics'][metric] for r in data['rows'] if r['source']==p)/sum(r['source']==p for r in data['rows']) for p in parents])
            parent[arm][metric]=values
            check(values.mean(),data['summary']['parent_macro'][metric])
            check(values,[r[metric] for r in sorted(data['summary']['parent_rows'],key=lambda r:r['source'])])
        assert len(parents)==16
    for arm in arms:
        if arm=='original':continue
        for metric in metrics:
            delta=parent[arm][metric]-parent['original'][metric];c=report['comparisons'][arm+'_minus_original'][metric]
            rng=np.random.default_rng(20260927);samples=rng.integers(0,16,size=(10000,16));b=(delta[samples].sum(axis=1)/16)*100
            check(delta.mean()*100,c['mean_delta_pp']);check(np.percentile(b,[2.5,97.5]),c['bootstrap_ci95_pp'])
            assert c['negative_parents']==sum(delta < -1e-10) and c['positive_parents']==sum(delta>1e-10)
            assert c['severe_loss_below_minus5pp']==sum(delta<-.05)
            check(delta*100,[c['parent_delta_pp'][p] for p in sorted(c['parent_delta_pp'])])
    for metric,record in report['retention'].items():
        good={r['key'] for r in arms['original']['rows'] if r['metrics'][metric]>.5}
        assert len(good)==record['eligible']
        for arm,data in arms.items():
            lost=[r['key'] for r in data['rows'] if r['key'] in good and r['metrics'][metric]<=.5]
            assert lost==record['arms'][arm]['lost'] and len(good)-len(lost)==record['arms'][arm]['retained']
    result={'status':'passed','report_sha':sha(path/'REPORT.json'),'checked_numeric_reductions':len(checks),'max_error':max(checks),
        'tail_retention_all_checked':True,'new_GPU':False,'new_GT_read':False,'scope':'Independent aggregation of scalar-scored rows; geometry independently tensor-checked by scorer'}
    write(path/'ROOT_SUMMARY_CROSSCHECK.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)

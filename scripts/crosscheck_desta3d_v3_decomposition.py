"""Second parent/CI/retention reduction, using only sealed scoring results."""
import argparse,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha

def run(name):
    o=OUT/name/'independent_readback_v1';r=read(o/'REPORT.json');assert read(o/'COMPLETE.json')['report_sha']==sha(o/'REPORT.json')
    errors=[];parents={};ms=['vIoU','sIoU','tIoU']
    def ck(a,b):
        aa=np.asarray(a);bb=np.asarray(b);e=float(np.max(np.abs(aa-bb)));assert e<1e-9;(errors.append(e))
    for a,x in r['arms'].items():
        pp=sorted({v['source'] for v in x['rows']});assert len(pp)==16
        parents[a]={m:np.array([sum(v['metrics'][m] for v in x['rows'] if v['source']==p)/sum(v['source']==p for v in x['rows']) for p in pp]) for m in ms}
        for m in ms:
            ck(np.mean(parents[a][m]),x['summary']['parent_macro'][m]);ck(parents[a][m],[v[m] for v in sorted(x['summary']['parent_rows'],key=lambda z:z['source'])])
    for key,c in r['comparisons'].items():
        a,b=key.split('_minus_')
        for m in ms:
            dd=parents[a][m]-parents[b][m];v=c[m];ix=np.random.default_rng(20260927).integers(0,16,size=(10000,16))
            ck(dd.sum()/16*100,v['mean_delta_pp']);ck(np.percentile(dd[ix].sum(1)/16*100,[2.5,97.5]),v['bootstrap_ci95_pp'])
            ck(dd*100,[v['parent_delta_pp'][p] for p in sorted(v['parent_delta_pp'])])
            assert sum(dd<-.05)==v['severe_loss_below_minus5pp'] and sum(dd>1e-10)==v['positive_parents'] and sum(dd< -1e-10)==v['negative_parents']
    for m,v in r['retention'].items():
        good={x['key'] for x in r['arms']['original']['rows'] if x['metrics'][m]>.5};assert len(good)==v['eligible']
        for a,x in r['arms'].items():
            lost=[x['key'] for x in x['rows'] if x['key'] in good and x['metrics'][m]<=.5]
            assert lost==v['arms'][a]['lost'] and len(good)-len(lost)==v['arms'][a]['retained']
    co=[x['gradient_cosine'] for x in r['geometry']];ck(sum(co)/16,r['gradient_summary']['mean']);assert sum(x<0 for x in co)==r['gradient_summary']['negative']
    for b in ['T','S']:
        for k in ['T','S','J','J_pass']:
            vals=[x['descent_dot'][b][k] for x in r['geometry']];v=r['local_descent_summary'][b+'_on_'+k]
            ck(sum(vals)/16,v['mean']);assert sum(x>0 for x in vals)==v['positive'] and sum(x<0 for x in vals)==v['negative']
    for key,v in r['finite_objective_summary'].items():
        vals=[x['finite'][key]['delta_CE'] for x in r['finite_objectives']];ck(sum(vals)/16,v['mean_delta_CE'])
        assert sum(x<0 for x in vals)==v['decreased'] and sum(x>0 for x in vals)==v['increased']
    out=dict(status='passed',report_sha=sha(o/'REPORT.json'),numeric_reductions=len(errors),max_error=max(errors),tail_retention_checked=True,new_GPU=False,new_GT_read=False)
    write(o/'ROOT_SUMMARY_CROSSCHECK.json',out);print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)

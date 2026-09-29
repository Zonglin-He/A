"""Independent parent/CI/tail/retention/gate reduction of sealed native scores."""
import json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
N=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/a03_shared_basis_v1/native'


def main():
    start=time.monotonic();dest=N/'independent_readback_v1';out=dest/'ROOT_SUMMARY_CROSSCHECK.json';assert not out.exists()
    report=json.loads((dest/'REPORT.json').read_text())
    assert json.loads((dest/'COMPLETE.json').read_text())['report_sha']==hashlib.sha256((dest/'REPORT.json').read_bytes()).hexdigest()
    arms=('B1','oracle','Shared-R16','Shared-R32');metrics=('tIoU','sIoU','vIoU');rows={a:report['arms'][a]['rows'] for a in arms}
    parents=sorted({r['source'] for r in rows['B1']});assert len(parents)==16 and len(rows['B1'])==64
    assert all(sum(r['source']==p for r in rows['B1'])==4 for p in parents)
    errors=[]
    def close(x,y):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors.append(e);assert e<1e-8,(x,y,e)
    mat={};qmat={}
    for arm in arms:
        assert [(r['key'],r['source']) for r in rows[arm]]==[(r['key'],r['source']) for r in rows['B1']]
        qmat[arm]=np.array([[r['metrics'][m] for m in metrics] for r in rows[arm]])
        mat[arm]=np.array([[sum(r['metrics'][m] for r in rows[arm] if r['source']==p)/4 for m in metrics] for p in parents])
        for j,m in enumerate(metrics):
            close(mat[arm][:,j].mean(),report['arms'][arm]['summary']['parent_macro'][m]);close(qmat[arm][:,j].mean(),report['arms'][arm]['summary']['query_macro'][m])
        close(sum(not r['format_ok'] for r in rows[arm]),report['arms'][arm]['summary']['format_failures'])
    draws=np.random.default_rng(20260927).integers(0,16,(10000,16))
    for arm,base,collection in [(a,'B1','comparisons') for a in arms[1:]]+[(a,'oracle','versus_full') for a in arms[2:]]:
        delta=mat[arm]-mat[base]
        for j,m in enumerate(metrics):
            x=delta[:,j];saved=report[collection][arm][m]
            close(x.mean()*100,saved['mean_delta_pp']);close(np.quantile(x[draws].mean(1),[.025,.975])*100,saved['bootstrap_ci95_pp'])
            for p,v in zip(parents,x):close(v*100,saved['parent_delta_pp'][p])
            for name,val in [('positive_parents',(x>1e-10).sum()),('negative_parents',(x< -1e-10).sum()),('zero_parents',(abs(x)<=1e-10).sum()),('severe_loss_below_minus5pp',(x< -.05).sum()),('worst_delta_pp',x.min()*100),('best_delta_pp',x.max()*100)]:close(val,saved[name])
    gates={};cases={}
    for arm in arms[1:]:
        d=mat[arm]-mat['B1'];qd=(qmat[arm]-qmat['B1'])*100
        collapse=any(d[:,j].mean()*100<=-1 and (d[:,j]< -1e-10).sum()>=12 for j in (0,1))
        g=dict(v_positive=bool(d[:,2].mean()>0),t_nonnegative=bool(d[:,0].mean()>=0),s_nonnegative=bool(d[:,1].mean()>=0),no_systematic_branch_collapse=not collapse)
        g['pass']=all(g.values());assert g==report['gates'][arm];gates[arm]=g;cases[arm]={}
        denom=(mat['oracle']-mat['B1'])[:,2].mean();ratio=d[:,2].mean()/denom if denom else None
        if ratio is not None:close(ratio,report['oracle_gain_retention'][arm])
        else:assert report['oracle_gain_retention'][arm] is None
        for j,m in enumerate(metrics):
            good=qmat['B1'][:,j]>.5;kept=good&(qmat[arm][:,j]>.5);ret=report['B1_good_retention'][arm][m]
            close(good.sum(),ret['eligible']);close(kept.sum(),ret['retained']);x=qd[:,j];tail=report['query_tails'][arm][m]
            for name,val in [('query_harm_gt5pp',(x< -5).sum()),('query_gain_gt5pp',(x>5).sum()),('worst_pp',x.min()),('best_pp',x.max()),('positive',(x>0).sum()),('negative',(x<0).sum()),('zero',(x==0).sum())]:close(val,tail[name])
            cases[arm][m]=dict(harm_indices=np.flatnonzero(x< -5).tolist(),lost_good_indices=np.flatnonzero(good&~kept).tolist(),gain_indices=np.flatnonzero(x>5).tolist())
    decision='fixed_rank16_factorized_predictor_candidate' if gates['Shared-R16']['pass'] else ('fixed_rank32_factorized_predictor_candidate' if gates['Shared-R32']['pass'] else 'stop_fixed_low_rank_correction')
    assert decision==report['decision']
    result=dict(status='passed',checks=len(errors),max_abs=max(errors),queries=64,parents=16,
        gates=gates,decision=decision,cases=cases,CPU_seconds=time.monotonic()-start,
        scope='Independent saved-score parent means/CIs/tails/retention/routing; all64, no selection or new labels.')
    out.write_text(json.dumps(result,indent=2)+'\n');print('A03_ROOT_SUMMARY_PASS',len(errors),max(errors),decision)


if __name__=='__main__':main()

"""A0 all-Dev64 seal-first double geometry scoring and independent parent checks."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D,load,labels_for
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins

def score():
    import numpy as np,torch
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
    from vg_tta.external_evidence_metrics import tensor_metrics
    torch.set_num_threads(4);start=time.monotonic();dest=D/'native';out=dest/'independent_readback_v1';assert not out.exists()
    seal=read(dest/'PREDICTIONS_SEAL.json');assert read(dest/'COMPLETE.json')['seal_sha']==sha(dest/'PREDICTIONS_SEAL.json') and seal['predictions']==128
    check_pins({str(dest/p):h for p,h in seal['files'].items()});check_pins(read(dest/'LOCK.json')['pins'])
    assert read(D/'ROOT_DIRECTION_READBACK.json')['decision']=='native_dev64'
    write(out/'PRE_SCORE_AUDIT.json',dict(status='passed',queries=64,parents=16,predictions=128,seal_sha=sha(dest/'PREDICTIONS_SEAL.json'),source_GT_privilege=True,exposed_dev=True,fresh_read=False))
    rows=read(D/'DEV64.json');labels=labels_for(rows,'dev');results={a:[] for a in ['B1','A0']};maxerr=0.
    for i,row in enumerate(rows):
        for arm in results:
            pred=load(dest/'episodes'/f'{i:04}'/(arm+'.pt'));assert pred['key']==row['key'] and pred['source']==row['source']
            m=score_tube_independently(pred,labels[row['key']]);other=tensor_metrics(pred,labels[row['key']])
            err=max(abs(m[k]-other[k]) for k in ['tIoU','sIoU','vIoU']);assert err<1e-6;maxerr=max(maxerr,err)
            results[arm].append(dict(key=row['key'],source=row['source'],metrics=m,interval=pred['interval'],format_ok=pred['format_ok'],invalid_geometry=int((~pred['geometry_valid']).sum())))
    parents={a:summarize_parents(rs) for a,rs in results.items()};metrics=['tIoU','sIoU','vIoU']
    comp={k:paired_parent_bootstrap(parents['A0'],parents['B1'],k) for k in metrics};ret={};tails={};ind={}
    for k in metrics:
        b=np.array([r['metrics'][k] for r in results['B1']]);a=np.array([r['metrics'][k] for r in results['A0']]);d=(a-b)*100
        ret[k]=dict(eligible=int((b>.5).sum()),retained=int(((b>.5)&(a>.5)).sum()))
        tails[k]=dict(query_harm_gt5pp=int((d< -5).sum()),query_gain_gt5pp=int((d>5).sum()),worst_query_pp=float(d.min()))
        ds=[]
        for parent in sorted({r['source'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['source']==parent];assert len(ix)==4
            ds.append(float(d[ix].mean()))
        mean=float(np.mean(ds));assert abs(mean-comp[k]['mean_delta_pp'])<1e-10
        ind[k]=dict(mean_delta_pp=mean,negative_parents=int((np.array(ds)<0).sum()),parent_deltas_pp=ds)
    collapse=any(ind[k]['mean_delta_pp']<=-1 and ind[k]['negative_parents']>=12 for k in ['tIoU','sIoU'])
    decision='trust_or_noop_gate_next' if ind['vIoU']['mean_delta_pp']<=0 else ('branch_collapse_stop' if collapse else 'eligible_later_full_expansion')
    report=dict(status='completed_double_scored',arms={a:dict(rows=rs,summary=summarize_arm(rs,parents[a])) for a,rs in results.items()},comparisons=comp,retention=ret,query_tails=tails,independent_parent_checks=ind,
        scalar_tensor_max_abs=maxerr,systematic_branch_collapse=collapse,decision=decision,scope='Dev64/16 exposed source parents; GT privilege; one seed; descriptive unadjusted CI, not fresh/target efficacy.',CPU_seconds=time.monotonic()-start)
    write(out/'REPORT.json',report);write(out/'COMPLETE.json',dict(report_sha=sha(out/'REPORT.json')));print('A0_NATIVE_SCORED',decision,comp,flush=True)
if __name__=='__main__':score()

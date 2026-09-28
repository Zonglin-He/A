"""Seal-first full-source447 oracle vs both frozen learned corrections, CPU only."""
import json,sqlite3,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_oracle_mixer_gap import D,E,OLD,ROSTER,SEEDS,ARMS,load,ADAPTER_SHA
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins

def score():
    import numpy as np,torch
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
    from vg_tta.external_evidence_metrics import tensor_metrics
    torch.set_num_threads(4);start=time.monotonic();out=D/'independent_readback_v1';assert not out.exists()
    check_pins(read(D/'LOCK.json')['pins']);done=read(D/'COMPLETE.json');seal=read(D/'PREDICTIONS_SEAL.json')
    assert done['seal_sha']==sha(D/'PREDICTIONS_SEAL.json') and seal['predictions']==done['predictions']==1788
    check_pins({str(D/p):h for p,h in seal['files'].items()});raw=read(D/'ROOT_ALL_RAW_READBACK.json')
    assert raw['status']=='passed' and raw['episodes']==447
    rows=read(D/'INPUTS.json');assert len(rows)==447 and len({r['source'] for r in rows})==31
    write(out/'PRE_SCORE_AUDIT.json',dict(status='passed',queries=447,parents=31,predictions=1788,
        seal_sha=sha(D/'PREDICTIONS_SEAL.json'),raw_audit_sha=sha(D/'ROOT_ALL_RAW_READBACK.json'),
        GT_disclosure='All447 are now exposed diagnosis. Source GT used for native oracle gradients and prior privileged evidence, never target; not held-out confirmation.',optimizer_steps=0))
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
    results={a:[] for a in ARMS};error=0.;old=read(E/'independent_readback_v1/REPORT.json');reuse_error=0.
    geometry=[]
    for i,row in enumerate(rows):
        ep=D/'episodes'/f'{i:04}';inp=read(ep/'INPUT.json')
        label=json.loads(db.execute('select labels_json from examples where key=?',(row['key'],)).fetchone()[0])
        for arm in ARMS:
            pred=load(ep/(arm+'.pt'));assert pred['key']==row['key'] and pred['source']==row['source']
            assert pred['support']==inp['support'] and pred['preprocess']==inp['preprocess'] and pred['adapter_sha']==ADAPTER_SHA
            assert pred['GT_read'] and not pred['decoder_GT_prefix'] and not pred['target_read'] and pred['optimizer_steps']==0
            assert pred['injection']['same_field_both_passes'] and pred['injection']['relative_norm']<=read(D/'CONFIG.json')['radius']+2e-6
            metrics=score_tube_independently(pred,label);independent=tensor_metrics(pred,label)
            error=max(error,max(abs(metrics[m]-independent[m]) for m in ['tIoU','sIoU','vIoU']));assert error<1e-6
            if arm!='oracle':
                prior=old['arms'][arm]['rows'][i];assert (prior['key'],prior['source'])==(row['key'],row['source'])
                reuse_error=max(reuse_error,max(abs(metrics[m]-prior['metrics'][m]) for m in ['vIoU','sIoU','tIoU']));assert reuse_error==0
            results[arm].append(dict(key=row['key'],source=row['source'],metrics=metrics,format_ok=pred['format_ok'],interval=pred['interval'],invalid_geometry_frames=int((~pred['geometry_valid']).sum())))
        geometry.append(dict(index=i,key=row['key'],source=row['source'],**read(ep/'GEOMETRY.json')))
    parents={a:summarize_parents(x) for a,x in results.items()}
    comp={a:{m:paired_parent_bootstrap(parents[a],parents['B1'],m) for m in ['vIoU','sIoU','tIoU']} for a in ARMS[1:]}
    oracle_learned={a:{m:paired_parent_bootstrap(parents['oracle'],parents[a],m) for m in ['vIoU','sIoU','tIoU']} for a in SEEDS}
    retention={m:{a:dict(eligible=sum(r['metrics'][m]>.5 for r in results['B1']),retained=sum(b['metrics'][m]>.5 and r['metrics'][m]>.5 for b,r in zip(results['B1'],results[a]))) for a in ARMS[1:]} for m in ['vIoU','tIoU']}
    gate={a:dict(v_positive=comp[a]['vIoU']['mean_delta_pp']>0,v_lower_CI_positive=comp[a]['vIoU']['bootstrap_ci95_pp'][0]>0,
        t_nonnegative=comp[a]['tIoU']['mean_delta_pp']>=0,s_nonnegative=comp[a]['sIoU']['mean_delta_pp']>=0,
        no_severe_parent_v_harm=comp[a]['vIoU']['severe_loss_below_minus5pp']==0,
        native_good_retained=all(retention[m][a]['eligible']==retention[m][a]['retained'] for m in retention)) for a in ARMS[1:]}
    tails={a:{m:dict(loss_below_minus5pp=sum(r['metrics'][m]-b['metrics'][m]<-.05 for b,r in zip(results['B1'],results[a])),gain_above5pp=sum(r['metrics'][m]-b['metrics'][m]>.05 for b,r in zip(results['B1'],results[a]))) for m in ['vIoU','sIoU','tIoU']} for a in ARMS[1:]}
    def desc(values):
        x=np.asarray([v for v in values if v is not None]);return dict(defined=len(x),undefined=len(values)-len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None,positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()))
    def geometry_summary(indices):
        return {a:dict(cosine=desc([geometry[i]['cosine_to_oracle'][a] for i in indices]),
            T_descent=desc([geometry[i]['descent_dot']['event'][a] for i in indices]),S_descent=desc([geometry[i]['descent_dot']['spatial'][a] for i in indices]),
            norm_over_cap=desc([geometry[i]['norm_over_cap'][a] for i in indices]),
            tanh_abs_gt099=desc([geometry[i]['mixer'][a]['fraction_abs_above_099'] for i in indices]) if a!='oracle' else None,
            delta_v_pp=desc([100*(results[a][i]['metrics']['vIoU']-results['B1'][i]['metrics']['vIoU']) for i in indices])) for a in ARMS[1:]}
    groups={'all':list(range(447))}
    for m in ['vIoU','tIoU']:
        groups[m+'_B1_good']=[i for i,x in enumerate(results['B1']) if x['metrics'][m]>.5]
        groups[m+'_B1_other']=[i for i,x in enumerate(results['B1']) if x['metrics'][m]<=.5]
    by_parent={p:geometry_summary([i for i,r in enumerate(rows) if r['source']==p]) for p in sorted({r['source'] for r in rows})}
    report=dict(status='completed_scored_awaiting_root_summary',arms={a:dict(rows=x,summary=summarize_arm(x,parents[a])) for a,x in results.items()},comparisons=comp,
        oracle_vs_learned=oracle_learned,retention=retention,query_tails=tails,gate=gate,oracle_qualified=all(gate['oracle'].values()),
        metric_scalar_tensor_max_abs=error,reused_metric_max_abs=reuse_error,
        geometry_groups={name:dict(queries=len(ids),parents=len({rows[i]['source'] for i in ids}),stats=geometry_summary(ids)) for name,ids in groups.items()},geometry_by_parent=by_parent,
        scope='447 exposed source diagnosis queries/31 parents; fixed GT analytic oracle, not a theoretical upper bound or held-out generalization. Two frozen learned seeds; zero optimizer/target/expert/OPD. Descriptive unadjusted parent CIs.',CPU_seconds=time.monotonic()-start)
    write(out/'GEOMETRY_CASES.json',geometry);write(out/'REPORT.json',report)
    lines=['# Oracle–Mixer Gap Audit','',report['scope'],'','|arm|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
    for a in ARMS:lines.append('|'+a+'|'+'|'.join(f"{100*report['arms'][a]['summary']['parent_macro'][m]:.6f}" for m in ['tIoU','sIoU','vIoU'])+'|')
    lines+=['',f"Oracle practical qualification: {report['oracle_qualified']}",'',json.dumps(comp,indent=2),'',json.dumps(report['geometry_groups']['all'],indent=2)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');write(out/'COMPLETE.json',dict(report_sha=sha(out/'REPORT.json')))
if __name__=='__main__':
    try:score()
    except BaseException as e:
        f=D/'SCORER_FAILURE.json'
        if not f.exists():write(f,dict(error=repr(e),traceback=traceback.format_exc()))
        raise

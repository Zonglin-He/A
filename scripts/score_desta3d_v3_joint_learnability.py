"""Seal-first independent source-privileged qualification, no model/GPU."""
import json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_joint_learnability import D,ROSTER,load,ADAPTER_SHA
from scripts.desta3d_v3_joint_learnability_eval import E,ARMS
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,tensor_sha

def score():
    import numpy as np,torch
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
    from vg_tta.external_evidence_metrics import tensor_metrics
    torch.set_num_threads(4);out=E/'independent_readback_v1';assert not out.exists();start=time.monotonic()
    check_pins(read(D/'LOCK.json')['pins']);check_pins(read(E/'LOCK.json')['pins'])
    done=read(E/'COMPLETE.json');assert done['seal_sha']==sha(E/'PREDICTIONS_SEAL.json')
    seal=read(E/'PREDICTIONS_SEAL.json');check_pins({str(E/p):h for p,h in seal['files'].items()})
    rows=read(D/'VALIDATION_INPUTS.json');assert done['predictions']==seal['predictions']==3*len(rows)
    train_parents={r['source'] for r in read(D/'TRAIN_INPUTS.json')};assert not train_parents&{r['source'] for r in rows}
    states={}
    for arm in ARMS[1:]:
        state=load(D/arm/'FINAL.pt');assert state['cursor']==618 and state['windows']==155
        assert state['lock_sha']==sha(D/'LOCK.json') and all(isinstance(k,int) for k in state['optimizer']['state'])
        assert {int(v['step']) for v in state['optimizer']['state'].values()}=={state['steps']}
        assert torch.equal(state['mixer']['basis'],load(D/'BASIS.pt'))
        hs=[read(p) for p in sorted((D/arm/'history').glob('*.json'))];assert len(hs)==155
        assert sum(len(h['queries']) for h in hs)==618 and hs[-1]['cursor']==618
        states[arm]=dict(steps=state['steps'],empty_windows=state['empty_windows'],windows=155,queries=618,
            clip_windows=sum(h['clip'] for h in hs),missing_support={b:sum('missing' in q['branches'][b] for h in hs for q in h['queries']) for b in ['event','spatial']})
    payload={a:[] for a in ARMS}
    for i,row in enumerate(rows):
        ep=E/'episodes'/f'{i:04}';inp=read(ep/'INPUT.json')
        for a in ARMS:
            p=load(ep/(a+'.pt'));assert p['key']==row['key'] and p['source']==row['source']
            assert p['support']==inp['support'] and p['preprocess']==inp['preprocess'] and p['adapter_sha']==ADAPTER_SHA
            assert p['injection']['same_field_both_passes'] and not p['decoder_GT_prefix'] and not p['target_read']
            assert p['injection']['relative_norm']<=read(D/'CONFIG.json')['radius']+2e-6
            payload[a].append(p)
    write(out/'PRE_SCORE_AUDIT.json',dict(status='passed',queries=len(rows),parents=31,predictions=3*len(rows),states=states,
        GT_disclosure='source GT used for privileged inference; metrics not used for training/selection',target_read=False))
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
    results={a:[] for a in ARMS};error=0.
    for i,row in enumerate(rows):
        label=json.loads(db.execute('select labels_json from examples where key=?',(row['key'],)).fetchone()[0])
        for a in ARMS:
            p=payload[a][i];v=score_tube_independently(p,label);vv=tensor_metrics(p,label)
            error=max(error,max(abs(v[m]-vv[m]) for m in ['vIoU','sIoU','tIoU']));assert error<1e-6
            results[a].append(dict(key=row['key'],source=row['source'],metrics=v,format_ok=p['format_ok'],interval=p['interval']))
    parents={a:summarize_parents(x) for a,x in results.items()}
    comp={a:{m:paired_parent_bootstrap(parents[a],parents['B1'],m) for m in ['vIoU','sIoU','tIoU']} for a in ARMS[1:]}
    retention={m:{a:dict(eligible=sum(r['metrics'][m]>.5 for r in results['B1']),
        retained=sum(b['metrics'][m]>.5 and r['metrics'][m]>.5 for b,r in zip(results['B1'],results[a]))) for a in ARMS[1:]} for m in ['vIoU','tIoU']}
    gate={a:dict(v_positive=comp[a]['vIoU']['mean_delta_pp']>0,v_lower_CI_positive=comp[a]['vIoU']['bootstrap_ci95_pp'][0]>0,
        t_nonnegative=comp[a]['tIoU']['mean_delta_pp']>=0,s_nonnegative=comp[a]['sIoU']['mean_delta_pp']>=0,
        no_severe_parent_v_harm=comp[a]['vIoU']['severe_loss_below_minus5pp']==0,
        native_good_retained=all(retention[m][a]['eligible']==retention[m][a]['retained'] for m in retention)) for a in ARMS[1:]}
    # Separate implementation independently aggregates the scalar per-query
    # values, then resamples parent pairs with the fixed seed.
    errors=[]
    for a in ARMS[1:]:
        for metric in ['vIoU','sIoU','tIoU']:
            by={s:[] for s in sorted({r['source'] for r in rows})}
            for b,r in zip(results['B1'],results[a]):by[r['source']].append(100*(r['metrics'][metric]-b['metrics'][metric]))
            delta=np.array([np.mean(by[s]) for s in by]);mean=float(delta.mean())
            # Existing scorer fixes both order and seed; inspect its API result
            # and verify point mean/tails independently without inventing CIs.
            errors.append(abs(mean-comp[a][metric]['mean_delta_pp']))
            assert errors[-1]<1e-8 and int((delta < -5).sum())==comp[a][metric]['severe_loss_below_minus5pp']
            draws=np.random.default_rng(20260927).integers(0,len(delta),size=(10000,len(delta)))
            ci=np.quantile(np.mean(delta[draws],axis=1),[.025,.975])
            errors.append(float(np.max(np.abs(ci-np.asarray(comp[a][metric]['bootstrap_ci95_pp'])))))
            assert errors[-1]<1e-8
    report=dict(status='source_privileged_learnability_scored',arms={a:dict(rows=x,summary=summarize_arm(x,parents[a])) for a,x in results.items()},
        comparisons=comp,retention=retention,gate=gate,qualified=all(all(g.values()) for g in gate.values()),
        metric_scalar_tensor_max_abs=error,parent_mean_crosscheck_max_abs=max(errors),training=states,
        scope='Two seeds;31 newly locked source parents relative to enumerated development manifests; GT privileged evidence; no target/OPD/generalization-to-experts claim; descriptive unadjusted paired-parent CIs',CPU_seconds=time.monotonic()-start)
    write(out/'REPORT.json',report)
    lines=['# Joint mixer source GT-privileged learnability','',report['scope'],'','|arm|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
    for a in ARMS:lines.append('|'+a+'|'+'|'.join(f"{100*report['arms'][a]['summary']['parent_macro'][m]:.6f}" for m in ['tIoU','sIoU','vIoU'])+'|')
    lines+=['',f"Qualified by every registered condition: {report['qualified']}",'',str(comp),'',str(gate)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');write(out/'COMPLETE.json',dict(report_sha=sha(out/'REPORT.json')))

if __name__=='__main__':score()

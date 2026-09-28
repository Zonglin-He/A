"""Offline scoring; all target labels stay outside legal adaptation construction."""
import argparse,collections,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_spatial_three_rounds_v1 import OUT,verify,ARMS
from vg_tta.spatial_three_rounds_v1 import summary,continuation
COHORTS=['hcstvg1_test','vidstg_test'];METRICS=['sIoU','vIoU_corrected','tIoU']

def cell(d):
    if d['mean'] is None:return 'NA'
    a,b=d['ci95'];return f"{100*d['mean']:+.3f} [{100*a:+.3f},{100*b:+.3f}]"

def score_p0():
    import numpy as np,torch
    from scripts.analyze_spatial10_components_v1 import checked_score
    p=verify();assert read(OUT/'p0_STATUS.json')['status']=='complete'
    assert sha(p['labels'])==p['labels_sha256'];labels=read(p['labels']);rows=[]
    audit=dict(metric_checks=0,uniform_previous_exact_states=0,full_reinsertions=0,legal_barriers=0,all64=True)
    for row in p['rows']:
        f=OUT/'p0'/f'{row["key"].replace(":","_")}.pt';receipt=read(f.with_suffix('.json'))
        assert sha(f)==receipt['sha256'] and receipt['lock_sha256']==sha(OUT/'LOCK.json')
        x=load(f);g=labels[x['key']];parent=load(row['path']);valid=np.asarray(g['valid'],bool)
        assert all(x['audit'][k] for k in ['native_exact','lr0_exact','steps0_exact','reset_exact','temporal_logits_invariant','teacher_frozen'])
        assert x['audit']['parameters']==1792 and not x['audit']['legal_weights_GT']
        assert torch.equal(x['weights']['D_soft'].sort().values,x['weights']['E_shifted'].sort().values)
        def score(pred):
            m,q=checked_score(pred['boxes'],g,x['frame_ids'],pred['indices']);audit['metric_checks']+=5
            return m,q
        base,bq=score(x['baseline']);native,_=score(parent['predictions']['Frozen']);full,_=score(parent['predictions']['Full_DeCoTA'])
        scores={};unsupervised={};changes={}
        for name,arm in x['arms'].items():
            assert arm['prediction']['indices']==x['baseline']['indices']
            assert all(torch.equal(a,b) for a,b in zip(arm['prediction']['logits'],x['baseline']['logits']))
            m,q=score(arm['prediction']);assert m['tIoU']==base['tIoU'];scores[name]=m
            weights=x['weights'][name.split('_step')[0]].numpy();mask=valid&(weights==0)
            unsupervised[name]=dict(n_frames=int(mask.sum()),delta_s=float((q[mask]-bq[mask]).mean()) if mask.any() else None)
            delta=torch.cat([(arm['state'][k]-v).flatten().double() for k,v in x['initial'].items()])
            assert abs(float(delta.norm())-arm['delta_norm'])<1e-9
            assert int((delta!=0).sum())==arm['changed_parameters']
            changes[name]=dict(delta_norm=arm['delta_norm'],changed=arm['changed_parameters'])
        tr=x['teacher_receipt'];assert sha(tr['path'])==tr['sha256'];teacher=load(tr['path']);assert not teacher['GT_access']
        ts,_=score({**x['baseline'],'boxes':teacher['target']});audit['legal_barriers']+=1
        audit['uniform_previous_exact_states']+=x['audit']['uniform_prior_parity'];audit['full_reinsertions']+=x['audit']['full_reinsertions']
        rows.append(dict(key=x['key'],source=x['source'],cohort=x['cohort'],caption=x['input']['caption'],
          baseline=base,native=native,full_method=full,metrics=scores,teacher=ts,unsupervised=unsupervised,
          changes=changes,confidence=x['confidence'],paths=x['paths'],origin_losses=x['origin_losses'],
          empty_F=x['audit']['empty_F'],seconds=x['seconds'],peak_memory_bytes=x['peak_memory_bytes']))
    stats={}
    for c in COHORTS:
        rr=[r for r in rows if r['cohort']==c];assert len(rr)==len({r['source'] for r in rr})==32
        d=dict(n=32,context={},arms={},contrasts={})
        for label in ['baseline','native','full_method','teacher']:
            d['context'][label]={m:summary([r[label][m] for r in rr]) for m in METRICS}
        for name in rr[0]['metrics']:
            arm={m:dict(absolute=summary([r['metrics'][name][m] for r in rr]),
                delta=summary([r['metrics'][name][m]-r['baseline'][m] for r in rr])) for m in METRICS}
            arm['unsupervised_s']=summary([r['unsupervised'][name]['delta_s'] for r in rr])
            arm['parameter_changed']=sum(r['changes'][name]['changed']>0 for r in rr)
            root,step=name.split('_step');arm['loss_improved']=sum(r['paths'][root][int(step)-1]['loss']<r['origin_losses'][root] for r in rr)
            d['arms'][name]=arm
        pairs=[('F_GT','A_uniform'),('C_adapted','A_uniform'),('C_adapted','B_native'),
               ('D_soft','A_uniform'),('D_soft','E_shifted'),('B_native','A_uniform')]
        for a,b in pairs:
            for step in [1,5]:
                aa=f'{a}_step{step}';bb=f'{b}_step{step}'
                d['contrasts'][aa+'-'+bb]={m:summary([r['metrics'][aa][m]-r['metrics'][bb][m] for r in rr]) for m in METRICS}
        stats[c]=d
    decision=continuation(stats)
    def pooled(name,m,step):return sum(stats[c]['arms'][f'{name}_step{step}'][m]['delta']['mean'] for c in COHORTS)/2
    legal=['A_uniform','C_adapted','D_soft']
    best=max(legal,key=lambda a:(pooled(a,'sIoU',5),pooled(a,'vIoU_corrected',5)))
    useful=max(pooled(best,'sIoU',1),pooled(best,'sIoU',5))>.001
    decision.update(P2_objective=best if useful else None,P2_run=useful,
      P2_eligible_pooled={a:{str(s):{m:pooled(a,m,s) for m in METRICS} for s in [1,5]} for a in legal},
      P1_must_finish_before_P2_if_enabled=decision['P1_run'],created=time.time())
    write(OUT/'P0_SOURCE_RESULTS.json',rows);write(OUT/'P0_SUMMARY.json',stats);write(OUT/'P0_DECISION.json',decision)
    assert audit['uniform_previous_exact_states']==32 and audit['full_reinsertions']==832
    audit.update(status='passed',n_queries=64,n_sources=64,lock_sha256=sha(OUT/'LOCK.json'),completed=time.time())
    write(OUT/'P0_AUDIT.json',audit)
    report=['# P0: temporal-weighted ROI spatial adaptation','',
      '64 historical development sources,32+32; not untouched confirmation. TA-STVG,1792 spatial parameters.',
      'ROI teacher is actual2x native-prediction crop/re-localization; no external expert. Raw1/5 AdamW steps, LR .005/.05, eps1e-4.',
      'All arms use the same saved locked temporal5step eta=.25 parameters, reinserted into the original model. Primary deltas subtract frozen boxes plus that same adapted time.',
      'Uniform exactly replays16 old queries at both endpoints. No new temporal update or production method modification.','']
    for c,d in stats.items():
        report+=['## '+c,'','| context | vIoU % | sIoU % | tIoU % |','|---|---:|---:|---:|']
        for name in ['native','baseline','full_method']:
            report.append('| '+name+' | '+' | '.join(f"{100*d['context'][name][m]['mean']:.3f}" for m in ['vIoU_corrected','sIoU','tIoU'])+' |')
        report+=['','| weighting | step | Δv pp [95%CI] | Δs pp [95%CI] | median Δs | negative s | Q10 Δs | harm>5pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in ARMS:
            for step in [1,5]:
                m=d['arms'][f'{name}_step{step}'];s=m['sIoU']['delta']
                report.append(f"| {name} | {step} | {cell(m['vIoU_corrected']['delta'])} | {cell(s)} | {100*s['median']:+.3f} | {s['negative_rate']:.1%} | {100*s['q10']:+.3f} | {s['harm_gt5pp']}/32 |")
        report+=['','| paired contrast | Δs pp [95%CI] | Δv pp [95%CI] |','|---|---:|---:|']
        for name,m in d['contrasts'].items():report.append(f"| {name} | {cell(m['sIoU'])} | {cell(m['vIoU_corrected'])} |")
    report+=['','## Prelocked continuation decision','',json.dumps(decision,ensure_ascii=False,indent=2),'',
      'Nominal source-paired bootstrap10,000, seed20260917; exploratory within exposed development. F uses GT time only as a diagnostic, never an eligible method.',
      'Dense ROI forward observes every frame. Unobserved gain is NA; zero-weight GT-frame results for hard masks are UNSUPERVISED, not unobserved, and are in JSON.',
      'Temporal influence can alter which supervision enters this spatial suffix; native temporal parameters/logits themselves stay unchanged across arms.']
    (OUT/'P0_RESULTS.md').write_text('\n'.join(report)+'\n');print(json.dumps(decision,indent=2))

def correlations(a,b):
    import numpy as np
    from scipy.stats import rankdata
    a=np.asarray(a,float);b=np.asarray(b,float)
    def corr(x,y):
        x=rankdata(x,axis=-1);y=rankdata(y,axis=-1)
        x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
        den=np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
        return np.divide((x*y).sum(-1),den,out=np.full_like(den,np.nan),where=den>0)
    rho=float(corr(a,b));rng=np.random.default_rng(20260917);idx=rng.integers(len(a),size=(10000,len(a)))
    bootstrap=corr(a[idx],b[idx]);bootstrap=bootstrap[np.isfinite(bootstrap)]
    return dict(rho=rho if np.isfinite(rho) else None,ci95=np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap) else None,
                n=len(a),valid_bootstrap=len(bootstrap),ties='average rank')

def score_p2():
    import numpy as np,torch
    from scripts.analyze_spatial10_components_v1 import checked_score
    p=verify();assert read(OUT/'p2_STATUS.json')['status']=='complete';labels=read(p['labels']);rows=[]
    old={r['key']:r for r in read(OUT/'P0_SOURCE_RESULTS.json')};audit=dict(full_reinsertions=0,metric_checks=0)
    for row in p['rows']:
        f=OUT/'p2'/f'{row["key"].replace(":","_")}.pt';r=read(f.with_suffix('.json'))
        assert sha(f)==r['sha256'] and r['lock_sha256']==sha(OUT/'LOCK.json')
        x=load(f);parent=load(row['path']);gt=labels[x['key']];scores={}
        for key,arm in x['arms'].items():
            pred=arm['prediction'];assert pred['indices']==x['baseline']['indices']
            m,_=checked_score(pred['boxes'],gt,parent['frame_ids'],pred['indices']);scores[key]=m;audit['metric_checks']+=5
        assert abs(scores['1']['sIoU']-old[x['key']]['metrics'][x['objective']+'_step5']['sIoU'])<1e-12
        assert abs(scores['0']['sIoU']-old[x['key']]['baseline']['sIoU'])<1e-12
        optimal=max(scores,key=lambda k:(scores[k]['sIoU'],-float(k)))
        strictinterior=(0<float(optimal)<1 and scores[optimal]['sIoU']>max(scores['0']['sIoU'],scores['1']['sIoU'])+.001)
        harmed=scores['1']['sIoU']<scores['0']['sIoU']
        rows.append(dict(key=x['key'],cohort=x['cohort'],source=x['source'],objective=x['objective'],metrics=scores,
                 oracle_alpha=float(optimal),strict_interior=strictinterior,harmed_endpoint=harmed,
                 confidence=x['confidence']['strength'],seconds=x['seconds']))
        audit['full_reinsertions']+=x['audit']['full_reinsertions'];assert all(x['audit'].values())
    stats={}
    for c in COHORTS:
        rr=[r for r in rows if r['cohort']==c];assert len(rr)==32
        d=dict(n=32,alphas={},oracle_distribution=dict(collections.Counter(str(r['oracle_alpha']) for r in rr)),
               harmed=sum(r['harmed_endpoint'] for r in rr),harmed_with_strict_interior=sum(r['harmed_endpoint'] and r['strict_interior'] for r in rr))
        for a in p['alphas']:
            k=str(a);d['alphas'][k]={m:summary([r['metrics'][k][m]-r['metrics']['0'][m] for r in rr]) for m in METRICS}
        d['confidence_vs_oracle_alpha']=correlations([r['confidence'] for r in rr],[r['oracle_alpha'] for r in rr])
        d['confidence_vs_gain']=correlations([r['confidence'] for r in rr],[r['metrics']['1']['sIoU']-r['metrics']['0']['sIoU'] for r in rr])
        stats[c]=d
    harmed=sum(d['harmed'] for d in stats.values());strict=sum(d['harmed_with_strict_interior'] for d in stats.values())
    decision=dict(heldout_selector_run=harmed>0 and strict/harmed>.5,harmed= harmed,
                  harmed_with_strict_interior=strict,ratio=strict/harmed if harmed else None,
                  criterion='strict interior s gain >0.1pp over both endpoints in more than half of harmed cases',
                  confidence_gate_supported=all(d[k]['rho'] is not None and d[k]['rho']>.4 for d in stats.values()
                      for k in ['confidence_vs_oracle_alpha','confidence_vs_gain']),
                  no_per_sample_GT_rule_deployed=True,created=time.time())
    write(OUT/'P2_SOURCE_RESULTS.json',rows);write(OUT/'P2_SUMMARY.json',stats);write(OUT/'P2_DECISION.json',decision)
    audit.update(status='passed',no_new_backward=True,queries=64,completed=time.time());write(OUT/'P2_AUDIT.json',audit)
    report=['# P2: real parameter-path shrinkage; P3 confidence diagnostic','',
       'Same64 development sources; objective chosen on P0, not new held-out confirmation. Saved step5 displacement; no backward. All temporal parameters and readout fixed.',
       'Oracle alpha uses GT sIoU only for diagnosis. No GT alpha is deployed or called a legal method.','']
    for c,d in stats.items():
        report+=['## '+c,'','| alpha | Δs pp [95%CI] | Δv pp [95%CI] | s harm>5pp |','|---|---:|---:|---:|']
        for a,m in d['alphas'].items():report.append(f"| {a} | {cell(m['sIoU'])} | {cell(m['vIoU_corrected'])} | {m['sIoU']['harm_gt5pp']}/32 |")
        report+=['',f"Oracle alpha counts: {d['oracle_distribution']}",
                 f"Harmed alpha1 sources: {d['harmed']}; strict helpful interior: {d['harmed_with_strict_interior']}.",'',
                 'Confidence correlation: '+json.dumps({k:d[k] for k in ['confidence_vs_oracle_alpha','confidence_vs_gain']})]
    report+=['','## Bounded stopping decision','',json.dumps(decision,indent=2),'',
        'A smaller loss than alpha1 is not proof of a useful adaptation direction. Strict positive gain over alpha0 and alpha1 is required by this diagnostic. Correlations are descriptive, not causal.']
    (OUT/'P2_RESULTS.md').write_text('\n'.join(report)+'\n');print(json.dumps(decision,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['p0','p2']);a=ap.parse_args()
    score_p0() if a.stage=='p0' else score_p2()

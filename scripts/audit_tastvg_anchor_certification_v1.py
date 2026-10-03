"""Independent anonymous audit: sklearn isotonic, fresh bootstrap and metrics."""
import sys,json,hashlib,collections,time
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_anchor_certification/2026-10-03'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
EPS=1e-12;SEED=20261003;B=10000;checks=0;maximum=0.
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(x,y):
    global checks,maximum
    if x is None or y is None:assert x is y;checks+=1;return
    a=np.asarray(x,float);b=np.asarray(y,float);assert a.shape==b.shape
    e=float(np.max(np.abs(a-b))) if a.size else 0.;assert e<3e-12,(e,x,y)
    maximum=max(maximum,e);checks+=a.size
def winner(z,a):
    m=max(z);ids=[i for i,v in enumerate(z) if m-v<=EPS]
    return ids[0] if len(ids)==1 and m>z[a]+EPS else a
def stat(rows,fields,ratios):
    if not rows:return dict(metrics={},ratios={},sources=0,cells=0)
    groups=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    for r in rows:groups[r['source_id']][r['order']][r['condition']].append([r[f] for f in fields])
    src=sorted(groups);m=[];orders=collections.defaultdict(list)
    for s in src:
        os=[]
        for o in sorted(groups[s]):
            v=np.mean([np.mean(groups[s][o][c],axis=0) for c in sorted(groups[s][o])],axis=0);os.append(v);orders[o].append(v)
        m.append(np.mean(os,axis=0))
    m=np.array(m);rng=np.random.default_rng(SEED);idx=rng.integers(0,len(m),(B,len(m)))
    bs=m[idx].mean(1);q=np.percentile(bs,[2.5,97.5],axis=0);res={}
    for j,f in enumerate(fields):
        loo=np.array([np.mean(np.delete(m[:,j],i)) for i in range(len(m))]) if len(m)>1 else m[:,j]
        res[f]=dict(mean=float(m[:,j].mean()),ci95=q[:,j].tolist(),source_values={str(s):float(v) for s,v in zip(src,m[:,j])},
            order_values={o:float(np.array(v)[:,j].mean()) for o,v in orders.items()},leave_one_out_range=[float(loo.min()),float(loo.max())])
    rr={};at={f:j for j,f in enumerate(fields)}
    for name,(num,den) in ratios.items():
        d=bs[:,at[den]];valid=d>EPS;point=m[:,at[den]].mean()
        rr[name]=dict(mean=float(m[:,at[num]].mean()/point) if point>EPS else None,
            ci95=np.percentile(bs[valid,at[num]]/d[valid],[2.5,97.5]).tolist() if valid.any() else None,
            valid_draws=int(valid.sum()),bootstrap_zero_denominator_draws=int((~valid).sum()))
    return dict(metrics=res,ratios=rr,sources=len(m),cells=len(rows))
def validate_stat(saved,rows,fields,ratios=None):
    z=stat(rows,fields,ratios or {});assert z['sources']==saved['sources'] and z['cells']==saved['cells']
    for f,v in z['metrics'].items():
        for k in ['mean','ci95','leave_one_out_range']:eq(v[k],saved['metrics'][f][k])
        for k in ['source_values','order_values']:
            assert set(v[k])==set(saved['metrics'][f][k])
            for a,b in zip(v[k].values(),saved['metrics'][f][k].values()):eq(a,b)
    for f,v in z['ratios'].items():
        for k in v:eq(v[k],saved['ratios'][f][k])

def audit():
    start=time.time();cfg=read(OUT/'CONFIG.json');src={(r['dataset'],r['source_id']):r for r in read(OLD/'SOURCE_ROWS.json')}
    cal=read(OUT/'CALIBRATION_SEAL.json');seal=read(OUT/'GLOBAL_DECISION_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    assert seal['time']>cal['time'] and join['time']>seal['time'] and not seal['target_GT_read']
    assert sha(OUT/'DECISIONS.json')==seal['decisions_sha256'] and sha(OUT/'CALIBRATION_SEAL.json')==seal['calibration_seal_sha256']
    assert cfg['source_anchor']=='frozen native candidate0' and cfg['target_anchor']=='unchanged A8'
    assert cfg['user_explicit_native_to_A8_CPU_authorization'] and cfg['source_validation_reused_after_ridge_alpha_selection']
    for path,h in join['label_files'].items():assert sha(ROOT/path)==h
    if (ROOT/'artifacts/tastvg_anchor_certification_v1/RUNTIME_LOCK.json').exists():
        lock=read(ROOT/'artifacts/tastvg_anchor_certification_v1/RUNTIME_LOCK.json')
        for path,h in {**lock['code'],**lock['inputs']}.items():assert sha(ROOT/path)==h
    if (OUT/'RUNTIME_BINDING.json').exists():
        lock=read(OUT/'RUNTIME_BINDING.json')
        for path,h in {**lock['code'],**lock['public_required_inputs']}.items():assert sha(ROOT/path)==h
        assert sha(OUT/'CONFIG.json')==lock['config_sha256']
    prior_scores={r['cell_key']:r for r in read(OLD/'SCORE_ROWS.json')};decisions=read(OUT/'DECISIONS.json');ev={r['cell_key']:r for r in decisions}
    assert len(ev)==288
    for ds in ['vidstg','hc2']:
        rows=read(OUT/ds/'SOURCE_CALIBRATION_ROWS.json');model=read(OUT/ds/'CALIBRATION.json')
        assert len(rows)=={'vidstg':31,'hc2':16}[ds] and sha(OUT/ds/'CALIBRATION.json')==cal['models'][ds]
        assert sha(OUT/ds/'SOURCE_CALIBRATION_ROWS.json')==cal['rows'][ds]
        for r in rows:
            old=src[(ds,r['source_id'])];assert old['split']=='validation' and r['native_index']==old['native_index']==0
            eq(r['scores'],old['frozen_scores']['L']);eq(r['candidate_t'],old['candidate_t'])
            k=winner(r['scores'],0);assert k==r['top_index'] and r['eligible']==(k!=0)
            eq(r['margin'],r['scores'][k]-r['scores'][0]);eq(r['true_delta_t'],r['candidate_t'][k]-r['candidate_t'][0])
        eligible=[r for r in rows if r['eligible']];assert model['eligible_sources']==len(eligible)
        if eligible:
            x=np.array([r['margin'] for r in eligible]);y=np.array([r['true_delta_t'] for r in eligible]);knots=np.unique(x)
            eq(model['knots'],knots);s=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,y);mu=s.predict(knots)
            eq(model['mean'],mu);s.fit(x,(y>EPS).astype(float));eq(model['probability'],s.predict(knots))
            rng=np.random.default_rng(SEED);by=np.empty((B,len(knots)));bp=np.empty_like(by)
            for i in range(B):
                w=np.bincount(rng.integers(0,len(x),len(x)),minlength=len(x));s.fit(x,y,sample_weight=w);by[i]=s.predict(knots)
                s.fit(x,(y>EPS).astype(float),sample_weight=w);bp[i]=s.predict(knots)
            for q in cfg['curve_lower_quantiles']:eq(model['lower'][str(q)],np.minimum(np.quantile(by,q,axis=0),mu))
            eq(model['probability_ci95'],np.quantile(bp,[.025,.975],axis=0));eq(model['thresholds'],np.unique(np.quantile(x,np.linspace(0,1,21))))
        for r in [z for z in decisions if z['dataset']==ds]:
            p=prior_scores[r['cell_key']];a=p['anchor_index'];scores=p['scores']['L'];k=winner(scores,a);m=scores[k]-scores[a]
            eq(r['scores'],scores);eq(r['margin'],m);assert r['top_index']==k
            assert r['candidate_indices']==p['candidate_indices'] and r['intervals']==p['intervals']
            for f in ['pixel_sha256','A_state_pre_sha256','A_state_post_sha256','probe_sha256']:assert r[f]==p[f]
            inside=model['available'] and model['knots'][0]<=m<=model['knots'][-1]
            select=lambda q:k if k!=a and inside and np.interp(m,model['knots'],model['lower'][str(q)])>EPS else a
            assert r['choices']['A']==a and r['choices']['L32']==p['choices']['L32']==k and r['choices']['Selective']==select(.05)
            for q in cfg['curve_lower_quantiles']:assert r['choices']['LCB_'+str(q)]==select(q)
            for j,t in enumerate(model['thresholds']):assert r['choices']['Margin_'+str(j)]==(k if k!=a and m>=t else a)
            if inside:
                for f,v in [('mean',model['mean']),('probability',model['probability']),('lower',model['lower']['0.05'])]:eq(r['calibration'][f],np.interp(m,model['knots'],v))
            else:assert r['calibration']['lower'] is None
            ds0,de=np.array(p['intervals'][k])-p['intervals'][a];rad=(abs(ds0)+abs(de))/(p['intervals'][a][1]-p['intervals'][a][0])
            eq(r['top_geometry']['radius'],rad);assert r['top_geometry']['large']==(rad>=.5-EPS)
    fields=['v','t','dv','dt','gross_gain','gross_loss','accepted','physical_replacement','benefit_t','accepted_dt','severe','accepted_severe','benefit_v','harm_v','neutral_v']
    ratios=dict(beneficial_precision=('benefit_t','accepted'),accepted_mean_delta_t=('accepted_dt','accepted'),accepted_severe_rate=('accepted_severe','accepted'))
    total=0;expert=0
    for ds in ['vidstg','hc2']:
        for split in ['search','confirm']:
            rows=read(OUT/split/ds/'ROWS.json');old={str(r['arrival'])+'/'+r['condition']+'/'+r['order']:r for r in read(OLD/split/ds/'ROWS.json')}
            for r in rows:
                p=old[str(r['arrival'])+'/'+r['condition']+'/'+r['order']];total+=1;expert+=int(r['expert_scheduled'])
                assert r['expert_scheduled']==p['expert_scheduled'];eq(r['A8_v'],p['A8_v']);eq(r['A8_t'],p['A8_t'])
                if r['expert_scheduled']:
                    d=ev['/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])]
                    eq(r['candidate_v'],p['candidate_v']);eq(r['candidate_t'],p['candidate_t']);assert r['choices']==d['choices']
                for arm,z in r['arms'].items():
                    a=p.get('anchor_index');i=d['choices'][arm] if r['expert_scheduled'] else None
                    v=p['candidate_v'][i] if i is not None else p['A8_v'];t=p['candidate_t'][i] if i is not None else p['A8_t']
                    accepted=i is not None and i!=a;dv=v-p['A8_v'];dt=t-p['A8_t']
                    physical=accepted and np.max(np.abs(np.array(d['intervals'][i])-d['intervals'][a]))>EPS
                    expected=[v,t,dv,dt,max(dv,0),max(-dv,0),accepted,physical,accepted and dt>EPS,dt if accepted else 0,
                        dv<-.05,accepted and dv<-.05,dv>EPS,dv< -EPS,abs(dv)<=EPS]
                    eq([z[f] for f in fields],expected)
                eq(r['arms']['L32']['v'],p['L32_v']);eq(r['arms']['L32']['t'],p['L32_t'])
            summary=read(OUT/split/ds/'SUMMARY.json');curves=read(OUT/split/ds/'CURVES.json')
            for cat in ['corruption','clean']:
                rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean')]
                for sub in ['all','expert','nonexpert']:
                    ss=[r for r in rr if sub=='all' or r['expert_scheduled']==(sub=='expert')]
                    for arm in ['A','L32','Selective']:
                        packs=[dict(source_id=r['source_id'],condition=r['condition'],order=r['order'],**r['arms'][arm]) for r in ss]
                        validate_stat(summary[cat][sub][arm],packs,fields,ratios)
                pp=[]
                for r in rr:
                    a=r['arms'];pp.append(dict(source_id=r['source_id'],condition=r['condition'],order=r['order'],Selective_vs_L32_v=a['Selective']['v']-a['L32']['v'],Selective_vs_L32_t=a['Selective']['t']-a['L32']['t'],Selective_vs_L32_gross_loss=a['Selective']['gross_loss']-a['L32']['gross_loss']))
                validate_stat(summary[cat]['paired_vs_L32'],pp,['Selective_vs_L32_v','Selective_vs_L32_t','Selective_vs_L32_gross_loss'])
                for group in ['all','small','large']:
                    ss=[r for r in rr if r['expert_scheduled'] and (group=='all' or r['top_geometry']['large']==(group=='large'))]
                    for arm,saved in curves[cat][group].items():
                        packs=[dict(source_id=r['source_id'],condition=r['condition'],order=r['order'],**r['arms'][arm]) for r in ss]
                        validate_stat(saved,packs,fields,ratios)
                        assert saved['accepting_sources']==len({r['source_id'] for r in ss if r['arms'][arm]['accepted']})
                        for f,v in saved['raw_counts'].items():eq(v,sum(r['arms'][arm][f] for r in ss))
    assert total==1152 and expert==288
    decision=read(OUT/'DECISION.json');all_pass=True
    for ds in ['vidstg','hc2']:
        rows=[r for r in read(OUT/'confirm'/ds/'ROWS.json') if r['condition']!='clean']
        summary=read(OUT/'confirm'/ds/'SUMMARY.json')['corruption']['all']
        sel=summary['Selective']['metrics'];old=summary['L32']['metrics']
        accepting=len({r['source_id'] for r in rows if r['arms']['Selective']['accepted']})
        tests=dict(positive_mean=sel['dv']['mean']>EPS,accepting_sources=accepting>=3,
            positive_leave_one_out=sel['dv']['leave_one_out_range'][0]>EPS,
            less_gross_loss=sel['gross_loss']['mean']<old['gross_loss']['mean']-EPS,
            fewer_severe_harms=sum(r['arms']['Selective']['severe'] for r in rows)<sum(r['arms']['L32']['severe'] for r in rows))
        assert decision['datasets'][ds]['tests']==tests
        assert decision['datasets'][ds]['pass_all']==all(tests.values())
        assert decision['datasets'][ds]['accepting_sources']==accepting
        eq(decision['datasets'][ds]['mean_v_delta'],sel['dv']['mean'])
        all_pass=all_pass and all(tests.values())
    assert decision['status']==('GO_development_only' if all_pass else 'NO_GO_locked_calibration')
    assert not decision['method_promoted'] and not decision['followup_started']
    if (OUT/'DESCRIPTIVE_CHECKS.json').exists():
        for key,saved in read(OUT/'DESCRIPTIVE_CHECKS.json').items():
            ds,arm=key.split('/');rr=[]
            for r in read(OUT/'confirm'/ds/'ROWS.json'):
                if r['condition']!='clean':rr.append(dict(source_id=r['source_id'],condition=r['condition'],order=r['order'],**r['arms'][arm]))
            validate_stat(saved,rr,['dv','dt','gross_gain','gross_loss'])
    if (OUT/'FIGURE_DATA.json').exists():
        f=read(OUT/'FIGURE_DATA.json')
        for ds in ['vidstg','hc2']:
            assert f['curves'][ds]==read(OUT/'confirm'/ds/'CURVES.json')['corruption']
            assert f['confirmation'][ds]==read(OUT/'confirm'/ds/'SUMMARY.json')['corruption']
            assert f['calibration'][ds]==read(OUT/ds/'CALIBRATION.json')
    result=dict(status='passed',scalar_checks=int(checks),maximum_numeric_error=maximum,
        source_validation_only=True,independent_sklearn_isotonic_and_10000_source_bootstrap=True,
        all_frozen_target_rankings_and_choices=True,L32_bitwise_metric_parity=True,all_curve_aggregations_and_paired_intervals=True,
        labels_joined_after_new_global_seal=True,prelocked_GO_tests_recomputed=True,figure_and_descriptive_readback=True,CPU_wall_seconds=time.time()-start,
        scope='cached scalar/calibration/decision audit; no repeated inference or target-domain safety certification')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':audit()

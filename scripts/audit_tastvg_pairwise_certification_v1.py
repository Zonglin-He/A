"""Independent pair reconstruction, SciPy PAV bootstraps, decisions and metrics."""
import sys, time, json, hashlib, collections
from pathlib import Path
import numpy as np
from scipy.optimize import isotonic_regression
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_tastvg_anchor_certification_v1 as stats
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_pairwise_certification/2026-10-03'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
EPS=1e-12;B=10000;SEED=20261003;ARMS=['A','L32','Pair-Raw','Pair-Norm']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):stats.eq(a,b)
def fit(x,y,w):
    keep=w>0;x=x[keep];y=y[keep];w=w[keep]
    k,inv=np.unique(x,return_inverse=True)
    mass=np.bincount(inv,weights=w);values=np.bincount(inv,weights=w*y)/mass
    return k,isotonic_regression(values,weights=mass,increasing=True).x

def audit():
    start=time.time();cfg=read(OUT/'CONFIG.json');cal=read(OUT/'CALIBRATION_SEAL.json')
    seal=read(OUT/'GLOBAL_DECISION_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    assert cal['time']<seal['time']<join['time'] and not cal['target_GT_read'] and not seal['target_GT_read']
    assert not cal['target_score_rows_opened'] and sha(OUT/'DECISIONS.json')==seal['decisions_sha256']
    assert sha(OUT/'CALIBRATION_SEAL.json')==seal['calibration_seal_sha256']
    assert cfg['fixed_lower_quantile']==.05 and cfg['primary']=='Pair-Norm' and cfg['IQR_epsilon']==1e-8
    assert not cfg['threshold_family_searched'] and cfg['source_validation_reused_after_ridge_alpha_and_calibration']
    bind=read(OUT/'RUNTIME_BINDING.json')
    for p,h in {**bind['code'],**bind['public_required_inputs']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==bind['config_sha256']
    for p,h in join['label_files'].items():assert sha(ROOT/p)==h
    private=ROOT/'artifacts/tastvg_pairwise_certification_v1/RUNTIME_LOCK.json'
    if private.exists():
        lock=read(private)
        for p,h in {**lock['code'],**lock['inputs']}.items():assert sha(ROOT/p)==h,p
    src=[r for r in read(OLD/'SOURCE_ROWS.json') if r['split']=='validation'];models={};source_counts={}
    for ds in ['vidstg','hc2']:
        rr=[r for r in src if r['dataset']==ds];assert len(rr)=={'vidstg':31,'hc2':16}[ds]
        saved=read(OUT/ds/'SOURCE_CANDIDATES.json');pr=read(OUT/ds/'SOURCE_PAIRS.json');metadata=read(OUT/ds/'PAIR_METADATA.json')
        assert sha(OUT/ds/'SOURCE_CANDIDATES.json')==cal['datasets'][ds]['source_candidates']
        assert sha(OUT/ds/'SOURCE_PAIRS.json')==cal['datasets'][ds]['source_pairs']
        assert sha(OUT/ds/'PAIR_METADATA.json')==cal['datasets'][ds]['metadata']
        expected=[];meta=[]
        for sid,(r,z) in enumerate(zip(rr,saved)):
            assert r['source_id']==z['source_id'] and r['source_id_sha256']==z['source_id_sha256']
            eq(r['frozen_scores']['L'],z['scores']);eq(r['candidate_t'],z['candidate_t'])
            s=np.array(z['scores']);t=np.array(z['candidate_t']);scale=float(np.percentile(s,75)-np.percentile(s,25))
            local=[]
            for i in range(32):
                for j in range(i+1,32):
                    if abs(s[i]-s[j])<=EPS:continue
                    a,b=(i,j) if s[i]>s[j] else (j,i);m=float(s[a]-s[b])
                    local.append(dict(source_index=sid,source_id=r['source_id'],high_index=a,low_index=b,
                                      raw_margin=m,norm_margin=m/(scale+1e-8),delta_t=float(t[a]-t[b])))
            for v in local:v['weight']=1/len(local)
            expected.extend(local);meta.append(dict(source_id=r['source_id'],strict_pairs=len(local),score_ties=496-len(local),
                                                     IQR=scale,zero_IQR=scale==0,total_fit_weight=float(bool(local))))
        assert len(expected)==len(pr) and metadata==meta
        for a,b in zip(pr,expected):
            assert a.keys()==b.keys()
            eq(list(a.values()),list(b.values()))
        source_counts[ds]=dict(sources=len(rr),pairs=len(pr));models[ds]={}
        for arm,field in [('Pair-Raw','raw_margin'),('Pair-Norm','norm_margin')]:
            model=read(OUT/ds/(arm+'.json'));models[ds][arm]=model
            assert sha(OUT/ds/(arm+'.json'))==cal['datasets'][ds]['models'][arm]
            x=np.array([r[field] for r in expected]);y=np.array([r['delta_t'] for r in expected])
            w=np.array([r['weight'] for r in expected]);ids=np.array([r['source_index'] for r in expected])
            order=np.argsort(x,kind='mergesort');x,y,w,ids=x[order],y[order],w[order],ids[order]
            knots,central=fit(x,y,w);eq(knots,model['knots']);eq(central,model['mean'])
            eq(model['domain'],[knots[0],knots[-1]])
            assert model['source_count']==len(rr)==model['effective_independent_units'] and model['strict_pair_count']==len(pr)
            rng=np.random.default_rng(SEED);boots=np.empty((B,len(knots)))
            for b in range(B):
                counts=np.bincount(rng.integers(0,len(rr),len(rr)),minlength=len(rr));ww=w*counts[ids]
                k,v=fit(x,y,ww);boots[b]=np.interp(knots,k,v)
            eq(model['lower'],np.minimum(np.percentile(boots,5,axis=0),central))
            print('INDEPENDENT_SOURCE_BOOTSTRAP_PASS',ds,arm,flush=True)
    prior={r['cell_key']:r for r in read(OLD/'SCORE_ROWS.json')};dec=read(OUT/'DECISIONS.json');ev={r['cell_key']:r for r in dec};assert len(ev)==288
    for r in dec:
        p=prior[r['cell_key']];ds=p['dataset'];s=p['scores']['L'];a=p['anchor_index'];k=stats.winner(s,a)
        eq(r['scores'],s);assert r['top_index']==k and r['anchor_index']==a
        for f in ['candidate_indices','intervals','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:assert r[f]==p[f]
        spread=float(np.percentile(s,75)-np.percentile(s,25));eq(r['IQR'],spread);eq(r['raw_margin'],s[k]-s[a])
        assert r['choices']['A']==a and r['choices']['L32']==p['choices']['L32']==k
        for arm in ARMS[2:]:
            m=(s[k]-s[a])/(spread+1e-8) if arm=='Pair-Norm' else s[k]-s[a];model=models[ds][arm]
            inside=model['domain'][0]<=m<=model['domain'][1];e=r['evidence'][arm];eq(e['margin'],m)
            assert e['in_domain']==inside
            if inside:
                eq(e['mean'],np.interp(m,model['knots'],model['mean']));eq(e['lower'],np.interp(m,model['knots'],model['lower']))
            else:assert e['lower'] is None and e['mean'] is None
            reason=('no_unique_positive_winner' if k==a else 'outside_source_pair_margin_support' if not inside else
                    'nonpositive_lower_mean_delta' if e['lower']<=EPS else 'accepted')
            assert e['reason']==reason and r['choices'][arm]==(k if reason=='accepted' else a)
        ds0,de=np.array(p['intervals'][k])-p['intervals'][a]
        radius=(abs(ds0)+abs(de))/(p['intervals'][a][1]-p['intervals'][a][0]);eq(r['top_geometry']['radius'],radius)
        assert r['top_geometry']['large']==(radius>=.5-EPS)
    fields=['v','t','dv','dt','gross_gain','gross_loss','accepted','physical_replacement','benefit_t','accepted_dt',
            'severe','accepted_severe','benefit_v','harm_v','neutral_v']
    ratios=dict(beneficial_precision=('benefit_t','accepted'),accepted_mean_delta_t=('accepted_dt','accepted'),accepted_severe_rate=('accepted_severe','accepted'))
    counts=collections.Counter()
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            rows=read(OUT/sp/ds/'ROWS.json');old={stats_key(r):r for r in read(OLD/sp/ds/'ROWS.json')}
            for r in rows:
                p=old[stats_key(r)];counts['arrivals']+=1;counts['expert' if r['expert_scheduled'] else 'nonexpert']+=1
                for f in ['expert_scheduled','source_id','A_state_pre_sha256','A_state_post_sha256','A8_v','A8_t']:assert r[f]==p[f]
                d=ev[stats_key(r)] if r['expert_scheduled'] else None
                if d:
                    eq(r['candidate_v'],p['candidate_v']);eq(r['candidate_t'],p['candidate_t']);assert r['choices']==d['choices']
                for arm in ARMS:
                    i=d['choices'][arm] if d else None;v=p['candidate_v'][i] if d else p['A8_v'];t=p['candidate_t'][i] if d else p['A8_t']
                    accepted=d is not None and i!=p['anchor_index'];dv=v-p['A8_v'];dt=t-p['A8_t']
                    physical=accepted and np.max(abs(np.array(d['intervals'][i])-d['intervals'][p['anchor_index']]))>EPS
                    eq([r['arms'][arm][f] for f in fields],[v,t,dv,dt,max(dv,0),max(-dv,0),accepted,physical,
                        accepted and dt>EPS,dt if accepted else 0,dv<-.05,accepted and dv<-.05,dv>EPS,dv< -EPS,abs(dv)<=EPS])
                assert r['arms']['L32']['v']==p['L32_v'] and r['arms']['L32']['t']==p['L32_t']
            summary=read(OUT/sp/ds/'SUMMARY.json');strata=read(OUT/sp/ds/'DIAGNOSTIC_STRATA.json')
            def validate(saved,rr,arm):
                packs=[dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],**r['arms'][arm]) for r in rr]
                stats.validate_stat(saved,packs,fields,ratios)
                assert saved['accepting_sources']==len({r['source_id'] for r in rr if r['arms'][arm]['accepted']})
                for f,n in saved.get('raw_counts',{}).items():eq(n,sum(r['arms'][arm][f] for r in rr))
            for cat in ['corruption','clean']:
                rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean')]
                for sub in ['all','expert','nonexpert']:
                    packs=[r for r in rr if sub=='all' or r['expert_scheduled']==(sub=='expert')]
                    for arm in ARMS:validate(summary[cat][sub][arm],packs,arm)
                paired=[]
                for r in rr:
                    z=dict(source_id=r['source_id'],order=r['order'],condition=r['condition'])
                    for arm in ARMS[2:]:
                        for f in ['v','t','gross_loss']:z[arm+'_vs_L32_'+f]=r['arms'][arm][f]-r['arms']['L32'][f]
                    z['Norm_vs_Raw_v']=r['arms']['Pair-Norm']['v']-r['arms']['Pair-Raw']['v'];paired.append(z)
                stats.validate_stat(summary[cat]['paired'],paired,list(paired[0])[3:])
                for group in ['all','small','large']:
                    packs=[r for r in rr if r['expert_scheduled'] and (group=='all' or r['top_geometry']['large']==(group=='large'))]
                    for arm in ARMS:validate(strata[cat][group][arm],packs,arm)
    assert dict(counts)==dict(arrivals=1152,expert=288,nonexpert=864)
    decision=read(OUT/'DECISION.json');joint={}
    for arm in ARMS[2:]:
        good=[]
        for ds in ['vidstg','hc2']:
            z=read(OUT/'confirm'/ds/'SUMMARY.json')['corruption']['all'];m=z[arm]['metrics'];l=z['L32']
            test=dict(positive_mean=m['dv']['mean']>EPS,accepting_sources=z[arm]['accepting_sources']>=3,
                positive_leave_one_out=m['dv']['leave_one_out_range'][0]>EPS,less_gross_loss=m['gross_loss']['mean']<l['metrics']['gross_loss']['mean']-EPS,
                fewer_severe_harms=z[arm]['raw_counts']['severe']<l['raw_counts']['severe'])
            d=decision['arm_decisions'][arm][ds];assert d['tests']==test and d['pass_all']==all(test.values());eq(d['mean_v_delta'],m['dv']['mean'])
            assert d['accepting_sources']==z[arm]['accepting_sources'];good.append(all(test.values()))
        joint[arm]=all(good)
    assert decision['status']==('GO_development_only' if joint['Pair-Norm'] else 'NO_GO_locked_pairwise_certification')
    assert decision['raw_joint_go']==joint['Pair-Raw'] and not decision['method_promoted'] and not decision['followup_started']
    if (OUT/'FIGURE_DATA.json').exists():
        z=read(OUT/'FIGURE_DATA.json')
        for ds in ['vidstg','hc2']:
            assert z['confirmation'][ds]==read(OUT/'confirm'/ds/'SUMMARY.json')
            assert z['strata'][ds]==read(OUT/'confirm'/ds/'DIAGNOSTIC_STRATA.json')
            assert z['calibration'][ds]==models[ds]
    result=dict(status='pass',scalar_checks=int(stats.checks),maximum_numeric_error=stats.maximum,
                source_counts=source_counts,independent_SciPy_PAV_10000_source_bootstrap=True,
                source_pair_weight_reconstruction=True,all_target_choices_and_frozen_A_L32_parity=True,
                cached_metrics_and_paired_source_intervals=True,global_seal_before_label_join=True,
                GO_recomputed=True,figure_data_readback=(OUT/'FIGURE_DATA.json').exists(),CPU_wall_seconds=time.time()-start)
    print(json.dumps(result,indent=2));return result

def stats_key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
if __name__=='__main__':audit()

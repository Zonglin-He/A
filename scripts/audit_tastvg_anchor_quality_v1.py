"""Independent anonymous audit: SciPy pivoted QR, fresh metrics/bootstrap."""
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
from scipy import linalg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_tastvg_anchor_certification_v1 as audit_stats
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_anchor_quality/2026-10-03'
OLD=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'
PAIR=ROOT/'results/tastvg_pairwise_certification/2026-10-03'
EPS=1e-12;ARMS=['A','L32','Pair-Norm','M0','M1']
FIELDS=['v','t','dv','dt','gross_gain','gross_loss','accepted','physical_replacement','benefit_t','accepted_dt','severe','accepted_severe','benefit_v','harm_v','neutral_v']
RATIOS=dict(beneficial_precision=('benefit_t','accepted'),accepted_mean_delta_t=('accepted_dt','accepted'),accepted_severe_rate=('accepted_severe','accepted'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):audit_stats.eq(a,b)
def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
def features(z,a):
    med=float(np.percentile(z,50));q=np.percentile(z,[25,75]);den=q[1]-q[0]+1e-8
    ids=[i for i,v in enumerate(z) if max(z)-v<=EPS];unique=len(ids)==1;k=ids[0] if unique else a
    return k,unique,(z[k]-z[a])/den,(z[a]-med)/den,(z[k]-med)/den,float(q[1]-q[0])
def solve(rows,arm):
    x=np.array([[1,r['margin']]+([r['anchor_score']] if arm=='M1' else []) for r in rows],float)
    y=np.array([r['y'] for r in rows]);w=np.sqrt([r['weight'] for r in rows]);xx=x*w[:,None]
    b,_,rank,_=linalg.lstsq(xx,y*w,cond=1e-12,lapack_driver='gelsy')
    singular=linalg.svdvals(xx)
    return b,int(rank),singular
def predict(b,r):return float(b[0]+b[1]*r['margin']+(b[2]*r['anchor_score'] if len(b)==3 else 0))
def corr(x,y):
    if len(x)<2 or np.std(x)<=EPS or np.std(y)<=EPS:return None
    return float(np.corrcoef(x,y)[0,1])
def check_summary(saved,rows):
    for a in ARMS:
        packs=[dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],**r['arms'][a]) for r in rows]
        audit_stats.validate_stat(saved[a],packs,FIELDS,RATIOS)
        eq(saved[a]['accepting_sources'],len({r['source_id'] for r in rows if r['arms'][a]['accepted']}))
        for f,n in saved[a]['raw_counts'].items():eq(n,sum(r['arms'][a][f] for r in rows))
def check_quartiles(saved,rows,arms):
    rr=sorted(rows,key=lambda r:(r['A8_t'],r['cell_key']))
    for q,g in zip(saved,np.array_split(np.array(rr,object),4)):
        assert q['cell_keys']==[r['cell_key'] for r in g];eq(q['cells'],len(g));eq(q['sources'],len({r['source_id'] for r in g}))
        eq(q['cell_mean_A8_t'],np.mean([r['A8_t'] for r in g]));eq(q['A8_t_range'],[g[0]['A8_t'],g[-1]['A8_t']])
        for a in arms:
            z=q['arms'][a]
            for f in ['dt','dv','accepted']:eq(z['cell_means'][f],np.mean([r['arms'][a][f] for r in g]))
            for f,n in z['counts'].items():eq(n,sum(r['arms'][a][f] for r in g))
            eq(z['harmful_t'],sum(r['arms'][a]['dt']< -EPS for r in g));eq(z['neutral_t'],sum(abs(r['arms'][a]['dt'])<=EPS for r in g))
            packs=[dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],A8_t=r['A8_t'],**r['arms'][a]) for r in g]
            audit_stats.validate_stat(z['source_balanced'],packs,['A8_t','dt','dv','accepted','gross_gain','gross_loss','severe'])
        if 'proposal_retention' in q:
            for a in ['Pair-Norm','M0','M1']:
                p=q['proposal_retention'][a];good=[r for r in g if r['arms']['L32']['dt']>EPS];bad=[r for r in g if r['arms']['L32']['dt']< -EPS]
                eq(p['helpful_total'],len(good));eq(p['helpful_accepted'],sum(r['arms'][a]['accepted']>0 for r in good))
                eq(p['harmful_total'],len(bad));eq(p['harmful_rejected'],sum(r['arms'][a]['accepted']==0 for r in bad))

def audit():
    tick=time.time();binding=read(OUT/'RUNTIME_BINDING.json');cfg=read(OUT/'CONFIG.json')
    for p,h in {**binding['code'],**binding['public_required_inputs']}.items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'CONFIG.json')==binding['config_sha256']
    assert cfg['source_LOSO_selects_nothing'] and not cfg['target_GT_fit'] and not cfg['target_threshold_selection']
    assert cfg['IQR_epsilon']==1e-8 and cfg['rcond']==1e-12
    private=ROOT/'artifacts/tastvg_anchor_quality_v1/RUNTIME_LOCK.json'
    if private.exists():
        lock=read(private)
        for p,h in {**lock['code'],**lock['inputs']}.items():assert sha(ROOT/p)==h,p
    exposure=read(OUT/'GT_EXPOSURE.json');baseline=read(OUT/'BASELINE_DIAGNOSIS.json');cal=read(OUT/'CALIBRATION_SEAL.json')
    seal=read(OUT/'GLOBAL_DECISION_SEAL.json');join=read(OUT/'LABEL_JOIN.json')
    assert exposure['time']<cal['time']<seal['time']<join['time']
    assert cal['no_target_label_or_case_reads_in_fit_process'] and seal['no_target_label_or_case_reads_in_decision_process']
    assert seal['prior_baseline_GT_exposure_disclosed'] and not join['target_GT_fit']
    assert sha(OUT/'DECISIONS.json')==seal['decisions_sha256'] and sha(OUT/'CALIBRATION_SEAL.json')==seal['calibration_sha256']
    for p,h in {**exposure['label_files'],**join['label_files']}.items():assert sha(ROOT/p)==h
    src={(r['dataset'],r['source_id']):r for r in read(OLD/'SOURCE_ROWS.json')};source_models={};source_loso={}
    for ds in ['vidstg','hc2']:
        rows=read(OUT/ds/'SOURCE_CANDIDATES.json');pairs=read(OUT/ds/'SOURCE_PAIRS.json');meta=read(OUT/ds/'PAIR_METADATA.json')
        assert len(rows)=={'vidstg':31,'hc2':16}[ds] and len(pairs)==32*len(rows)
        for r in rows:
            old=src[(ds,r['source_id'])];assert old['split']=='validation'
            eq(r['scores'],old['frozen_scores']['L']);eq(r['candidate_t'],old['candidate_t']);assert r['source_id_sha256']==old['source_id_sha256']
        expected=[]
        for i,r in enumerate(rows):
            k,u,_,_,c,iqr=features(r['scores'],0);assert u
            z=meta[i];assert z['source_id']==r['source_id'] and z['top_index']==k and z['unique_top']
            eq(z['total_weight'],1);eq(z['pseudo_anchors'],32);eq(z['top_context'],c);eq(z['IQR'],iqr)
            for j in range(32):
                k,u,m,a,c,iqr=features(r['scores'],j)
                expected.append(dict(source_id=r['source_id'],source_index=i,pseudo_anchor=j,top_index=k,margin=m,anchor_score=a,top_context=c,
                    y=r['candidate_t'][k]-r['candidate_t'][j],weight=1/32,eligible=u and r['scores'][k]>r['scores'][j]+EPS))
        for a,b in zip(pairs,expected):
            for f in a:eq(a[f],b[f])
            eq(a['margin']+a['anchor_score'],a['top_context'])
        source_models[ds]={}
        for a in ['M0','M1']:
            model=read(OUT/ds/(a+'.json'));b,rank,singular=solve(expected,a)
            eq(model['coefficients'],b);eq(model['rank'],rank);eq(model['singular_values'],singular)
            eq(model['condition_number'],singular[0]/singular[-1]);eq(model['pairs'],len(expected));eq(model['source_count'],len(rows))
            for f in ['margin','anchor_score']:eq(model['feature_ranges'][f],[min(r[f] for r in expected),max(r[f] for r in expected)])
            source_models[ds][a]=model
            assert sha(OUT/ds/(a+'.json'))==cal['datasets'][ds]['models'][a]
        for n,f in [('source_candidates','SOURCE_CANDIDATES.json'),('pairs','SOURCE_PAIRS.json'),('metadata','PAIR_METADATA.json'),
                    ('folds','LOSO_FOLDS.json'),('oof','LOSO_ROWS.json'),('summary','LOSO_SUMMARY.json')]:assert sha(OUT/ds/f)==cal['datasets'][ds][n]
        folds=read(OUT/ds/'LOSO_FOLDS.json');oof=read(OUT/ds/'LOSO_ROWS.json');check_oof=[]
        for fold in folds:
            sid=fold['held_out_source'];tr=[r for r in expected if r['source_id']!=sid];te=[r for r in expected if r['source_id']==sid];fm={}
            for a in ['M0','M1']:
                b,rank,s=solve(tr,a);eq(fold['models'][a]['coefficients'],b);eq(fold['models'][a]['rank'],rank);eq(fold['models'][a]['singular_values'],s);fm[a]=b
            for r in te:
                pred={a:predict(fm[a],r) for a in fm};e={a:pred[a]-r['y'] for a in pred}
                check_oof.append(dict(**r,predictions=pred,M0_MSE=e['M0']**2,M1_MSE=e['M1']**2,zero_MSE=r['y']**2,
                    M0_MAE=abs(e['M0']),M1_MAE=abs(e['M1']),MSE_improvement=e['M0']**2-e['M1']**2))
        for a,b in zip(oof,check_oof):
            assert a['source_id']==b['source_id'] and a['pseudo_anchor']==b['pseudo_anchor']
            for f in ['M0_MSE','M1_MSE','zero_MSE','M0_MAE','M1_MAE','MSE_improvement']:eq(a[f],b[f])
            for arm in ['M0','M1']:eq(a['predictions'][arm],b['predictions'][arm])
        ls=read(OUT/ds/'LOSO_SUMMARY.json');source_loso[ds]=ls
        for name,rr in [('all_pseudo_anchors',check_oof),('eligible_nonself',[r for r in check_oof if r['eligible']])]:
            pp=[dict(r,order='LOSO',condition='source') for r in rr]
            audit_stats.validate_stat(ls[name],pp,['M0_MSE','M1_MSE','zero_MSE','M0_MAE','M1_MAE','MSE_improvement'])
            eq(ls[name]['positive_sources'],sum(v>EPS for v in ls[name]['metrics']['MSE_improvement']['source_values'].values()))
        eq(ls['GT_positive'],sum(r['y']>EPS for r in expected));eq(ls['GT_negative'],sum(r['y']< -EPS for r in expected));eq(ls['GT_neutral'],sum(abs(r['y'])<=EPS for r in expected))
        for r in rows:eq(ls['within_source_anchor_proxy_GT_correlations'][str(r['source_id'])],corr(r['scores'],r['candidate_t']))
    score={r['cell_key']:r for r in read(OLD/'SCORE_ROWS.json')};prior={r['cell_key']:r for r in read(PAIR/'DECISIONS.json')}
    decisions=read(OUT/'DECISIONS.json');ev={r['cell_key']:r for r in decisions};assert len(ev)==288;choice_counts=collections.Counter()
    for d in decisions:
        r=score[d['cell_key']];a=r['anchor_index'];k,u,m,q,c,iqr=features(r['scores']['L'],a)
        for f in ['candidate_indices','intervals','A_state_pre_sha256','A_state_post_sha256','pixel_sha256','probe_sha256']:assert d[f]==r[f]
        eq(d['scores'],r['scores']['L']);eq(d['margin'],m);eq(d['anchor_score'],q);eq(d['top_context'],c);eq(d['IQR'],iqr)
        assert d['choices']['A']==a and d['choices']['L32']==r['choices']['L32']==k
        assert d['choices']['Pair-Norm']==prior[d['cell_key']]['choices']['Pair-Norm']
        for arm in ['M0','M1']:
            model=source_models[r['dataset']][arm];mu=predict(model['coefficients'],dict(margin=m,anchor_score=q))
            accepted=u and r['scores']['L'][k]>r['scores']['L'][a]+EPS and mu>EPS
            eq(d['evidence'][arm]['predicted_delta_t'],mu);assert d['choices'][arm]==(k if accepted else a)
            assert d['evidence'][arm]['accepted']==accepted
            features_used={'margin':m} if arm=='M0' else dict(margin=m,anchor_score=q)
            outside=any(v<model['feature_ranges'][f][0] or v>model['feature_ranges'][f][1] for f,v in features_used.items())
            assert d['evidence'][arm]['outside_marginal_source_ranges']==outside
            choice_counts[arm+'/accepted']+=accepted
        p,q0=np.array(r['intervals'][k])-r['intervals'][a];radius=(abs(p)+abs(q0))/(r['intervals'][a][1]-r['intervals'][a][0]);eq(d['top_geometry']['radius'],radius)
    total=expert=0;decision=read(OUT/'DECISION.json');check_dec={};cited=[]
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            old={key(r):r for r in read(OLD/sp/ds/'ROWS.json')};rows=read(OUT/sp/ds/'ROWS.json');assert len(rows)==len(old)
            for r in rows:
                p=old[r['cell_key']];total+=1;expert+=int(p['expert_scheduled']);assert r['expert_scheduled']==p['expert_scheduled']
                for f in ['A_state_pre_sha256','A_state_post_sha256']:assert r[f]==p[f]
                eq(r['A8_t'],p['A8_t']);eq(r['A8_v'],p['A8_v']);d=ev[r['cell_key']] if p['expert_scheduled'] else None
                if d:assert r['choices']==d['choices'];eq(r['candidate_v'],p['candidate_v']);eq(r['candidate_t'],p['candidate_t'])
                for a in ARMS:
                    i=d['choices'][a] if d else None;v=p['candidate_v'][i] if d else p['A8_v'];t=p['candidate_t'][i] if d else p['A8_t']
                    accepted=d is not None and i!=p['anchor_index'];physical=accepted and max(abs(x-y) for x,y in zip(d['intervals'][i],d['intervals'][p['anchor_index']]))>EPS
                    dv=v-p['A8_v'];dt=t-p['A8_t']
                    z=[v,t,dv,dt,max(dv,0),max(-dv,0),accepted,physical,accepted and dt>EPS,dt if accepted else 0,
                       dv<-.05,accepted and dv<-.05,dv>EPS,dv< -EPS,abs(dv)<=EPS]
                    eq([r['arms'][a][f] for f in FIELDS],z)
                if r['expert_scheduled'] and ds=='vidstg' and sp=='confirm' and r['source_id'] in [34,37] and 'exposure' in r['condition']:cited.append(r)
            summary=read(OUT/sp/ds/'SUMMARY.json')
            for cat in ['corruption','clean']:
                rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean')]
                for sub in ['all','expert','nonexpert']:
                    pp=[r for r in rr if sub=='all' or r['expert_scheduled']==(sub=='expert')];check_summary(summary[cat][sub],pp)
                pairs=[]
                for r in rr:
                    p=dict(source_id=r['source_id'],order=r['order'],condition=r['condition'])
                    for a,b in [('M1','M0'),('M1','L32'),('M0','L32'),('M1','Pair-Norm')]:
                        for f in ['v','t','gross_loss']:p[a+'_vs_'+b+'_'+f]=r['arms'][a][f]-r['arms'][b][f]
                    pairs.append(p)
                audit_stats.validate_stat(summary[cat]['paired'],pairs,list(pairs[0].keys())[3:])
            rr=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean'];strata=read(OUT/sp/ds/'ANCHOR_STRATA.json')
            check_quartiles(strata['quartiles'],rr,ARMS);eq(strata['anchor_score_vs_A8_t'],corr([r['anchor_score'] for r in rr],[r['A8_t'] for r in rr]))
            bp=baseline['panels'][ds+'/'+sp];oldb={r['cell_key']:r for r in bp['rows']}
            for r in rr:
                b=oldb[r['cell_key']]
                for a in ['A','L32','Pair-Norm']:eq([b['arms'][a][f] for f in FIELDS],[r['arms'][a][f] for f in FIELDS])
                for f in ['margin','anchor_score','top_context']:eq(b[f],r[f])
            check_quartiles(bp['quartiles'],rr,['L32','Pair-Norm'])
            br=bp['rows'];random_delta=[np.mean(old[r['cell_key']]['candidate_t'])-r['A8_t'] for r in br]
            eq(bp['mean_candidate_delta'],random_delta)
            eq(bp['correlations']['A8_t_vs_L32_dt'],corr([r['A8_t'] for r in br],[r['arms']['L32']['dt'] for r in br]))
            eq(bp['correlations']['A8_t_vs_mean_candidate_delta'],corr([r['A8_t'] for r in br],random_delta))
            eq(bp['correlations']['anchor_score_vs_A8_t'],corr([r['anchor_score'] for r in br],[r['A8_t'] for r in br]))
            if sp=='confirm':
                s=summary['corruption']['all'];m=s['M1'];l=s['L32'];ls=source_loso[ds]
                tests=dict(source_LOSO_MSE_improves=ls['all_pseudo_anchors']['metrics']['MSE_improvement']['mean']>EPS,
                    positive_mean=m['metrics']['dv']['mean']>EPS,better_than_M0=summary['corruption']['paired']['metrics']['M1_vs_M0_v']['mean']>EPS,
                    accepting_sources=m['accepting_sources']>=3,positive_leave_one_out=m['metrics']['dv']['leave_one_out_range'][0]>EPS,
                    less_gross_loss=m['metrics']['gross_loss']['mean']<l['metrics']['gross_loss']['mean']-EPS,
                    fewer_severe_harms=m['raw_counts']['severe']<l['raw_counts']['severe'])
                assert decision['datasets'][ds]['tests']==tests;assert decision['datasets'][ds]['pass_all']==all(tests.values());check_dec[ds]=all(tests.values())
    assert total==1152 and expert==288
    assert decision['status']==('GO_development_only' if all(check_dec.values()) else 'NO_GO_locked_anchor_score_linear_models')
    assert not decision['method_promoted'] and not decision['followup_started'] and not decision['causal_anchor_blindness_identified']
    output=dict(status='pass',time=time.time(),scalar_checks=audit_stats.checks,maximum_numeric_error=audit_stats.maximum,
        arrivals=total,expert=expert,nonexpert=total-expert,choice_counts=dict(choice_counts),CPU_wall_seconds=time.time()-tick,
        independent_solver='SciPy pivoted QR against NumPy SVD; independent source-level bootstrap and scalar metric reconstruction',
        hash_seal_order_verified=True,baseline_GT_exposure_disclosed=True,source_LOSO_all_folds_verified=True,
        cited_case_readback=cited,convenience_filter_note='Baseline named-case lists used exact exposure but actual condition is prefixed; all rows/strata intact; these cases use the real condition string.')
    name='PUBLIC_AUDIT.json' if len(sys.argv)>2 and sys.argv[2]=='public' else 'ROOT_AUDIT.json'
    (OUT/name).write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='cited_case_readback'},indent=2))

if __name__=='__main__':audit()

"""Independent full-query/parent readback, paired contrasts, tails and figures.
All files consumed here are postseal CPU outputs; no model execution or method selection.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

FIELDS=['After_v','After_t','After_s','After_R30','After_R50','delta_total_v','delta_current_v','delta_inherited_v']
METHODS=['source_only','tent_stvg','sar_stvg','dino_refine','spatial_opd','target_trained_reference']
OLDPUB=ROOT/'results/decota_paper_baseline_scoring/2026-10-08'

def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]

def independent(rr,fields=FIELDS):
    # Sum by query, then average the complete queries within each parent.
    qsum={};qn=collections.Counter();qparent={}
    for r in rr:
        q=r['query_ordinal'];qparent[q]=r['source_id'];qn[q]+=1
        if q not in qsum:qsum[q]=np.zeros(len(fields))
        qsum[q]+=np.array([r[f] for f in fields])
    assert qsum and set(qn.values())=={3}
    psum={};pn=collections.Counter();queryvalues=[]
    for q in sorted(qsum):
        value=qsum[q]/qn[q];s=qparent[q];queryvalues.append(value);pn[s]+=1
        if s not in psum:psum[s]=np.zeros(len(fields))
        psum[s]+=value
    sources=sorted(psum);matrix=np.stack([psum[s]/pn[s] for s in sources]);rng=np.random.default_rng(20261008)
    bootstrap=np.concatenate([matrix[rng.integers(0,len(sources),(50,len(sources)))].sum(1)/len(sources) for _ in range(200)])
    ci=np.percentile(bootstrap,[2.5,97.5],axis=0);queryvalues=np.stack(queryvalues)
    summary=dict(queries=len(qsum),parent_sources=len(sources),logical_arrivals=len(rr),metrics={
        f:dict(source_macro=float(matrix[:,j].sum()/len(sources)),official_query_macro=float(queryvalues[:,j].sum()/len(qsum)),
            paired_parent_ci95=ci[:,j].tolist(),harm_gt5pp_parent_sources=int((matrix[:,j]<-.05).sum()) if f.startswith('delta_') else None,
            harm_gt20pp_parent_sources=int((matrix[:,j]<-.2).sum()) if f.startswith('delta_') else None) for j,f in enumerate(fields)})
    return summary,sources,matrix

def adapt(r):
    return {**r,'query_ordinal':r['query_id'],'After_R30':r['After_R0.3'],'After_R50':r['After_R0.5']}

def load_methods(ds):
    result={'spatial_opd':rows(PUB/('P1_'+ds)/'ROWS.jsonl.gz')}
    for m in METHODS+(['eata_stvg'] if ds=='hc2' else []):
        if m!='spatial_opd':result[m]=[adapt(r) for r in rows(OLDPUB/ds/(m+'.jsonl.gz'))]
    return result

def feature_attribution(rr):
    # Description only: quartiles do not choose parameters, filters or a new roster.
    def group(items):
        bysource=collections.defaultdict(list)
        for r in items:bysource[r['source_id']].append(r)
        return dict(arrivals=len(items),parents=len(bysource),
            parent_mean_delta_total_v=float(np.mean([np.mean([r['delta_total_v'] for r in x]) for x in bysource.values()])) if bysource else None,
            parent_mean_delta_current_v=float(np.mean([np.mean([r['delta_current_v'] for r in x]) for x in bysource.values()])) if bysource else None)
    quality={name:group([r for r in rr if test(r['admitted_expert_GT_IoU'])]) for name,test in [
        ('missing_GT_evaluable_expert',lambda x:x is None),('expert_IoU_below_0.3',lambda x:x is not None and x<.3),
        ('expert_IoU_0.3_to_0.5',lambda x:x is not None and .3<=x<.5),('expert_IoU_at_least_0.5',lambda x:x is not None and x>=.5)]}
    types=sorted({json.dumps(r['query_type'],sort_keys=True) for r in rr});features={};cutpoints={}
    # Feature thresholds are computed once per distinct query, independently of outcomes.
    first={r['query_ordinal']:r for r in rr}
    for f in ['event_duration_fraction','normalized_target_motion']:
        cuts=np.quantile([r[f] for r in first.values()],[.25,.5,.75]);cutpoints[f]=cuts.tolist()
        features[f]={f'quartile_{j+1}':group([r for r in rr if np.searchsorted(cuts,r[f],side='right')==j]) for j in range(4)}
    return dict(quality=quality,features=features,cutpoints=cutpoints,
        query_type={kind:group([r for r in rr if json.dumps(r['query_type'],sort_keys=True)==kind]) for kind in types},
        descriptive_only_not_an_intervention=True)

def run():
    verify();assert read(BASE/'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
    assert read(BASE/'P1_TABLE_ASSEMBLY_COMPLETION.json')['status']=='pending_root_independent_table_audit_visual_publication'
    assembled=read(PUB/'TABLE1_SUMMARY.json');binding=read(BASE/'P1_BASELINE_BINDING.json')
    for f,h in binding['files'].items():assert sha(ROOT/f)==h
    independent_results={};contrasts={};effects={};costs={};diagnosis={};checks=0;count=0
    for ds in DATASETS:
        completion=read(BASE/'stages'/('P1_'+ds)/'CPU_COMPLETION.json')
        for f,h in completion['files'].items():assert sha(ROOT/f)==h
        methods=load_methods(ds);full=methods['spatial_opd'];match={(r['query_ordinal'],r['order']):r for r in full}
        expected=3482 if ds=='hc2' else 10303;assert len(match)==expected*3
        dev=set(read(PUB/('P1_'+ds)/'SUMMARY.json')['excluded_development_anonymous_source_ids']);assert len(dev)==32
        independent_results[ds]={};contrasts[ds]={};effects[ds]={}
        for method,rr in methods.items():
            assert len(rr)==expected*3;count+=len(rr)
            for r in rr:
                other=match[r['query_ordinal'],r['order']]
                assert r['source_id']==other['source_id'] and r['arrival']==other['arrival']
                for f in ['v','t','s']:assert abs(r['Frozen_'+f]-other['Frozen_'+f])<2e-10;checks+=1
                assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v'])<2e-10;checks+=1
            independent_results[ds][method]={}
            for subset,part in [('all_official_queries',rr),('excluding_32_tuning_parent_sources',[r for r in rr if r['source_id'] not in dev])]:
                check,sources,matrix=independent(part);saved=assembled['datasets'][ds][method][subset]
                assert check['queries']==saved['queries'] and check['parent_sources']==saved['parent_sources'];checks+=2
                for f,v in check['metrics'].items():
                    old=saved['metrics'][f]
                    for key in ['source_macro','official_query_macro']:assert abs(v[key]-old[key])<3e-12;checks+=1
                    assert np.max(np.abs(np.array(v['paired_parent_ci95'])-old['paired_parent_ci95']))<3e-12;checks+=1
                independent_results[ds][method][subset]=check
                if subset=='all_official_queries':
                    j=FIELDS.index('delta_total_v');effects[ds][method]=[dict(source_id=s,delta_total_v=float(matrix[i,j])) for i,s in enumerate(sources)]
            paired=[]
            for r in rr:
                f=match[r['query_ordinal'],r['order']]
                paired.append(dict(query_ordinal=r['query_ordinal'],source_id=r['source_id'],order=r['order'],
                    **{'OPD_minus_'+k:f['After_'+k]-r['After_'+k] for k in ['v','t','s','R30','R50']}))
            contrast,_,_=independent(paired,['OPD_minus_'+k for k in ['v','t','s','R30','R50']]);contrasts[ds][method]=contrast
        for r in full:
            assert r['Frozen_t']==r['Before_t']==r['After_t'];checks+=1
        costs[ds]=dict(arrivals=len(full),
            synchronized_fit_wall_seconds=float(sum(r['compute']['fit_GPU_seconds'] for r in full)),
            synchronized_fit_wall_seconds_per_arrival=float(np.mean([r['compute']['fit_GPU_seconds'] for r in full])),
            shared_capture_wall_seconds=float(sum(r['compute']['shared_capture_seconds'] for r in full)),
            independent_CPU_math_seconds=float(sum(r['compute']['CPU_math_seconds'] for r in full)),
            actual_new_DINO_calls=int(sum(r['compute']['new_DINO_calls'] for r in full)),
            nominal_DINO_observation_budget=4,
            max_CUDA_peak_allocated=int(max(r['compute']['CUDA_peak_allocated'] for r in full)),
            actual_backward_calls=int(sum(r['gradient_calls'] for r in full)),
            postseal_CPU_score_seconds=read(PUB/('P1_'+ds)/'SUMMARY.json')['CPU_seconds'],
            old_matched_DINO_inputs_reused=True,not_cold_uncached_end_to_end_latency=True)
        diagnosis[ds]=feature_attribution(full)
    write(PUB/'TABLE1_ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',independent_checks=checks,
        anonymous_logical_rows=count,datasets=independent_results,paired_OPD_vs_methods=contrasts,
        original_baseline_binding_sha256=sha(BASE/'P1_BASELINE_BINDING.json'),no_parameter_selection=True,
        decoder_Jacobian_independently_reimplemented=False,time=time.time()))
    write(PUB/'TABLE1_ALL_PARENT_EFFECTS.json',effects);write(PUB/'TABLE1_COST.json',costs);write(PUB/'TABLE1_FAILURE_STRATA.json',diagnosis)
    render(independent_results,effects)
    write(BASE/'P1_CPU_COMPLETION.json',dict(status='pending_actual_root_visual_and_publication',
        actual_new_OPD_arrivals=41355,all_table_anonymous_rows=count,independent_checks=checks,
        root_must_continue_real_visual_publication_then_P2_P6=True,paper_suite_complete=False,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in PUB.glob('TABLE1*') if p.is_file()},time=time.time()))
    status(BASE/'P1_CPU_STAGE.json',dict(status='pending_actual_root_visual_and_publication',time=time.time()))
    print('P1_FULL_TABLE_ROOT_CPU_READBACK',count,checks)

def render(result,effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,ds in zip(axes,DATASETS):
        ms=METHODS+(['eata_stvg'] if ds=='hc2' else [])
        for y,m in enumerate(ms):
            metric=result[ds][m]['all_official_queries']['metrics']['After_v'];lo,hi=np.array(metric['paired_parent_ci95'])*100;mean=metric['source_macro']*100
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color='#1261a0' if m=='spatial_opd' else '#888' if m=='target_trained_reference' else '#468276')
            ax.plot(metric['official_query_macro']*100,y,marker='x',color='#111',ms=5)
        ax.set_yticks(range(len(ms)),['OPD (fixed)' if m=='spatial_opd' else 'Supervised in-domain reference' if m=='target_trained_reference' else m.replace('_',' ') for m in ms])
        ax.set_xlabel('m_vIoU (%): circles parent macro ± 95% CI; x query macro');ax.set_title(ds+' full clean cross-domain');ax.grid(axis='x',alpha=.15)
    fig.savefig(PUB/'TABLE1_full_query_parent.png',dpi=180);fig.savefig(PUB/'TABLE1_full_query_parent.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,ds in zip(axes,DATASETS):
        for m in ['spatial_opd','tent_stvg','sar_stvg','dino_refine']:
            vv=np.sort([r['delta_total_v']*100 for r in effects[ds][m]]);ax.plot(np.linspace(0,100,len(vv)),vv,label=m.replace('_',' '))
        ax.axhline(0,color='#777',lw=1);ax.axhline(-5,color='#c06',ls='--',lw=.7);ax.axhline(-20,color='#c06',ls=':',lw=.7)
        ax.set_xlabel('All parents ranked within method (%)');ax.set_ylabel('Method − Source Only vIoU (pp)');ax.set_title(ds+' complete positive and negative tails');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(PUB/'TABLE1_complete_negative_tails.png',dpi=180);fig.savefig(PUB/'TABLE1_complete_negative_tails.pdf');plt.close(fig)

if __name__=='__main__':run()

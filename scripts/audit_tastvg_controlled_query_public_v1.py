"""Independent anonymous coverage, source aggregation and paired bootstrap audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_tastvg_query_swap_public_v1 as math


def subset(rows,key):
    mode,panel,cohort=key.split('/')
    rr=[r for r in rows if (r['condition']=='clean')==(mode=='clean')]
    if panel in ['search','confirm']:rr=[r for r in rr if r['panel']==panel]
    elif panel!='all':rr=[r for r in rr if r['order']==panel]
    if cohort.endswith('_available'):
        rr=[r for r in rr if cohort[:-10] in r['available_arms']]
    elif cohort in ['common','common_strict']:
        rr=[r for r in rr if {'event','subject'}<=set(r['available_arms'])]
        if cohort=='common_strict':rr=[r for r in rr if max(r['tiers'].values())<=2]
    elif cohort.startswith('same_video_'):
        arm=cohort[11:];rr=[r for r in rr if arm in r['available_arms'] and r['tiers'][arm]<=1]
    else:assert cohort=='all'
    return rr


def audit(folder):
    folder=Path(folder);cfg=math.read(folder/'CONFIG.json');mapping=math.read(folder/'QUERY_MAPPING.json')
    assert cfg['original_cells']==288 and cfg['corrupt_cells']==240 and cfg['clean_cells']==48
    for key in ['probe_refits','new_expert_calls','new_candidates','parameter_updates']:assert cfg[key]==0
    assert cfg['annotation_assisted_referent_matching'] and not cfg['GT_time_or_metric_used_for_pairing']
    assert cfg['source_models_unchanged']==136 and cfg['support']==32 and cfg['old8_preserved']
    assert cfg['donors_fixed_in_bootstrap'] and cfg['donor_reuse_allowed']
    totals={'vidstg':(30,36,24,18,12,6),'hc2':(138,120,120,30,0,0)}
    for ds in ['vidstg','hc2']:
        mm=[m for m in mapping if m['dataset']==ds]
        assert len(mm)==96 and len({(m['source_index'],m['arm']) for m in mm})==96
        lookup={(m['source_index'],m['arm']):m for m in mm}
        for m in mm:
            if not m['available']:
                assert m['tier'] is None and m['donor_caption_sha256'] is None and m['eligible_donors']==0
                continue
            assert m['tier'] in range(4) and m['true_caption_sha256']!=m['donor_caption_sha256']
            assert m['eligible_donors']>0
            assert m['same_exact_media']==(m['tier']<=1)
            if m['tier']==0:assert m['same_input_segment']
            if m['arm']=='event' and m['tier']>=2:assert m['same_native_subject']
            if m['arm']=='subject' and m['tier']<=2:assert m['exact_action_signature']
        rows=math.read(folder/ds/'ROWS.json');summ=math.read(folder/ds/'SUMMARY.json')
        assert len(rows)==144 and len({r['cell'] for r in rows})==144
        assert sum(r['condition']=='clean' for r in rows)==24
        assert len({r['source_index'] for r in rows})==cfg['target_sources'][ds]
        counts=[]
        for cohort in ['event_available','subject_available','common','common_strict','same_video_event','same_video_subject']:
            counts.append(sum(len(subset(rows,f'{m}/all/{cohort}')) for m in ['clean','corrupt']))
        assert tuple(counts)==totals[ds],(ds,counts)
        for r in rows:
            assert r['candidates']==32 and r['frames']>=9
            expected={'true','generic'}|{a for a in ['event','subject'] if lookup[r['source_index'],a]['available']}
            assert set(r['available_arms'])==expected
            assert len(r['metrics'])==10*len(expected)
            for a in ['event','subject']:
                m=lookup[r['source_index'],a]
                if a in expected:
                    assert r['tiers'][a]==m['tier'] and r['pair_kinds'][a]==m['kind']
                    assert r['donor_hashes'][a]==m['donor_caption_sha256']
            for name,m in r['metrics'].items():
                arm,family,view,task,control=name.split('/')
                assert arm in expected and control=='real'
                assert m['n']==(r['frames'] if family=='frame' else 32)
                assert m['mse']>=0 and m['mae']>=0
                base=r['metrics']['true/'+name.split('/',1)[1]]
                for f in ['y','y2','n']:math.equal(m[f],base[f])
                if view=='Geometry':
                    for f in math.FIELDS:math.equal(m[f],base[f])
        for key,part in summ.items():
            rr=subset(rows,key);n=len({r['source_index'] for r in rr})
            assert part['coverage']['cells']==len(rr) and part['coverage']['sources']==n
            for a in ['event','subject']:
                assert part['coverage']['unique_donor_hashes'][a]==len({r['donor_hashes'][a] for r in rr if a in r['donor_hashes']})
            names=set.intersection(*(set(r['metrics']) for r in rr)) if rr else set()
            assert set(part['metrics'])==names
            arrays={};w=np.random.default_rng(20261003).multinomial(n,np.ones(n)/n,size=10000)/n if n else None
            for name,p in part['metrics'].items():
                ss=math.source_arrays(rr,name);a=np.array(list(ss.values()));arrays[name]=a
                assert len(ss)==p['sources']==n and p['draws']==10000 and p['seed']==20261003
                for s,v in ss.items():
                    for i,x in enumerate(v):math.equal(math.finite(x),p['source_moments'][str(s)][i])
                b=math.sample(a,w);point=math.nanmean(a)
                for f,z in p['metrics'].items():
                    vals=math.values(b,f);math.equal(math.finite(math.values(point,f)),z['mean'])
                    ci=math.interval(vals)
                    if ci is None:assert z['ci95'] is None
                    else:
                        for x,y in zip(ci,z['ci95']):math.equal(x,y)
                    assert int(np.isfinite(vals).sum())==z['bootstrap_defined']
                    assert int(np.isfinite(math.values(a,f)).sum())==z['sources_defined']
            def difference(terms,field,z):
                assert z['sources']==n
                point=0.;draw=np.zeros(10000)
                for name,weight in terms:
                    a=arrays[name];point+=weight*math.values(math.nanmean(a),field)
                    draw+=weight*math.values(math.sample(a,w),field)
                math.equal(math.finite(point),z['mean']);ci=math.interval(draw)
                if ci is None:assert z['ci95'] is None
                else:
                    for x,y in zip(ci,z['ci95']):math.equal(x,y)
                assert int(np.isfinite(draw).sum())==z['bootstrap_defined']
            for name,fields in part['paired_original_minus_intervention'].items():
                arm,tail=name.split('/',1)
                for f,z in fields.items():difference([('true/'+tail,1),(name,-1)],f,z)
            for tail,fields in part['specificity_subject_minus_event'].items():
                for f,z in fields.items():difference([('subject/'+tail,1),('event/'+tail,-1)],f,z)
            for task,z in part['Contrast_minus_Full_specificity'].items():
                difference([(f'subject/candidate/Contrast/{task}/real',1),(f'event/candidate/Contrast/{task}/real',-1),
                    (f'subject/candidate/Full/{task}/real',-1),(f'event/candidate/Full/{task}/real',1)],'r2',z)
    seal=math.read(folder/'GLOBAL_READOUT_SEAL.json');join=math.read(folder/'LABEL_JOIN.json')
    review=math.read(folder/'TEXT_CONTROL_REVIEW.json')
    assert review['time']<seal['time'] and not review['GT_time_or_metrics_read'] and review['fixed_mapping_unchanged']
    assert review['not_semantic_equivalence_certification'] and len(review['flags'])==3
    assert seal['time']<join['time'] and not seal['GT_read'] and seal['controlled_cells']==324
    assert join['only_original_cached_GT_span'] and not join['donor_GT_read'] and not join['GT_for_pairing']
    r=math.read(folder/'RESOURCES.json')
    assert r['encoder_inputs']==306 and r['backbone_offset_forwards']==612
    for k in ['expert_calls','probe_refits','backward_calls','parameter_updates','new_candidates']:assert r[k]==0
    assert not r['CUDA_initialized_in_scoring'] and r['original_and_generic_reused_cells']==288
    return dict(status='passed',checks=math.checks,maximum_numeric_error=math.maximum,
        missing_arms_not_imputed=True,strict_and_weak_cohorts_separate=True,fixed_geometry_control=True,
        independent_source_aggregation=True,independent_10000_paired_bootstrap=True,
        original_GT_join_after_global_seal=True,scope='anonymous scalar/statistic audit, not repeated GPU inference or semantic-equivalence certification')


if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2,allow_nan=False))

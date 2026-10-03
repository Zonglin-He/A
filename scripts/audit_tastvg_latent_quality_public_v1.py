"""Independent decisions, candidate metrics and bootstrap on anonymous exports."""
import sys,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tastvg_large_evidence_public_v1 import independent_summary

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(folder):
    folder=Path(folder);checks=collections.Counter();maximum=0.;eps=1e-12
    def equal(a,b):
        nonlocal maximum
        if isinstance(a,dict):
            assert set(a)==set(b),(set(a)^set(b))
            for k in a:equal(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            err=abs(float(a)-float(b));assert err<1e-9,(a,b,err)
            maximum=max(maximum,err);checks['numeric_checks']+=1
        else:assert a==b,(a,b)
    cfg=read(folder/'CONFIG.json');fit=read(folder/'SOURCE_FIT_SUMMARY.json')
    if (folder/'CODE_PINS.json').exists():
        for f,h in read(folder/'CODE_PINS.json').items():
            assert sha(ROOT/f)==h,f;checks['code_pins']+=1
    if (folder/'CALL_ACCOUNTING.json').exists():
        calls=read(folder/'CALL_ACCOUNTING.json');sources=sum(sum(v.values()) for v in cfg['source_counts'].values())
        assert calls['source_queries']==sources==190
        assert calls['source_full_model_offset_forwards']==2*sources
        assert calls['source_extra_cached_suffix_offset_calls']==4*sources
        assert calls['target_cached_suffix_offset_calls']==2*cfg['targets']['expert']
        assert calls['direct_head_identity_calls']==2*(sources+cfg['targets']['expert'])
        checks['call_accounting_fields']+=5
    assert cfg['window_seconds']==1 and cfg['source_GT'] and not cfg['target_GT_training'] and not cfg['target_GT_selection']
    assert cfg['arms']==['L8','L32','G8','G32'] and cfg['new_experts']==cfg['temporal_parameter_updates']==0
    for ds in ['vidstg','hc2']:
        for s in ['L','G']:
            x=fit[ds][s];path=x['path'];assert [v['alpha'] for v in path]==cfg['alphas']
            best=min(range(len(path)),key=lambda i:(-path[i]['validation_top1_tIoU'],path[i]['validation_MSE'],path[i]['alpha']))
            assert best==x['selected_index'] and path[best]['alpha']==x['selected_alpha']
            assert x['training_queries']==cfg['source_counts'][ds]['train'] and x['validation_queries']==cfg['source_counts'][ds]['validation']
            assert x['dimensions']==(1792 if s=='L' else 3)
            checks['source_selected_models']+=1
    if (folder/'SOURCE_ROWS.json').exists():
        source=read(folder/'SOURCE_ROWS.json');assert len(source)==190
        for ds in ['vidstg','hc2']:
            for signal in ['L','G']:
                for split,field in [('train','training_MSE'),('validation','validation_MSE')]:
                    rr=[r for r in source if r['dataset']==ds and r['split']==split]
                    assert len(rr)==cfg['source_counts'][ds][split]
                    mse=np.mean([(r['frozen_scores'][signal][i]-r['candidate_t'][i])**2 for r in rr for i in range(32)])
                    z=fit[ds][signal];equal(mse,z['path'][z['selected_index']][field])
                    if split=='validation':
                        value=np.mean([r['candidate_t'][int(np.argmax(r['frozen_scores'][signal]))] for r in rr])
                        equal(value,z['path'][z['selected_index']]['validation_top1_tIoU'])
                        equal(np.mean([r['candidate_t'][0] for r in rr]),z['validation_native'])
                        equal(np.mean([max(r['candidate_t']) for r in rr]),z['validation_oracle'])
        checks['anonymous_source_queries']=len(source)
    seal=read(folder/'GLOBAL_SCORE_SEAL.json');join=read(folder/'LABEL_JOIN.json')
    assert seal['target_GT_read'] is False and join['target_GT_used_for_fit'] is False and join['seal_time']==seal['time']<join['time']
    assert sha(folder/'SCORE_ROWS.json')==seal['score_rows_sha256']
    if (folder/'SOURCE_FIT_SEAL.json').exists():
        assert sha(folder/'SOURCE_FIT_SEAL.json')==seal['source_fit_barrier_sha256']
        assert read(folder/'SOURCE_FIT_SEAL.json')['time']<seal['time']
    evidence={r['cell_key']:r for r in read(folder/'SCORE_ROWS.json')};assert len(evidence)==288
    def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])
    derived={}
    for k,e in evidence.items():
        assert not e['GT_read'] and e['source_A_temporal_bitwise_parity'] and e['A_spatial_bitwise_parity']
        a=e['anchor_index'];assert 0<=a<8 and len(e['candidate_indices'])==len(e['intervals'])==32
        for signal in ['L','G']:
            for n in [8,32]:
                scores=e['scores'][signal][:n];mx=max(scores);top=[i for i,x in enumerate(scores) if mx-x<=eps]
                selected=top[0] if len(top)==1 and mx>scores[a]+eps else a
                assert selected==e['choices'][signal+str(n)];checks['decisions']+=1
        geos=[]
        for x in e['intervals']:
            ds,de=np.array(x)-e['intervals'][a];distance=abs(ds)+abs(de)
            balance=2*min(abs(ds),abs(de))/distance if distance>eps else 0.
            if distance<=eps:kind='same'
            elif balance<=.25+eps:
                if abs(ds)>=abs(de):kind='trim_start' if ds>0 else 'expand'
                else:kind='trim_end' if de<0 else 'expand'
            elif ds<0<de:kind='expand'
            elif de<0<ds:kind='trim_both'
            else:kind='shift'
            radius=distance/(e['intervals'][a][1]-e['intervals'][a][0]);geos.append(dict(kind=kind,large=radius>=.5-eps))
        derived[k]=geos
    groups=['all_changed','large','large_trim_start','large_trim_end','large_expand','large_shift','large_trim_both']
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            base=folder/split/ds;rows=read(base/'ROWS.json');binary=read(base/'BINARY_ROWS.json');expected=[]
            for r in rows:
                if r['expert_scheduled']:
                    e=evidence[key(r)];a=e['anchor_index'];v=r['candidate_v'];t=r['candidate_t'];geo=derived[key(r)]
                    assert r['A_state_pre_sha256']==e['A_state_pre_sha256'] and r['A_state_post_sha256']==e['A_state_post_sha256']
                    equal(r['A8_v'],v[a]);equal(r['A8_t'],t[a]);checks['expert_arrivals']+=1
                    for signal in ['L','G']:
                        delta=[x-e['scores'][signal][a] for x in e['scores'][signal]]
                        for group in groups:
                            ids=[i for i,g in enumerate(geo) if i!=a and g['kind']!='same' and (group=='all_changed' or
                                (g['large'] and (group=='large' or group=='large_'+g['kind']))) and abs(v[i]-v[a])>eps]
                            if not ids:continue
                            pos=[i for i in ids if v[i]>v[a]+eps];neg=[i for i in ids if v[i]<v[a]-eps]
                            tp=sum(delta[i]>eps for i in pos);fp=sum(delta[i]>eps for i in neg);fn=len(pos)-tp;tn=len(neg)-fp;den=len(ids)
                            auc=None
                            if pos and neg:
                                comparisons=[1 if delta[i]-delta[j]>eps else 0 if delta[i]-delta[j]< -eps else .5 for i in pos for j in neg]
                                auc=float(np.mean(comparisons))
                            expected.append(dict(**{k:r[k] for k in ['dataset','split','source_id','condition','order','arrival']},signal=signal,group=group,
                                tp=tp/den,fp=fp/den,fn=fn/den,tn=tn/den,positive=len(pos)/den,negative=len(neg)/den,accepted=(tp+fp)/den,
                                auc=auc,candidates=den,positives=len(pos),negatives=len(neg),TP=tp,FP=fp,FN=fn,TN=tn))
                for arm in ['L8','L32','G8','G32']:
                    n=int(arm[1:]);i=e['choices'][arm] if r['expert_scheduled'] else None
                    vv=v[i] if i is not None else r['A8_v'];tt=t[i] if i is not None else r['A8_t']
                    ov=max(v[:n]) if i is not None else vv;ot=max(t[:n]) if i is not None else tt;d=vv-r['A8_v']
                    for f,value in [('v',vv),('t',tt),('gain',d),('t_gain',tt-r['A8_t']),('regret',ov-vv),('t_regret',ot-tt),
                        ('gross_gain',max(d,0)),('gross_loss',max(-d,0)),('severe',float(d<-.05)),('changed',float(i is not None and i!=a)),
                        ('large_selected',float(i is not None and geo[i]['large']))]:equal(r[arm+'_'+f],value)
                    # Each scalar comparator reconstructs a common N/U/S result across both probes.
                    for signal in ['N','U','S']:
                        for met in ['v','t']:
                            legacy=r[arm+'_'+met]-r[arm+'_vs_'+signal+str(n)+'_'+met]
                            other=('G' if arm.startswith('L') else 'L')+str(n)
                            equal(legacy,r[other+'_'+met]-r[other+'_vs_'+signal+str(n)+'_'+met])
                for met in ['v','t']:equal(r['L32_vs_G32_'+met],r['L32_'+met]-r['G32_'+met])
                checks['arrivals']+=1
            equal(binary,expected)
            summary=read(base/'SUMMARY.json');dis=read(base/'DISCRIMINATION.json')
            for cat in ['corruption','clean']:
                for subset in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']=='clean')==(cat=='clean') and (subset=='all' or r['expert_scheduled']==(subset=='expert'))]
                    z=summary[cat][subset];calc=independent_summary(rr,list(z['metrics']));equal({k:z[k] for k in calc},calc)
                    transitions={arm:dict(improved=sum(r[arm+'_gain']>eps for r in rr),harmed=sum(r[arm+'_gain']< -eps for r in rr),
                        severe_gt5pp=sum(r[arm+'_gain']<-.05 for r in rr),large_selected=sum(r[arm+'_large_selected'] for r in rr)) for arm in ['L8','L32','G8','G32']}
                    equal(transitions,z['transitions'])
                for group in groups:
                    for signal in ['L','G']:
                        rr=[r for r in expected if r['group']==group and r['signal']==signal and (r['condition']=='clean')==(cat=='clean')]
                        calc=independent_summary(rr,['tp','fp','fn','tn','positive','negative','accepted'],
                            dict(precision=('tp','accepted'),benefit_recall=('tp','positive'),harm_acceptance=('fp','negative')))
                        valid=[r for r in rr if r['auc'] is not None];calc['auc']=independent_summary(valid,['auc'])
                        calc['raw_counts']={k:sum(r[k] for r in rr) for k in ['candidates','positives','negatives','TP','FP','FN','TN']}
                        calc['eligible_auc_cells']=len(valid);equal(calc,dis[cat][group][signal])
            cases=read(base/'CASES.json');expert=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
            for arm in ['L8','L32','G8','G32']:
                equal(cases[arm],dict(positive=sorted(expert,key=lambda r:-r[arm+'_gain'])[:3],negative=sorted(expert,key=lambda r:r[arm+'_gain'])[:3]))
            if (folder/'TOP1_DIAGNOSTICS.json').exists():
                extra=read(folder/'TOP1_DIAGNOSTICS.json')
                for cat in ['corruption','clean']:
                    rr=[r for r in rows if r['expert_scheduled'] and (r['condition']=='clean')==(cat=='clean')]
                    for arm in ['L8','L32','G8','G32']:
                        count=collections.Counter();by_source=collections.defaultdict(collections.Counter)
                        for r in rr:
                            e=evidence[key(r)];i=e['choices'][arm];a=e['anchor_index'];g=derived[key(r)][i];d=r[arm+'_gain']
                            count['arrivals']+=1;count['added_candidate_selected']+=i>=8;count['changed']+=i!=a
                            count['large_selected']+=g['large'];count['large_improved']+=g['large'] and d>eps
                            count['large_harmed']+=g['large'] and d< -eps;count['small_improved']+=(not g['large']) and d>eps
                            count['small_harmed']+=(not g['large']) and d< -eps;count['old_fast_gain_destroyed']+=r[arm+'_destroyed_old_fast']
                            count['operation_'+g['kind']]+=1
                            for q in [.3,.5]:
                                count[f'v{q}_correct_destroyed']+=r['A8_v']>q>=r[arm+'_v']
                                count[f'v{q}_correct_rescued']+=r[arm+'_v']>q>=r['A8_v']
                            by_source[str(r['source_id'])]['improved']+=d>eps;by_source[str(r['source_id'])]['harmed']+=d< -eps
                        equal(extra['/'.join([split,ds,cat])][arm],dict(counts={k:int(v) for k,v in count.items()},raw_source_counts={s:{k:int(v) for k,v in c.items()} for s,c in by_source.items()}))
            if (folder/'PAIRWISE_DIAGNOSTICS.json').exists():
                ex=read(folder/'PAIRWISE_DIAGNOSTICS.json')['panels']
                scalar=read(folder.parents[1]/'tastvg_large_correction_evidence'/'2026-10-03'/'EVIDENCE_ROWS.json')
                oldscores={r['cell_key']:r['scores'] for r in scalar}
                for cat in ['corruption','clean']:
                    rr=[r for r in rows if r['expert_scheduled'] and (r['condition']=='clean')==(cat=='clean')]
                    for signal in ['L','G','N','U','S']:
                        for n in [8,32]:
                            tmp=[]
                            for r in rr:
                                scores=evidence[key(r)]['scores'].get(signal,oldscores[key(r)].get(signal));z={k:r[k] for k in ['source_id','condition','order','arrival']}
                                for name,vals in [('vIoU',r['candidate_v']),('tIoU',r['candidate_t'])]:
                                    credit=[]
                                    for i in range(n):
                                        for j in range(i+1,n):
                                            if abs(vals[i]-vals[j])<=eps:continue
                                            d=scores[i]-scores[j]
                                            credit.append(.5 if abs(d)<=eps else float(d*(vals[i]-vals[j])>0))
                                    if credit:z[name+'_ordering']=float(np.mean(credit))
                                if 'vIoU_ordering' in z and 'tIoU_ordering' in z:tmp.append(z)
                            equal(ex['/'.join([split,ds,cat])][signal+str(n)],independent_summary(tmp,['vIoU_ordering','tIoU_ordering']))
            checks['panels']+=1
    assert checks['arrivals']==1152 and checks['expert_arrivals']==288 and checks['decisions']==1152
    result=dict(status='pass',checks=dict(checks),max_numeric_error=maximum,private_assets_required=False,
        learned_score_forward_reproduction='Private root readback; public audit verifies exported scores/decisions and labels/aggregation, not the excluded learned weights or latent extraction.',
        new_model_calls=0,new_expert_calls=0)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':audit(sys.argv[1])

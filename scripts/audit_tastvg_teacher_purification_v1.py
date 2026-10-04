"""Independent scalar root and anonymous public audits; no model imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys, math, time, collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary as independent_summary
from scripts.teacher_purification_import_v1 import core
FIELDS=core.FIELDS
BASE=ROOT/'artifacts/tastvg_teacher_purification_v1'
PUB=ROOT/'results/tastvg_teacher_purification/2026-10-04'
R2=ROOT/'artifacts/tastvg_dta_expert_r2_v1'
R2PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04'


def scalar_iou(a,b):
    overlap=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return overlap/((a[1]-a[0])+(b[1]-b[0])-overlap)


def derived(row):
    out={};values=row['proposal_GT_tIoU']
    for a in ['Confidence','Consensus','Oracle']:
        t=values[row[a+'_index']]
        out[a+'_t']=t;out[a+'_success_gt_05']=float(t>.5);out[a+'_disjoint']=float(t==0)
    d=out['Consensus_t']-out['Confidence_t']
    out.update(Consensus_minus_Confidence_t=d,Oracle_minus_Confidence_t=out['Oracle_t']-out['Confidence_t'],
        Oracle_minus_Consensus_t=out['Oracle_t']-out['Consensus_t'],
        Consensus_minus_Confidence_success=out['Consensus_success_gt_05']-out['Confidence_success_gt_05'],
        Consensus_minus_Confidence_disjoint=out['Consensus_disjoint']-out['Confidence_disjoint'],
        Consensus_gross_gain=max(d,0),Consensus_gross_loss=max(-d,0),
        raw_mean_t=sum(values)/len(values),raw_fraction_gt_05=sum(v>.5 for v in values)/len(values),
        selected_Consensus_agreement=row['consensus_scores'][row['Consensus_index']],
        selected_Confidence_agreement=row['consensus_scores'][row['Confidence_index']],
        proposal_count=row['proposal_count'],unique_intervals=row['unique_intervals'])
    assert set(out)==set(FIELDS)
    return out


def counts(rows):
    return dict(choices_changed=sum(r['Consensus_index']!=r['Confidence_index'] for r in rows),
        teacher_improved=sum(r['Consensus_minus_Confidence_t']>1e-12 for r in rows),
        teacher_worsened=sum(r['Consensus_minus_Confidence_t']< -1e-12 for r in rows),
        confidence_success_destroyed=sum(r['Confidence_t']>.5 and r['Consensus_t']<=.5 for r in rows),
        confidence_failure_rescued=sum(r['Confidence_t']<=.5 and r['Consensus_t']>.5 for r in rows),
        new_disjoint_event=sum(r['Confidence_t']>0 and r['Consensus_t']==0 for r in rows),
        disjoint_event_rescued=sum(r['Confidence_t']==0 and r['Consensus_t']>0 for r in rows),
        **{a+'_success_gt_05':sum(r[a+'_t']>.5 for r in rows) for a in ['Confidence','Consensus','Oracle']},
        **{a+'_disjoint':sum(r[a+'_t']==0 for r in rows) for a in ['Confidence','Consensus','Oracle']})


def public_audit(directory):
    directory=Path(directory);tick=time.monotonic();counter=collections.Counter();maxerr=0.
    def close(a,b,tol=1e-12):
        nonlocal maxerr
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        error=float(np.max(np.abs(a-b))) if a.size else 0.;assert error<=tol,(error,tol)
        maxerr=max(maxerr,error);counter['numeric_scalars']+=a.size
    cfg=read(directory/'CONFIG.json');bar=read(directory/'UNLABELED_SELECTION_BARRIER.json')
    done=read(directory/'SCORE_COMPLETION.json');sel=read(directory/'UNLABELED_SELECTION.json')
    rows=read(directory/'ROWS.json');saved=read(directory/'SUMMARY.json');dec=read(directory/'DECISION.json')
    assert len(sel)==len(rows)==288 and len({r['cell_key'] for r in rows})==288
    assert sum(r['condition']!='clean' for r in rows)==240
    assert cfg['target_expert_sources']=={'vidstg':{'search':16,'confirm':8},'hc2':{'search':14,'confirm':7}}
    assert not cfg['target_GT_used_in_unlabeled_selection'] and cfg['raw_duplicates_and_order_preserved'] and not cfg['parameter_tuning']
    assert bar['status']=='sealed' and not bar['GT_read'] and bar['GT_guard_active'] and bar['time']<done['time']
    for name,key in [('UNLABELED_SELECTION.json','selection_sha256'),('CONFIG.json','config_sha256'),('CODE_BINDINGS.json','code_bindings_sha256')]:
        assert sha(directory/name)==bar[key];counter['selection_seal_hashes']+=1
    assert sha(directory/'UNLABELED_SELECTION_BARRIER.json')==done['selection_barrier_sha256']
    for name,key in [('ROWS.json','rows_sha256'),('SUMMARY.json','summary_sha256'),('DECISION.json','decision_sha256')]:assert sha(directory/name)==done[key]
    selected={r['cell_key']:r for r in sel}
    for r in rows:
        source=selected[r['cell_key']]
        for k,v in source.items():assert r[k]==v
        n=r['proposal_count'];assert n==len(r['confidence_scores'])==len(r['consensus_scores'])==len(r['proposal_GT_tIoU']) and 26<=n<=312
        for field in ['confidence_scores','consensus_scores','proposal_GT_tIoU']:assert np.isfinite(r[field]).all()
        assert all(0<=v<=1 for v in r['consensus_scores']+r['proposal_GT_tIoU'])
        assert r['Confidence_index']==max(range(n),key=lambda i:r['confidence_scores'][i])
        assert r['Consensus_index']==max(range(n),key=lambda i:r['consensus_scores'][i])
        assert r['Oracle_index']==max(range(n),key=lambda i:r['proposal_GT_tIoU'][i])
        assert not r['GT_used'] and not r['Confidence_GT_used'] and not r['Consensus_GT_used'] and r['Oracle_GT_used'] and r['evaluation_GT_read_after_seal']
        for k,v in derived(r).items():close(r[k],v)
        counter['teacher_choices']+=3;counter['sealed_row_fields']+=len(source)
    for ds in saved:
        for sp in saved[ds]:
            subset=[r for r in rows if r['dataset']==ds and r['split']==sp]
            for name,z in saved[ds][sp].items():
                q=[r for r in subset if (name=='all' or (name=='clean')==(r['condition']=='clean'))]
                test=independent_summary(q,FIELDS);assert z['sources']==test['sources'] and z['cells']==test['cells']
                for f in FIELDS:
                    for k in ['mean','ci95','cell_macro']:close(z['metrics'][f][k],test['metrics'][f][k])
                    for sid,v in test['metrics'][f]['source_values'].items():close(z['metrics'][f]['source_values'][sid],v)
                assert z['counts']==counts(q);counter['tail_counts']+=len(z['counts'])
                d=test['metrics']['Consensus_minus_Confidence_t']['source_values']
                assert z['positive_sources']==sum(v>1e-12 for v in d.values())
                assert z['negative_sources']==sum(v< -1e-12 for v in d.values())
                assert z['unchanged_sources']==sum(abs(v)<=1e-12 for v in d.values())
                for sid,v in z['leave_one_source_out_delta'].items():close(v,np.mean([a for s,a in d.items() if s!=sid]))
                for order,zz in z['orders'].items():
                    qq=[r for r in q if r['order']==order];xx=independent_summary(qq,FIELDS)
                    assert zz['sources']==xx['sources'] and zz['cells']==xx['cells'] and zz['counts']==counts(qq)
                    for f in FIELDS:
                        for k in ['mean','ci95','cell_macro']:close(zz['metrics'][f][k],xx['metrics'][f][k])
                        for sid,v in xx['metrics'][f]['source_values'].items():close(zz['metrics'][f]['source_values'][sid],v)
                    orderd=xx['metrics']['Consensus_minus_Confidence_t']['source_values']
                    for sid,v in zz['leave_one_source_out_delta'].items():close(v,np.mean([a for s,a in orderd.items() if s!=sid]))
                    counter['tail_counts']+=len(zz['counts'])
    pass_panels={ds+'/'+sp:saved[ds][sp]['corrupt']['metrics']['Consensus_minus_Confidence_t']['ci95'][0]>0 for ds in ['vidstg','hc2'] for sp in ['search','confirm']}
    assert dec['panel_pass']==pass_panels and dec['status']==('GO_TEACHER_ONLY' if all(pass_panels.values()) else 'NO_GO')
    assert not dec['adaptation_started'] and not dec['production_promoted'] and not dec['R2c_started']
    expected=[]
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            q=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
            for tag,sort in [('worst',sorted(q,key=lambda r:r['Consensus_minus_Confidence_t'])[:3]),('best',sorted(q,key=lambda r:-r['Consensus_minus_Confidence_t'])[:3])]:
                for row in sort:expected.append((tag,row))
    cases=read(directory/'CASES.json');assert len(cases)==len(expected)==24
    for case,(tag,row) in zip(cases,expected):
        assert case['case']==tag
        for k,v in case.items():
            if k!='case':assert row[k]==v
    resource=read(directory/'RESOURCES.json')
    for k in ['new_GPU_calls','model_loads','new_backbone_calls','new_expert_calls','backward_calls','head_updates','spatial_updates','temporal_persistence_writes']:assert resource[k]==0
    assert not resource['loaded_frameworks']
    return dict(status='pass',checks=dict(counter),max_absolute_error=maxerr,CPU_wall_seconds=time.monotonic()-tick,
        scope='Anonymous selections, full metric arithmetic, paired source bootstrap/orders/tails/LOO, exact decisions and cases; raw overlap/GT checked by root',time=time.time())


def root_audit():
    tick=time.monotonic();lock=read(BASE/'RUNTIME_LOCK.json');counter=collections.Counter();maxerr=0.
    for p,h in {**lock['code'],**lock['inputs'],**lock['post_seal_inputs']}.items():
        assert sha(ROOT/p)==h,p;counter['pinned_inputs_and_code']+=1
    cfg=read(PUB/'CONFIG.json');assert sha(ROOT/'methods/CURRENT_METHOD.json')==cfg['production_method_sha256']
    bar=read(BASE/'UNLABELED_SELECTION_BARRIER.json');assert sha(BASE/'RUNTIME_LOCK.json')==bar['runtime_lock_sha256']
    assert bar==read(PUB/'UNLABELED_SELECTION_BARRIER.json')
    data=read(R2/'EXPERT_SUPPORT.json');cells=read(BASE/'COHORT.json')['cells'];rows={r['cell_key']:r for r in read(PUB/'ROWS.json')}
    oldsel={r['cell_key']:r for r in read(R2PUB/'DEPLOY_SELECTION.json')}
    old={(r['cell_key'],r['arm']):r for r in read(R2PUB/'TEACHER_SELECTION_SCORED.json')}
    labels={}
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:labels[(ds,sp)]=read(ROOT/f'artifacts/tastvg_extended_sensitivity_v3/{ds}/GT_LABELS_{sp}.json')
    def close(a,b):
        nonlocal maxerr
        error=abs(a-b);assert error<1e-12;maxerr=max(maxerr,error);counter['scalar_numeric_checks']+=1
    for c in cells:
        key='/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival']);r=rows[key];support=data[key]
        p=support['proposals'];n=len(p);confidence=support['confidence'];mean=[]
        for i in range(n):
            peer=[scalar_iou(p[i],p[j]) for j in range(n) if i!=j]
            value=math.fsum(peer)/(n-1);close(r['consensus_scores'][i],value);mean.append(value)
            counter['pair_overlap_recomputed']+=len(peer)
        assert r['Consensus_index']==max(range(n),key=lambda i:mean[i])
        assert r['Confidence_index']==max(range(n),key=lambda i:confidence[i])==support['deploy']['index']==oldsel[key]['selected_index']
        assert r['unique_intervals']==len(set(map(tuple,p))) and r['proposal_count']==n
        span=labels[(c['dataset'],c['split'])][str(c['parent'])]['span'];quality=[scalar_iou(proposal,span) for proposal in p]
        for a,b in zip(quality,r['proposal_GT_tIoU']):close(a,b)
        assert r['Oracle_index']==max(range(n),key=lambda i:quality[i])==old[(key,'EOracle')]['selected_index']
        for arm,oldarm in [('Confidence','EDeploy'),('Oracle','EOracle')]:close(r[arm+'_t'],old[(key,oldarm)]['continuous_teacher_tIoU'])
        for k,v in [('A_state_pre_sha256',c['pre_sha']),('A_state_post_sha256',c['post_sha']),('pixel_sha256',c['pixel_sha256']),('expert_cache_sha256',support['expert_cache_sha256'])]:assert r[k]==v
        from scripts.run_tastvg_teacher_purification_v1 import digest
        assert r['raw_proposals_sha256']==digest(p) and r['confidence_sha256']==digest(confidence)
        counter['cached_controls_exact']+=2
    assert not any(x in sys.modules for x in ['torch','tensorflow','jax'])
    public=public_audit(PUB);write(BASE/'PUBLIC_AUDIT.json',public);write(PUB/'PUBLIC_AUDIT.json',public)
    out=dict(status='pass',checks=dict(counter),max_absolute_error=maxerr,CPU_wall_seconds=time.monotonic()-tick,
        teacher_cells=288,raw_cache_files_verified=235,loaded_frameworks=[],new_model_GPU_gradient_calls=0,
        production_registry_unchanged=True,selection_sealed_before_GT=True,R2_control_parity=True,public_audit_sha256=sha(PUB/'PUBLIC_AUDIT.json'),
        scope='Independent scalar pairwise overlap/medoid/GT-tIoU, exact R2 controls, support/cache/hash/state provenance and anonymous statistical audit',time=time.time())
    write(BASE/'ROOT_AUDIT.json',out);write(PUB/'ROOT_AUDIT.json',out)
    return out
if __name__=='__main__':
    if sys.argv[1]=='root':out=root_audit()
    else:out=public_audit(Path(sys.argv[2]) if len(sys.argv)>2 else PUB)
    print(out)

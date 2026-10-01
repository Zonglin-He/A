"""Root readback of sealed quick results, previous Frozen parity and anonymous export."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections, csv, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_best_quick_common_v1 import *

def mean(values):
    values=list(values)
    return math.fsum(values)/len(values)

def check_summary(rows,summary):
    if not rows:
        assert summary['sources']==0 and summary['cells']==0
        return 0
    assert summary['cells']==len(rows)
    checks=0
    for field,actual in summary['metrics'].items():
        grouped=collections.defaultdict(list)
        for r in rows:
            grouped[r['source_id'],r['order'],r['condition']].append(r[field])
        so=collections.defaultdict(list)
        for (s,o,c),v in grouped.items():
            so[s,o].append(mean(v))
        bysource=collections.defaultdict(list);byorder=collections.defaultdict(list)
        for (s,o),v in so.items():
            m=mean(v);bysource[s].append(m);byorder[o].append(m)
        values=[mean(bysource[s]) for s in sorted(bysource)]
        assert summary['sources']==len(values)
        assert abs(actual['mean']-mean(values))<1e-12
        assert abs(actual['query_macro']-mean(r[field] for r in rows))<1e-12
        np.testing.assert_allclose([mean(v) for v in byorder.values()],actual['order_values'],atol=1e-12,rtol=0)
        if len(byorder)==1:
            assert actual['order_sample_SD'] is None
        # Independent source-bootstrap for the primary comparison and its baseline.
        if field in ['delta_m_vIoU','Frozen_m_vIoU','Ours_m_vIoU']:
            a=np.asarray(values);rng=np.random.default_rng(20261001);boot=[]
            for _ in range(100):
                boot.extend(a[rng.integers(0,len(a),(100,len(a)))].mean(1))
            np.testing.assert_allclose(np.quantile(boot,[.025,.975]),actual['ci95'],atol=1e-12,rtol=0)
        checks+=3
    return checks

def run(dataset):
    import torch
    torch.set_num_threads(2)
    from methods.decota_final_simplified_v1.tensors import state_hash
    p=verify(dataset);out=BASE/dataset
    globalbar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert globalbar['cells']==8808
    for d in DATASETS:
        assert sha(BASE/d/'PREDICTION_BARRIER.json')==globalbar['datasets'][d]
    completion=read(out/'COMPLETION.json');audit=read(out/'AUDIT.json');bar=read(out/'PREDICTION_BARRIER.json')
    assert completion['audit_sha256']==sha(out/'AUDIT.json') and completion['rows_sha256']==sha(out/'ROWS.json')
    assert audit['status']=='pass' and not audit['GPU_initialized']
    assert bar['time']<globalbar['time']<read(out/'GT_EXPOSURE.json')['time']<=completion['time']
    rows=read(out/'ROWS.json');steps=read(out/'SPATIAL_STEP_ROWS.json');summary=read(out/'SUMMARY.json')
    assert len(rows)==p['total']==bar['cells']==audit['cells']==len(bar['files'])
    assert len(steps)==audit['diagnostic_steps']
    old=read(ROOT/'artifacts/tastvg_paper48_v1'/p['matched_previous_panel']/'ROWS.json')
    previous={(r['parent'],r['order'],r['condition']):r for r in old}
    assert len(previous)==len(rows)
    checks=links=parameter_checks=0;compute=collections.Counter();max_frozen_error=0.
    cfg=p['params'];initial=read(out/'SUPPORT.json')['center_sha256']
    for cond in p['conditions']:
        for order,seq in p['orders'].items():
            prior=initial
            for at,parent in enumerate(seq):
                rf=out/'online'/cond/order/f'{at:05}.pt.json'
                assert sha(rf)==bar['files'][str(rf.relative_to(out))]
                rec=read(rf);payload=rf.with_suffix('.gz');assert sha(payload)==rec['sha256']
                x=loadz(payload)
                assert x['parent']==parent and x['arrival']==at and x['expert_scheduled']==(at%4==0)
                assert prior==x['pre_sha']==state_hash(x['pre_state']) and rec['pre_sha']==prior
                state=x['pre_state']
                for step in x['update_steps']:
                    assert state_hash(state)==step['pre_state_sha256']==state_hash(step['pre_state'])
                    u=step['update']
                    if u:
                        assert u['lr']==cfg['lr'] and u['teacher_temperature']==cfg['teacher_temperature'] and u['student_temperature']==cfg['student_temperature']
                        assert u['candidate_targets_detached'] and u['reward_detached']
                        for n,v in step['pre_state'].items():
                            expected=(v.double().numpy()-u['update_scale']*u['gradients'][n].double().numpy()).astype(np.float32)
                            np.testing.assert_allclose(expected,step['post_state'][n],atol=1e-6,rtol=2e-6)
                            parameter_checks+=v.numel()
                    else:
                        assert all(torch.equal(v,step['post_state'][n]) for n,v in step['pre_state'].items())
                    state=step['post_state'];assert state_hash(state)==step['post_state_sha256']
                assert state_hash(state)==x['post_sha']==rec['post_sha']
                prior=x['post_sha'];compute.update(x['compute']);links+=1
    assert links==p['total'] and dict(compute)==audit['compute']
    paired=[]
    for r in rows:
        q=previous[r['parent'],r['order'],r['condition']]
        assert r['arrival']==q['arrival'] and r['expert_scheduled']==q['expert_scheduled']
        for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:
            error=abs(r['Frozen_'+k]-q['Frozen_'+k]);max_frozen_error=max(max_frozen_error,error)
            assert error<1e-10,(dataset,k,error)
            assert 0<=r['Frozen_'+k]<=1+1e-12 and 0<=r['Ours_'+k]<=1+1e-12
            assert abs(r['delta_'+k]-(r['Ours_'+k]-r['Frozen_'+k]))<1e-12;checks+=3
        assert abs(r['delta_total']-r['delta_inherited_boxes']-r['delta_inherited_interval']-r['delta_temporal_rerank'])<1e-12
        if not r['expert_scheduled']:
            assert not r['updated'] and r['delta_temporal_rerank']==0
        paired.append(dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],delta_vs_previous=r['Ours_m_vIoU']-q['Ours_m_vIoU']))
    for cohort,groups in summary.items():
        rr=[r for r in rows if cohort=='full' or r['tuning_source_exposed']==(cohort=='tuning_sources')]
        for group,subs in groups.items():
            seq=[r for r in rr if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
            for sub,z in subs.items():
                checks+=check_summary([r for r in seq if sub=='all' or r['expert_scheduled']==(sub=='expert')],z)
    # Diagnosis tables are scalar consequences of sealed observations, never new predictions.
    for s in steps:
        assert abs(s['delta_update']-s['post_v']+s['pre_v'])<1e-12
        assert abs(s['oracle_v']-max(s['candidate_v']))<1e-12
        if not s['updated']:
            assert abs(s['delta_update'])<1e-10
        checks+=3
    assert not torch.cuda.is_initialized()
    result=dict(status='pass',dataset=dataset,cells=links,scalar_checks=checks,
                independent_parameter_scalar_checks=parameter_checks,
                max_previous_Frozen_error=max_frozen_error,
                sealed_before_GT=True,GPU_initialized=False,
                compute=dict(compute),time=time.time())
    write(out/'ROOT_READBACK.json',result)
    export=out/'public_export';export.mkdir(exist_ok=True)
    for name in ['COHORT.json','SUMMARY.json','PIPELINE_DIAGNOSIS.json','CASES.json','AUDIT.json',
                 'ROOT_READBACK.json','ROWS.json','SPATIAL_STEP_ROWS.json']:
        (export/name).write_bytes((out/name).read_bytes())
    write(export/'PAIRED_VS_PREVIOUS_ROWS.json',paired)
    with (export/'SCALARS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        for r in rows:
            writer.writerow({k:__import__('json').dumps(v,separators=(',',':')) if isinstance(v,list) else v for k,v in r.items()})
    manifest={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in export.iterdir() if f.is_file() and f.name!='MANIFEST.json'}
    write(export/'MANIFEST.json',manifest)
    print(result,flush=True)

if __name__=='__main__':
    run(sys.argv[1])

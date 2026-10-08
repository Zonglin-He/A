"""Join actual new OPD and previously sealed/scored baselines by original query/order."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

def aggregate(rows):
    fields=['After_v','After_t','After_s','After_R30','After_R50','delta_total_v','delta_current_v','delta_inherited_v']
    queries=collections.defaultdict(list)
    for r in rows:queries[(r['source_id'],r['query_ordinal'])].append(r)
    qvalues=np.array([[np.mean([r[f] for r in rr]) for f in fields] for rr in queries.values()])
    byparent=collections.defaultdict(list)
    for (s,q),value in zip(queries,qvalues):byparent[s].append(value)
    matrix=np.array([np.mean(byparent[s],axis=0) for s in sorted(byparent)])
    rng=np.random.default_rng(20261008)
    boot=np.concatenate([matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1) for _ in range(100)])
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(queries=len(queries),parent_sources=len(matrix),logical_arrivals=len(rows),bootstrap_resamples=10000,
        metrics={f:dict(source_macro=float(matrix[:,j].mean()),official_query_macro=float(qvalues[:,j].mean()),
            paired_parent_ci95=ci[:,j].tolist(),harm_gt5pp_parent_sources=int((matrix[:,j]<-.05).sum()) if f.startswith('delta_') else None,
            harm_gt20pp_parent_sources=int((matrix[:,j]<-.2).sum()) if f.startswith('delta_') else None) for j,f in enumerate(fields)})

def read_rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]

def run():
    assert read(BASE/'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
    binding=read(BASE/'P1_BASELINE_BINDING.json')
    for f,h in binding['files'].items():assert sha(ROOT/f)==h
    results={};all_rows=[];checks=0
    oldpub=ROOT/'results/decota_paper_baseline_scoring/2026-10-08'
    for ds in DATASETS:
        completion=read(BASE/'stages'/('P1_'+ds)/'CPU_COMPLETION.json')
        for f,h in completion['files'].items():assert sha(ROOT/f)==h
        opd=read_rows(PUB/('P1_'+ds)/'ROWS.jsonl.gz')
        plan=read(PAPER/ds/'PLAN.json');n=len(plan['rows']);assert len(opd)==n*3
        sources={s:i for i,s in enumerate(sorted({r['source'] for r in plan['rows']}))}
        dev_ids={sources[plan['rows'][q]['source']] for q in read(BASE/'DESIGN_LOCK.json')['datasets'][ds]['development_query_ordinals']};assert len(dev_ids)==32
        frozen={(r['query_ordinal'],r['order']):r for r in opd}
        assert len(frozen)==n*3
        methods={'spatial_opd':opd}
        for method in ['source_only','tent_stvg','sar_stvg','dino_refine','target_trained_reference']+(['eata_stvg'] if ds=='hc2' else []):
            rr=read_rows(oldpub/ds/(method+'.jsonl.gz'));assert len(rr)==n*3
            adapted=[]
            for r in rr:
                q=r['query_id'];match=frozen[(q,r['order'])]
                assert r['source_id']==match['source_id'] and r['arrival']==match['arrival']
                for f in ['v','t','s']:assert abs(r['Frozen_'+f]-match['Frozen_'+f])<2e-10;checks+=1
                z={**r,'query_ordinal':q,'After_R30':r['After_R0.3'],'After_R50':r['After_R0.5']}
                adapted.append(z)
            methods[method]=adapted
        results[ds]={}
        for method,rr in methods.items():
            results[ds][method]=dict(all_official_queries=aggregate(rr),excluding_32_tuning_parent_sources=aggregate([r for r in rr if r['source_id'] not in dev_ids]),
                scope='supervised in-domain frozen reference' if method=='target_trained_reference' else 'task-adapted STVG TTA port' if method in ['tent_stvg','sar_stvg','eata_stvg'] else 'specialist spatial OPD' if method=='spatial_opd' else 'no parameter adaptation',
                missing_EATA_other_direction=method=='eata_stvg',source_and_query_cohort_identical=True)
            if method=='spatial_opd':
                for r in rr:all_rows.append({**r,'method':'spatial_opd','dataset':ds})
    write(PUB/'TABLE1_SUMMARY.json',dict(status='actual_postseal_paired_results',datasets=results,
        frozen_input_metric_join_checks=checks,baseline_scoring_commit='f15641a072086619efd95d05ead3c539d37f52c5',
        OPD_primary_configs_development_selected=True,not_fresh_unseen=True,EATA_vidstg='unavailable_user_paused',time=time.time()))
    lines=['# Table 1: fixed Spatial OPD and matching sealed baselines','',
        'Clean native cross-domain transfer on all official target queries and the original three orders. Parent macro averages queries and orders within each parent; official query macro averages each query over orders. The two target configurations were selected on exposed 32-parent development sets; those parents are separately excluded below. Remaining parents still have project history exposure.','',
        '| Target / method | Queries / parents | Official query m_vIoU % | vIoU > .3 % | vIoU > .5 % | Parent m_vIoU % [95% CI] | Parent m_vIoU % excluding 32 tuning parents |','|---|---:|---:|---:|---:|---|---:|']
    for ds,methods in results.items():
        for method,x in methods.items():
            a=x['all_official_queries'];m=a['metrics'];v=m['After_v'];lo,hi=np.array(v['paired_parent_ci95'])*100
            lines.append(f"| {ds} / {method} | {a['queries']} / {a['parent_sources']} | {v['official_query_macro']*100:.3f} | {m['After_R30']['official_query_macro']*100:.3f} | {m['After_R50']['official_query_macro']*100:.3f} | {v['source_macro']*100:.3f} [{lo:.3f}, {hi:.3f}] | {x['excluding_32_tuning_parent_sources']['metrics']['After_v']['source_macro']*100:.3f} |")
    lines+=['','Target-trained reference is supervised and not a TTA method or mathematical upper bound. TENT/SAR/EATA are STVG temporal/decoder ports. EATA HC2 supplemental uses the completed Vid-source Fisher; the unavailable EATA VidSTG direction remains user paused. Old IoU-energy Ours is excluded.','',
        'This assembler independently groups queries then parents and joins every baseline row to the new exact Frozen metric/query/order/arrival. Root still owes independent whole-table audit, all required plots/strata/negative tails, actual visual inspection, verified public publication and stage closing before later stages.']
    (PUB/'TABLE1_REPORT.md').write_text('\n'.join(lines)+'\n')
    write(BASE/'P1_TABLE_ASSEMBLY_COMPLETION.json',dict(status='pending_root_independent_table_audit_visual_publication',
        join_checks=checks,new_OPD_logical_arrivals=len(all_rows),complete_paper=False,time=time.time()))
    print('TABLE1_ACTUAL_ASSEMBLY',len(all_rows),checks)

if __name__=='__main__':run()

"""Publish privacy-filtered tables from completed formal runs; never infer or score."""
import argparse
import hashlib
import json
import math
from pathlib import Path

JOBS=[('tastvg','hc2'),('tastvg','vidstg'),('cgstvg','hc2'),('cgstvg','vidstg')]
ARMS=['Source','Spatial','Temporal','Full','WithoutOPD']
PRIMARY=['m_vIoU','vIoU@0.3','vIoU@0.5']
ABLATION=PRIMARY+['m_tIoU','m_sIoU','m_sIoU_unobserved_actual']
SUPPLEMENT=['tIoU_ge_05','tIoU_eq_0','boundary_MAE_seconds','m_sIoU_observed_actual']
FIELDS={'v','t','s','vIoU_at_03','vIoU_at_05','vIoU_ge_03','vIoU_ge_05','tIoU_ge_05','tIoU_eq_0',
        'boundary_MAE','boundary_MAE_seconds','fixed_actual_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    return value if isinstance(value,(int,float))and not isinstance(value,bool)and math.isfinite(value)else None


def mean(values):
    valid=[numeric(x)for x in values if numeric(x)is not None]
    return sum(valid)/len(valid)if valid else None


def safe_numbers(value):
    """Aggregate report dictionaries contain numbers only, never copied input metadata."""
    if isinstance(value,dict):return {str(k):safe_numbers(v)for k,v in value.items()if isinstance(v,(dict,list,int,float))or v is None}
    if isinstance(value,list):return [safe_numbers(v)for v in value]
    return numeric(value)


def efficiency(receipts):
    walls=[r.get('total_seconds')for r in receipts]
    adaptation=[r.get('adapt',{}).get('fit_seconds')for r in receipts]
    peaks=[];capture_peaks=[];source224=[];forwards=[]
    for r in receipts:
        capture=r.get('capture',{});worker=numeric(r.get('engineering',{}).get('worker_peak_allocated_GPU_bytes'))
        cp=numeric(capture.get('peak_GPU_bytes'));capture_peaks.append(cp)
        peaks.append(max(worker,cp)if worker is not None and cp is not None else None)
        source224.append(capture.get('Source224_seconds'))
        forwards.append(capture.get('ReferDINO_query_forwards'))
    valid_peaks=[x for x in peaks if x is not None]
    complete=len(valid_peaks)==len(receipts)
    return dict(queries=len(receipts),actual_full_query_wall_seconds_mean=mean(walls),actual_adapt_seconds_mean=mean(adaptation),
                full_wall_valid_queries=sum(numeric(v)is not None for v in walls),adapt_valid_queries=sum(numeric(v)is not None for v in adaptation),
                peak_allocated_cuda_bytes_max=max(valid_peaks)if valid_peaks and complete else None,
                per_query_peak_allocated_cuda_bytes_mean=mean(peaks)if complete else None,
                full_peak_valid_queries=len(valid_peaks),capture_only_peak_allocated_cuda_bytes_max=max([v for v in capture_peaks if v is not None],default=None),
                Source224_component_seconds_mean=mean(source224),Source_standalone_end_to_end_seconds_mean=None,
                teacher_query_forwards=sum(v for v in forwards if numeric(v)is not None),
                teacher_query_forwards_known_queries=sum(numeric(v)is not None for v in forwards),
                teacher_cache_reused_queries=sum(v==0 for v in forwards),
                interpretation='Actual resident-run query wall includes the expert work actually executed and cache hits. Source224 is a component, not a separately measured Source end-to-end baseline. Complete peak is max(worker allocated CUDA peak, teacher/capture allocated CUDA peak), not their sum; cold uncached latency is not inferred.')


def read_benchmark(output,backbone,dataset,seal):
    """Validate the actual benchmark schema; publish aggregates without private rows."""
    path=output/'benchmark.json'
    if backbone!='tastvg'or not path.is_file():return dict(status='pending')
    try:
        benchmark=json.loads(path.read_text())
        if benchmark.get('status')!='passed'or benchmark.get('backbone')!=backbone or benchmark.get('dataset')!=dataset or benchmark.get('GT_read')is not False:
            raise ValueError('benchmark_not_passed')
        if not seal.get('runtime_fingerprint')or benchmark.get('runtime_fingerprint')!=seal['runtime_fingerprint']or benchmark.get('manifest_sha256')!=seal['manifest_sha256']:
            raise ValueError('benchmark_runtime_or_manifest_binding_mismatch')
        if benchmark.get('warmup_excluded_from_means')is not True:raise ValueError('benchmark_warmup_not_excluded')
        expected_caches=dict(RGB=False,visual=False,teacher_results=False,model_weights_resident=True)
        if benchmark.get('caches')!=expected_caches:raise ValueError('benchmark_cache_scope_mismatch')
        rows=benchmark['rows'];n=benchmark['sample_parents']
        if not isinstance(n,int)or isinstance(n,bool)or not 1<=n<=16 or len(rows)!=n or len({r['id']for r in rows})!=n or len({r['parent']for r in rows})!=n:
            raise ValueError('benchmark_sample_count_mismatch')
        warmup=benchmark['warmup']
        if any(r.get('bitwise_sealed_regression')is not True or r.get('GT_read')is not False or r.get('id')not in seal['files']for r in rows+[warmup]):
            raise ValueError('benchmark_predictions_not_verified')
        # Existing schema records verification on every measured row and warmup.
        if 'predictions_verified'in benchmark and benchmark['predictions_verified']is not True:
            raise ValueError('benchmark_predictions_not_verified')
        summary={}
        for arm in ['Source','Full']:
            times=[numeric(r[arm]['wall_seconds'])for r in rows]
            peaks=[numeric(r[arm]['peak_allocated_cuda_bytes'])for r in rows]
            if any(v is None or v<=0 for v in times)or any(v is None or v<0 for v in peaks):raise ValueError('benchmark_invalid_measurement')
            computed=dict(queries=n,wall_seconds_mean=sum(times)/n,wall_seconds_min=min(times),wall_seconds_max=max(times),peak_allocated_cuda_bytes_max=max(peaks),peak_allocated_cuda_bytes_mean=sum(peaks)/n)
            saved=benchmark['summary'][arm]
            if any(numeric(saved.get(k))is None or not math.isclose(saved[k],v,rel_tol=1e-12,abs_tol=1e-12)for k,v in computed.items()):
                raise ValueError('benchmark_summary_mismatch')
            summary['SourceStandalone'if arm=='Source'else arm]=computed
        ratio=summary['Full']['wall_seconds_mean']/summary['SourceStandalone']['wall_seconds_mean']
        saved_ratio=numeric(benchmark['summary'].get('Full_to_Source_wall_ratio'))
        if saved_ratio is None or not math.isclose(saved_ratio,ratio,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('benchmark_ratio_mismatch')
        return dict(status='passed',benchmark_sha256=digest(path),sample_parents=n,predictions_verified=True,
            verified_measured_predictions=n,warmup_prediction_verified=True,summary=summary,
            Full_to_SourceStandalone_wall_ratio=ratio,model_load_seconds={a:numeric(benchmark.get('model_load_seconds',{}).get(a))for a in ['source','teacher']},
            timing_scope='Measured SourceStandalone and Full from RGB decode through final output; models resident, weight loading separate, warmup excluded, no RGB/visual/teacher-results cache. Full includes eight-frame ReferDINO, Strong and OPD.',
            memory_scope='Maximum allocated CUDA bytes of worker and teacher processes, not their sum or reserved/device memory.')
    except (OSError,ValueError,KeyError,TypeError)as error:
        return dict(status='blocked',integrity_error=str(error)if isinstance(error,ValueError)else type(error).__name__)


def read_job(root,backbone,dataset,state):
    key=backbone+'-'+dataset;progress=state.get('jobs',{}).get(key,{})
    status=progress.get('status','pending')
    result=dict(backbone=backbone,dataset=dataset,status='worker_completed_awaiting_score'if status=='completed'else status)
    if status!='scored':return result
    output=root/'outputs/formal'/backbone/dataset
    try:
        metrics_path=output/'metrics.json';seal_path=output/'sealed.json'
        if digest(metrics_path)!=progress.get('metrics_sha256')or digest(seal_path)!=progress.get('seal_sha256'):
            raise ValueError('completion_hash_mismatch')
        metrics=json.loads(metrics_path.read_text());seal=json.loads(seal_path.read_text())
        if seal.get('status')!='sealed_before_offline_scoring'or seal.get('labels_read')is not False or metrics.get('status')!='scored_sealed_predictions':raise ValueError('not_completed_sealed_and_scored')
        if metrics.get('provenance',{}).get('manifest_sha256')!=seal.get('manifest_sha256'):raise ValueError('score_cohort_binding_mismatch')
        rows=metrics['rows']
        if len(rows)!=seal['queries']or len({r['id']for r in rows})!=len(rows)or {r['id']for r in rows}!=set(seal['files'])or any(r['dataset']!=dataset for r in rows):raise ValueError('incomplete_cohort')
        receipts=[]
        for row in rows:
            entry=seal['files'][row['id']];pred=(output/entry['path']).resolve()
            if not pred.is_relative_to(output.resolve()):raise ValueError('unsafe_seal_path')
            if digest(pred)!=entry['sha256']:raise ValueError('prediction_content_hash_mismatch')
            receipt=json.loads(pred.with_suffix('.json').read_text())
            if receipt.get('prediction_sha256')!=entry['sha256']or receipt.get('resume_hash')!=seal.get('resume_hash')or receipt.get('labels_read')is not False:raise ValueError('query_receipt_binding_mismatch')
            receipts.append(receipt)
        summary=metrics['datasets'][dataset]
        if summary['queries']!=len(rows):raise ValueError('summary_count_mismatch')
        parents={p:f'Parent{i+1:05d}'for i,p in enumerate(dict.fromkeys(r['parent']for r in rows))}
        anonymous=[]
        for i,row in enumerate(rows):
            scores={a:{k:numeric(v)for k,v in row['values'][a].items()if k in FIELDS}for a in ARMS}
            changes={a+'-'+b:100*(scores[a]['v']-scores[b]['v'])for a,b in [('Full','Source'),('Full','WithoutOPD')]}
            anonymous.append(dict(query=f'Query{i+1:06d}',parent=parents[row['parent']],values=scores,change_vIoU_pp=changes))
        result.update(status='completed',queries=len(rows),parents=len(parents),
            manifest_sha256=seal['manifest_sha256'],seal_sha256=digest(seal_path),metrics_sha256=digest(metrics_path),
            query_main_table={a:{k:numeric(summary['query_main_table'][a].get(k))for k in ABLATION+SUPPLEMENT+['m_sIoU_fixed_actual']}for a in ARMS},
            query_paired=safe_numbers(summary.get('query_paired',{})),parent_paired=safe_numbers(summary.get('paired',{})),
            parent_main_table=safe_numbers(summary.get('parent_main_table',{})),
            development_strata=safe_numbers(summary.get('development_strata',{})),
            efficiency=efficiency(receipts),benchmark=read_benchmark(output,backbone,dataset,seal),signed_queries=anonymous)
    except (OSError,ValueError,KeyError,TypeError)as error:
        result.update(status='blocked',integrity_error=str(error)if isinstance(error,ValueError)else type(error).__name__)
    return result


def cell(value,seconds=False):
    return '—'if numeric(value)is None else f'{value if seconds else 100*value:.4f}'


def build_report(root):
    root=Path(root).resolve();statepath=root/'outputs/formal/state.json'
    state=json.loads(statepath.read_text())if statepath.is_file()else {}
    jobs=[read_job(root,b,d,state)for b,d in JOBS]
    public=dict(schema=1,status='completed'if all(j['status']=='completed'for j in jobs)else'partial_or_pending',
        score_units='IoU/recall fractions in JSON; percentage points in Markdown; MAE seconds; memory allocated bytes',
        aggregation='Equal-query main table; separate parent-level and parent-cluster paired estimates. Current development selection is disclosed and is not online label access.',
        jobs=jobs)
    lines=['# Formal evaluation','',
        'Only completely sealed, scored and hash-bound jobs contribute numbers. Pending, running, failed or integrity-blocked jobs remain unfilled. No new inference, scoring, GT reading, parameter selection or filtering is performed by this reporter.','',
        '## Main table','',
        '| Backbone | Target | Output | m_vIoU % | vIoU@0.3 % | vIoU@0.5 % | Status |',
        '|---|---|---|---:|---:|---:|---|']
    for job in jobs:
        for arm in ['Source','Full']:
            values=job.get('query_main_table',{}).get(arm,{})
            lines.append('| '+' | '.join([job['backbone'],job['dataset'],arm]+[cell(values.get(k))for k in PRIMARY]+[job['status']])+' |')
    lines+=['','Official vIoU@ thresholds use strict >0.3 and >0.5. m_sIoU uses the dense fixed-GT-frame spatial metric; actual feature-node metrics are separately named.','',
            '## TA-STVG five-row ablation','',
            '| Target | Output | m_vIoU % | vIoU@0.3 % | vIoU@0.5 % | m_tIoU % | m_sIoU % | Unobserved actual m_sIoU % |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for job in jobs:
        if job['backbone']!='tastvg':continue
        for arm in ARMS:
            values=job.get('query_main_table',{}).get(arm,{})
            lines.append('| '+' | '.join([job['dataset'],arm]+[cell(values.get(k))for k in ABLATION])+' |')
    lines+=['','| Target | Output | tIoU>=0.5 % | tIoU=0 % | Boundary MAE seconds | Observed actual m_sIoU % |',
            '|---|---|---:|---:|---:|---:|']
    for job in jobs:
        if job['backbone']!='tastvg':continue
        for arm in ARMS:
            values=job.get('query_main_table',{}).get(arm,{})
            lines.append('| '+' | '.join([job['dataset'],arm]+[cell(values.get(k),seconds=k=='boundary_MAE_seconds')for k in SUPPLEMENT])+' |')
    lines+=['','## Actual execution cost','',
            '| Backbone | Target | Full query wall mean s | Adaptation mean s | Complete allocated CUDA peak GiB | Standalone Source total s |',
            '|---|---|---:|---:|---:|---:|']
    for job in jobs:
        cost=job.get('efficiency',{});peak=cost.get('peak_allocated_cuda_bytes_max')
        lines.append('| '+' | '.join([job['backbone'],job['dataset'],cell(cost.get('actual_full_query_wall_seconds_mean'),True),cell(cost.get('actual_adapt_seconds_mean'),True),cell(peak/2**30 if peak is not None else None,True),'—'])+' |')
    lines+=['',
        'These are the actual resident-run costs, including the ReferDINO work actually executed; cache reuse counts are disclosed per job in results.json. They are not cold uncached latency estimates. Source224 phase time is only a component and cannot stand in for standalone Source end-to-end latency; the separate verified benchmark below is the only source of that measurement. CUDA memory is the maximum allocated peak of the student and teacher processes, not the sum; missing complete instrumentation remains pending.',
        '']
    lines+=['## Standalone Source versus Full end-to-end benchmark','',
            '| Backbone | Target | Parents | SourceStandalone mean s | Full mean s | Full / Source | Source allocated peak GiB | Full allocated peak GiB | Status |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for job in jobs:
        bench=job.get('benchmark',dict(status='pending'));summary=bench.get('summary',{})
        source=summary.get('SourceStandalone',{});full=summary.get('Full',{})
        source_peak=source.get('peak_allocated_cuda_bytes_max');full_peak=full.get('peak_allocated_cuda_bytes_max')
        lines.append('| '+' | '.join([job['backbone'],job['dataset'],str(bench['sample_parents'])if bench.get('status')=='passed'else'—',
            cell(source.get('wall_seconds_mean'),True),cell(full.get('wall_seconds_mean'),True),cell(bench.get('Full_to_SourceStandalone_wall_ratio'),True),
            cell(source_peak/2**30 if source_peak is not None else None,True),cell(full_peak/2**30 if full_peak is not None else None,True),bench['status']])+' |')
    lines+=['',
        'Benchmark numbers are published only after status, manifest/runtime binding, unique measured parents, every measured and warmup bitwise prediction check, cache scope and recomputed summaries pass. They cover at most sixteen prespecified parents, exclude warmup and separate weight-load time. Missing or invalid benchmarks remain unfilled; Source224 is never substituted.',
        '',
        'results.json retains all signed query outcomes and parent identifiers replaced by per-job anonymous ordinals. It contains no captions, original parent IDs, video paths, media, annotations, checkpoint tensors or raw caches. Parent-level paired intervals and query-weighted intervals remain separate; full-dataset results that include development parents are not labelled fully independent confirmation.','']
    return public,'\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    root=Path(__file__).resolve().parents[1];public,markdown=build_report(root)
    destination=root/'results/formal';destination.mkdir(parents=True,exist_ok=True)
    (destination/'results.json').write_text(json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (destination/'report.md').write_text(markdown)
    print(public['status'])

if __name__=='__main__':main()

"""Build anonymous A0.3 CPU/native report only after all independent audits."""
import sys,time,shutil,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,total_prior
D=OUT/'a03_shared_basis_v1';N=D/'native'


def main():
    cpu=read(D/'PUBLIC_REPORT.json');ca=read(D/'ROOT_READBACK.json');fa=read(N/'ROOT_FIELD_READBACK.json')
    r=read(N/'independent_readback_v1/REPORT.json');ra=read(N/'independent_readback_v1/ROOT_SUMMARY_CROSSCHECK.json')
    assert ca['status']==fa['status']==ra['status']=='passed' and read(N/'EXIT.json')['code']==0
    pid=read(N/'EXIT.json')['pid'];assert not Path(f'/proc/{pid}').exists(),'Worker must have exited'
    arms=('B1','oracle','Shared-R16','Shared-R32');metrics=('tIoU','sIoU','vIoU')
    sources=sorted({x['source'] for x in r['arms']['B1']['rows']});anon={s:f'P{i:02}' for i,s in enumerate(sources,1)}
    public=dict(status='completed_independently_audited',CPU_shared_basis={**cpu,'status':'completed_independently_audited'},
        native_scope=r['scope'],native_arms={},comparisons={},versus_full={},
        query_tails=r['query_tails'],B1_good_retention=r['B1_good_retention'],gates=r['gates'],
        oracle_gain_retention=r['oracle_gain_retention'],decision=r['decision'],
        numeric_audits=dict(CPU=ca,native_fields=fa,native_summary={k:v for k,v in ra.items() if k!='cases'},geometry_max_abs=r['scalar_tensor_max_abs'],reused_metric_max_abs=r['reused_score_max_abs']))
    for arm in arms:
        summary=dict(r['arms'][arm]['summary'])
        summary['parent_rows']=[{**{k:v for k,v in x.items() if k!='source'},'parent':anon[x['source']]} for x in summary['parent_rows']]
        public['native_arms'][arm]=dict(summary=summary,queries=[dict(index=i,parent=anon[x['source']],metrics={m:x['metrics'][m] for m in metrics},format_ok=x['format_ok'],invalid_geometry=x['invalid_geometry']) for i,x in enumerate(r['arms'][arm]['rows'])])
    for name in ('comparisons','versus_full'):
        for arm,entries in r[name].items():
            public[name][arm]={}
            for m,v in entries.items():
                public[name][arm][m]={**v,'parent_delta_pp':{anon[p]:x for p,x in v['parent_delta_pp'].items()}}
    worker=read(OUT/'a03_native001/RECEIPT.json');wrapper=read(OUT/'a03_native001_wrapper/RECEIPT.json')
    resource=dict(new_GPU_worker_seconds=worker['seconds'],wrapper_seconds=wrapper['seconds'],
        GPU_allocation_and_wrapper_seconds=worker['seconds']+wrapper['seconds'],cumulative_GPU_seconds=total_prior(),cap=None,
        CPU_primary_seconds=read(D/'CPU_RECEIPT.json')['seconds'],CPU_independent_seconds=read(D/'INDEPENDENT_CPU_RECEIPT.json')['seconds'],
        CPU_field_audit_seconds=fa['CPU_seconds'],CPU_score_seconds=r['CPU_seconds'],CPU_summary_seconds=ra['CPU_seconds'],
        free_disk_bytes=shutil.disk_usage(ROOT).free,output_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file()),worker_exited=True)
    public['resources']=resource;write(D/'PUBLIC_COMPLETION_REPORT.json',public)
    lines=['# A0.3 Shared Channel Basis Feasibility — completed / independently audited','',
        'Train128/95 original source-training parents alone estimated the basis; Dev64/16 already exposed diagnosis parents only evaluated it. The cache targets are source-GT native gradient oracles. No learned predictor was fitted, no fresh31/388 or target was read.','',
        '## Shared channel structure','',
        'Each uncentered covariance is normalized by the full field energy before equal-query averaging. The diagnostic ranks and random seed were fixed before measurement. Per-query-optimal SVD is an upper bound, not the shared basis.','',
        '|Rank|Train shared mean energy %|Dev shared mean %|Dev shared median %|Dev optimal mean %|Dev shared/optimal mean %|Dev random mean %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in cpu['ranks']:
        t=cpu['summary']['train']['shared'][str(k)];d=cpu['summary']['dev']['shared'][str(k)]
        o=cpu['summary']['dev']['optimal'][str(k)];x=cpu['summary']['dev']['random'][str(k)]
        vals=[t['energy']['mean'],d['energy']['mean'],d['energy']['median'],o['energy']['mean'],d['retention_ratio']['mean'],x['energy']['mean']]
        lines.append('|'+str(k)+'|'+'|'.join(f'{100*v:.6f}' for v in vals)+'|')
    lines+=['',f"The independent Dev rank32 median {cpu['dev_rank32_median_energy']:.12f} passed the locked .75 resource gate. Full energy/cosine/retention mean, median and ranges for both splits and controls remain in the JSON.",'','## Conditional native screen','',
        'Frozen PTD4B/B1/union256, same exposed Dev64, two new projected oracle arms, zero backwards/optimizer. Each projection was renormalized to the full original coefficient-field norm, mapped through the original union basis, and inserted into both native passes at the original radius .13545580427763146. B1 and Full Oracle were reused after sealed hash plus actual pixel/preprocessing/feature-context identity checks; no redundant baseline native decoding. All256 predictions sealed before scoring.','',
        '|Arm|tIoU %|sIoU %|vIoU %|Δt vs B1 pp|Δs pp|Δv pp|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        vals=[100*r['arms'][arm]['summary']['parent_macro'][m] for m in metrics]
        deltas=[r['comparisons'][arm][m]['mean_delta_pp'] for m in metrics] if arm!='B1' else [0.,0.,0.]
        lines.append('|'+arm+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|'+'|'.join(f'{v:+.6f}' for v in deltas)+'|')
    lines+=['','Parent-macro is primary; each parent has four queries so the query means coincide here. sIoU uses the existing scorer\'s valid GT-frame support. No metric or invalid-output filtering changed.','',
        '|Arm|Δv paired parent 95% CI pp|v query harm >5pp|v query gain >5pp|v B1-good retained|t B1-good retained|Full Oracle v-gain retained %|Native gate|',
        '|---|---|---:|---:|---|---|---:|---|']
    for arm in arms[1:]:
        ci=r['comparisons'][arm]['vIoU']['bootstrap_ci95_pp'];tail=r['query_tails'][arm]['vIoU'];v=r['B1_good_retention'][arm]['vIoU'];t=r['B1_good_retention'][arm]['tIoU'];gr=r['oracle_gain_retention'][arm]
        gain_text=f'{100*gr:.6f}' if gr is not None else 'undefined'
        lines.append(f"|{arm}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|{tail['query_harm_gt5pp']}|{tail['query_gain_gt5pp']}|{v['retained']}/{v['eligible']}|{t['retained']}/{t['eligible']}|{gain_text}|{r['gates'][arm]['pass']}|")
    lines+=['','All per-query positive/negative/zero changes, temporal/spatial tails, format failures, invalid geometry, source-parent intervals and both rank-vs-Full comparisons are retained in the anonymous JSON. B1-good means score>.5; continuous damage can remain even when a threshold is retained. Bootstrap intervals are descriptive and unadjusted on this repeatedly exposed panel.','',
        '## Decision and limits','',f"Locked route: **{r['decision']}**.",
        'The native gate requires positive mean vIoU, nonnegative tIoU and sIoU, and no registered systematic branch collapse. Rank16 has priority when both pass. This selects only the next candidate; no factorized predictor, extra rank, full training, expert, trust gate or OPD was run.',
        'Shared low-rank oracle feasibility does not establish predictability from ordinary inputs or learned-predictor advantage. This is a source-GT oracle and not a deployable correction. A basis fitted on Train and evaluated on exposed Dev is not fresh confirmation. Neither gain retention nor energy is a safety guarantee; all harm and retention failures remain reported.','',
        '## Verification and resources','',
        f"Four synthetic CPU controls passed. All192 raw cached fields were independently reconstructed in Torch FP64: scalar/summary max {ca['scalar_summary_max_abs']:.12g}; direct projection cosine/norm max {ca['direct_projection_cosine_norm_max_abs']:.12g}. Native128 factor fields were independently checked and actual FP32 delta hashes reconstructed; delta error {fa['independent_delta_max_absolute_error']:.12g}, norm relative error {fa['norm_max_relative_error']:.12g}.",
        f"All256 predictions /768 geometric metrics had scalar-versus-tensor max {r['scalar_tensor_max_abs']:.12g}; reused scores matched exactly. Independent root {ra['checks']} parent/CI/tail/retention/gate checks max {ra['max_abs']:.12g}.",
        'The original CPU field auditor failed exact delta hashes because it forced OpenBLAS4 while inference inherited OpenBLAS20. Coefficient hashes all matched; across128 fields,23 had28 FP32 elements different, max7.450580596923828e-9. Isolated audit v2 restored the original20-thread execution only for hash reconstruction: every actual delta hash then matched, with independent Torch checks and tolerances unchanged. Original failed auditor/pins/FAILURE/diagnosis remain; no GPU rerun, prediction/metric change or sample removal. Exact duration of the first failed CPU audit was not captured by its original failure handler.',
        fa['limitation'],
        f"Actual native worker {worker['seconds']:.6f}s + nonoverlapping wrapper {wrapper['seconds']:.6f}s; cumulative allocation/wrapper {resource['cumulative_GPU_seconds']:.11f}s, cap=null. CPU primary {resource['CPU_primary_seconds']:.6f}s and independent {resource['CPU_independent_seconds']:.6f}s counted separately; remaining CPU audit times in JSON. Worker exited; free disk {resource['free_disk_bytes']} bytes above8GiB. No scientific artifacts removed."]
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(D/'ROOT_COMPLETION_SUMMARY.json',dict(status='completed_independently_audited',decision=r['decision'],source_queries=192,native_queries=64,native_predictions=256,new_native=128,reused_native=128,optimizer_steps=0,backwards=0,resources=resource,report_sha=sha(D/'PUBLIC_COMPLETION_REPORT.json')))
    write(D/'CURRENT_HANDOFF.json',dict(status='completed',decision=r['decision'],next_candidate_only=True,new_model_implemented=False,GPU_active=False,fresh_read=False,report=str(D/'REPORT.md')))
    write(D/'ACTIVE.json',dict(status='completed',time=time.time(),GPU_active=False))
    active=read(N/'ACTIVE.json');active.update(status='completed',completed_time=time.time(),GPU_active=False)
    (N/'ACTIVE.json').write_text(json.dumps(active,indent=2)+'\n')
    print('A03_SUMMARY_COMPLETE',r['decision'],resource)


if __name__=='__main__':main()

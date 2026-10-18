"""Anonymous four-arm retest, common support and real query-cost report."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_dino_endpoint_confirm32_v1 import D,P,CODE,verify
from vg_tta.stvg_dino_endpoint_confirm32_v1 import ARMS

def ci(v):return 'unknown'if v['mean']is None else f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}], n={v['parents']}"

def main():
    p=verify();x=read(P/'RESULTS_PUBLIC.json');roster=read(D/'ROSTER_LOCK.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    lines=['# New-parent retest of frozen local teacher-endpoint update','',
        'HC16 + Vid16, one query per parent. Membership locked before cache inventory and inference. Excluded original tuning parents, old rollback/error-state32, query-only64, external20-64, soft-correspondence32, spatial-readout32, controls256 and E-expanded256. Selection used fixed salted hashes only, not scores, GT or reference quality. Historical official full-result exposure remains: this is not a globally unseen test.','',
        'Same Source checkpoints, E estimator (beta1/16/source calibration), original four anchor nodes, source exploration scales, aligned content/exclusion mask, threshold0.35, whole-state legality and final-MBR. All four complete outputs sealed before GT scoring. No training, network-weight updates, dynamic OPD or parameter search.','',
        '| Output | HC complete vIoU % | Vid complete vIoU % |','|---|---:|---:|']
    for a in ARMS:lines.append(f"| {a} | {100*x['groups']['hc2']['metrics'][a]['v']:.6f} | {100*x['groups']['vidstg']['metrics'][a]['v']:.6f} |")
    for ds,g in x['groups'].items():
        lines+=['',f'## {ds}: paired changes, pp','',
            '| Contrast | Complete vIoU [95% CI] | Observed actual sIoU | Common-admitted actual sIoU |','|---|---:|---:|---:|']
        for a,d in g['paired'].items():lines.append('| '+a+' | '+' | '.join(ci(d[k])for k in ['v','observed_actual_s','common_admitted_s'])+' |')
        lines+=['',f"Known observed-node support: {g['known_observed_query_count']}/16 queries. Common-admitted nodes with GT: {g['coverage']['common_known_GT_nodes']} across {g['common_known_queries']}/16 queries. Missing GT stays unknown, all16 remain in complete means.",'',
            '| Coverage | Count |','|---|---:|']
        for k,v in g['coverage'].items():lines.append(f'| {k} | {v} |')
        lines+=['','| Output minus E | Observed v contribution pp | Unobserved actual v contribution pp | Benchmark interpolated v contribution pp | Positive / negative / neutral | Complete loss below -5pp |',
            '|---|---:|---:|---:|---:|---:|']
        for a in ARMS[1:]:
            d=g['v_effect_contributions_pp'][a];v=g['paired'][a+'_minus_E_MBR']['v']
            lines.append(f"| {a} | {d['observed_actual']['mean']:+.6f} | {d['unobserved_actual']['mean']:+.6f} | {d['benchmark_interpolated']['mean']:+.6f} | {v['positive']} / {v['negative']} / {v['neutral']} | {v['severe_loss_below_minus5pp']} /16 |")
        lines+=['','| Output | Mean observed center displacement | Mean observed log-size displacement |','|---|---:|---:|']
        for a in ARMS[1:]:
            d=g['displacement'][a]['observed'];lines.append(f"| {a} | {d['center_mean']:.9f} | {d['log_size_mean']:.9f} |")
    lines+=['','## Cost and interfaces','',
        'Center_full uses a single center reference per active original anchor, no sampled admissions. Samples24 queries legal states from the initial frozen E-centered distribution once. Center_p reuses the same center output and sample admission ratio p=n/24, with no extra forward. It therefore costs sampling information and is not a low-cost center baseline. Rejected center references remain E, including in Center_p.','',
        'Common support is defined before GT by accepted center AND positive sample count. Common-node sIoU compares both outputs on that identical actual-node set. Full-tube comparisons can include differences in admission support. Equal p is not equal actual displacement norm, nor a calibrated correctness probability.','',
        '| Actual new work | HC | Vid |','|---|---:|---:|']
    for k in ['encoding_frames','encoding_native_seconds','center_logical_queries','sample_logical_queries','unique_sample_queries','center_decoder_seconds','sample_decoder_seconds']:
        lines.append(f"| {k} | {x['groups']['hc2']['cost'][k]} | {x['groups']['vidstg']['cost'][k]} |")
    lines+=['',f"Total new GPU-allocation wall time: {x['compute']['GPU_total_allocation_seconds']:.6f}s, including frozen Source restoration and teacher worker. Teacher-worker allocation: {x['compute']['GPU_teacher_allocation_seconds']:.6f}s. New frozen Source captures32; original cached DINO calls0. The newly counted encoding_frames are real new native DINO image/text forwards shared by center and sampled decoders. Qualification/IO/model-load overhead is included in worker allocation, not only decoder-time subtotals.",'',
        'These costs reuse historical original expert inputs and E calibration. Decoder subtotals are measured after shared encoding; they are not first-use end-to-end latency. All model use was serial under the original shared GPU lease and resource guards.','',
        '## All signed complete-vIoU changes against E, pp','',
        '| Alias | Domain | Center_full | Samples24 | Center_p | Samples24 minus Center_p |','|---|---|---:|---:|---:|---:|']
    for r in x['rows']:
        v=r['values'];d={a:100*(v[a]['v']-v['E_MBR']['v'])for a in ARMS[1:]}
        lines.append(f"| {r['alias']} | {r['dataset']} | {d['Center_full_local']:+.6f} | {d['Samples24_endpoint']:+.6f} | {d['Center_p_local']:+.6f} | {d['Samples24_endpoint']-d['Center_p_local']:+.6f} |")
    lines+=['','## Decision boundary','',
        'This retest can distinguish center endpoints from sampled endpoints under the frozen one-round rule; it is not dynamic or on-policy adaptation. Candidate states and their teacher feedback were acquired once. Source/E and all nonobserved actual decoder nodes remain unchanged. Changes on benchmark-interpolated frames arise from the official interpolation rule, not extra visual evidence. All tIoUs match exactly.','',
        'Paired bootstrap10000, seed20261018, parent/query unit. Candidate pairs are not independent observations. Unknown/no-support/failed cases are not removed by effect. Both positive and negative findings must be retained. No dynamic OPD, new formula variant, threshold search, extra cohort or production promotion follows automatically.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from methods.decota_final_simplified_v1.config import MethodConfig
    checkpoints={}
    for ds,direction in [('hc2','vid_to_hc1'),('vidstg','hc2_to_vid')]:
        c=MethodConfig.for_direction(direction);checkpoints[ds]=dict(path=c.checkpoint,sha256=c.checkpoint_sha256,source_dataset=c.source_dataset)
    put(P/'CONFIGURATION.json',dict(count=32,per_domain=16,seed=p['seed'],source_checkpoints=checkpoints,
        scale=p['scale'],beta=1/16,threshold=.35,samples=24,rounds=1,temporal='final-distribution MBR',
        aligned_content_mask=True,backbone='TA-STVG',network_updates=0,no_parameter_search=True,center_p_not_low_cost=True,
        same_p_not_equal_norm=True,resource_budget=dict(GPU_seconds=900,wall_seconds=2400,MemoryHighGiB=16,MemoryMaxGiB=20,SwapMaxGiB=1),
        no_dynamic_OPD=True))
    put(P/'COHORT_PUBLIC.json',dict(count=32,domains=roster['domains'],rows=[dict(alias=r['alias'],dataset=r['dataset'])for r in p['records']],
        exclusions=[dict(path=r['path'],sha256=r['sha256'],records=r['records'])for r in roster['exclusions']],
        membership_before_cache_inventory=True,selection='fixed salted hashes, no GT/scores/quality',historical_full_exposure=True,globally_unseen=False))
    put(P/'CPU_PROPERTIES.json',read(D/'CPU_PROPERTIES.json'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axs,['hc2','vidstg']):
        g=x['groups'][ds];vals=[g['paired'][a+'_minus_E_MBR']['v']for a in ARMS[1:]];m=[v['mean']for v in vals]
        err=[[v['mean']-v['CI95'][0]for v in vals],[v['CI95'][1]-v['mean']for v in vals]]
        ax.bar(range(3),m,yerr=err,capsize=3,color=['#8495ad','#ba8057','#63a09b']);ax.axhline(0,color='black',lw=.6)
        ax.set_xticks(range(3),['Center full','24 endpoints','Center with p']);ax.set_title(ds+' new16 parents');ax.set_ylabel('Complete vIoU difference from E (pp)')
    fig.tight_layout();fig.savefig(P/'RETEST_COMPARISON.svg');fig.savefig(D/'RETEST_COMPARISON.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str((P/n).relative_to(ROOT))for n in ['RESULTS_PUBLIC.json','ROOT_INDEPENDENT_AUDIT.json','REPORT.md','CONFIGURATION.json','COHORT_PUBLIC.json','CPU_PROPERTIES.json','RETEST_COMPARISON.svg']]))
    put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32));print(P/'REPORT.md',flush=True)

if __name__=='__main__':main()

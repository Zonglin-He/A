"""Root analysis of completed TTA/timing; saved scalars/logits only, no GT/model."""
from full_common import *
import argparse

def tails(rr,a,b):
    import numpy as np
    t=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr])
    return dict(severe_t=int((t< -5).sum()),severe_v=int((v< -5).sum()),severe_union_once=int(((t< -5)|(v< -5)).sum()),rescue_union_once=int(((t>5)|(v>5)).sum()))

def fmt(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}]"

def tta(ds):
    import numpy as np,torch
    from stvg.runtime import guard
    sys.addaudithook(guard);torch.set_num_threads(1);validate()
    folder=P/ds;ap=D/f'{ds}_reporting_receipt.json';sp=D/f'{ds}_score_receipt.json';audit=read(ap);score=read(sp);seal=read(D/f'{ds}_sealed.json')
    assert audit['status']=='verified_full_domain_Tent_SAR'and audit['score_sha256']==sha(sp)and score['seal_sha256']==sha(D/f'{ds}_sealed.json')and seal['queries']==COUNTS[ds]
    report=read(folder/'report.json');rr=read(D/ds/'metrics.json')['private_rows'];assert len(rr)==COUNTS[ds]
    for arm,mm in report['means'].items():
        for k,v in mm.items():
            vals=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None]
            assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12
    all_tails={k:tails(rr,*k.split('-'))for k in report['paired']};atomic(folder/'all_paired_tails.json',all_tails)
    extra={m:dict(changed_logits_queries=0,nonzero_parameter_delta_queries=0,changed_native_interval_queries=0,changed_joint_interval_queries=0)for m in ['Tent','SAR']};segments={}
    for key,e in seal['files'].items():
        assert sha(D/e['logits_path'])==e['logits_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
        v=torch.load(D/e['logits_path'],map_location='cpu',weights_only=False,mmap=True)
        native=v.get('native_logits')
        if native is None:
            context,_=payload_context(ds,key);assert context['frame_ids']==v['frame_ids'];native=[x['native_logits']for x in context['offsets']]
        for m in extra:
            extra[m]['changed_logits_queries']+=int(any(not torch.equal(a,b)for a,b in zip(v['adapted_logits'][m],native)))
            extra[m]['nonzero_parameter_delta_queries']+=int(e['traces'][m]['parameter_delta']>0)
        w=e['worker'];g=segments.setdefault(f"{w['pid']}:{w['startticks']}",dict(queries=0,reused_completed_queries=0,new_Source_queries=0,new_optimizer_iterations=0,Source_forward_seconds=0.,acquisition_wall_seconds=0.,conditional_query_wall_seconds=0.,adaptation_seconds=0.))
        g['queries']+=1;g['reused_completed_queries']+=int(e['reused_completed']);g['new_Source_queries']+=e['actual_new_counts']['new_Source_queries'];g['new_optimizer_iterations']+=e['actual_new_counts']['optimizer_iterations']
        if not e['reused_completed']:
            origin=e['context_origin'].get('origin',{});g['Source_forward_seconds']+=origin.get('Source_forward_seconds',0.);g['acquisition_wall_seconds']+=origin.get('acquisition_wall_seconds',0.);g['conditional_query_wall_seconds']+=e['query_wall_seconds'];g['adaptation_seconds']+=sum(x['seconds']for x in e['traces'].values())
    for m in extra:
        extra[m]['changed_native_interval_queries']=audit['diagnostics'][m]['changed_native_queries'];extra[m]['changed_joint_interval_queries']=audit['diagnostics'][m]['changed_jointMAP_queries'];assert extra[m]['nonzero_parameter_delta_queries']==audit['diagnostics'][m]['nonzero_parameter_delta_queries']
    joint={m:dict(all_query_scored_metrics_equal_SourceJointMAP=all(r['values'][m+'JointMAP']==r['values']['SourceJointMAP']for r in rr),changed_joint_intervals=extra[m]['changed_joint_interval_queries'])for m in extra}
    assert all(x['all_query_scored_metrics_equal_SourceJointMAP']and x['changed_joint_intervals']==0 for x in joint.values())
    atomic(folder/'optimizer_diagnostics.json',dict(saved_logits_diagnostics=extra,official_reliable_filter_counts=audit['diagnostics'],matched_joint_readout_control=joint,explanation='Parameter/logit changes do not imply a discrete interval or metric gain. Native and jointMAP readouts are different. The jointMAP change from nativeSource cannot be attributed to optimizer updates.'))
    failure=read(D/'failure_runtime_v3_20261025/preservation.json');costs=dict(actual_new_counts=seal['counts'],current_worker_segments=segments,current_model_load=read(D/f'{ds}_model_load.json'),failed_actual_model_load=failure['model_load'],original_reused_optimizer_source_cost_nonzero=True,original_completed_TTA_audit_sha256=sha(PREV/'published_TTA/reporting_receipt.json'),Source_native_parent_acquisition_nonzero=True,scope='Each PID/start segment is separate. Failed model load retained. Source/acquisition/query/adaptation timers can be nested; do not sum them. Existing-context/reused-result costs are inherited, not zero-cost deployment. Full worker setup/outer wall is not independently reconstructed from overlapping timers; these cached-context baseline timings are not matched end-to-end deployment latency.')
    atomic(folder/'cost_segments.json',costs)
    with(folder/'full_TTA_table.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','R03_percent','R05_percent']);w.writerows([[a,*[100*m[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]]for a,m in report['means'].items()])
    s=f'# Full-target Tent/SAR: {ds}\n\nComplete **{len(rr)} queries / {report["parents"]} parents**, two baselines / {audit["configs"]} method-query configurations. Whole-domain seal preceded official labels: {score["checks"]} original/dense checks, max discrepancy {score["max_error"]:.17g}. All {audit["SHA_checks"]} prediction/logit/receipt SHA checks, native/joint readout replay, Source parent arrays, episodic resets and means passed independent audit. Root additionally checks saved-logit diagnostics and every scored mean, with zero model/Source/expert/optimizer/GT-scoring calls.\n\n|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|\n|---|---:|---:|---:|---:|---|\n'
    for a,m in report['means'].items():
        z=report['severe_vs_Source'][a];s+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{z['t']}/{z['v']}/{z['union']}|\n"
    s+='\n## Actual optimization and matched readout\n\n'+json.dumps(extra,indent=2)+'\n\n'+json.dumps(joint,indent=2)+'\n\nTent and SAR use the pinned official implementations with the previously frozen lr1e-4/five episodic steps; SAR margin fraction .8. Only temporal-decoder LayerNorm affine parameters are adapted, Source boxes and cached upstream contexts stay fixed, all decoder/optimizer state resets per method/query. This is a disclosed STVG adapter, not canonical BatchNorm/full-network/continual TTA or a general TTA ranking. Nonempty update counts and actual interval changes are distinguished; SAR reliable filtering may yield empty steps. Neither unchanged argmax nor this restricted negative result licenses retuning from full labels or a universal TTA-failure claim.\n\nPrimary Native uses original per-offset MAP plus physical hull. Secondary JointMAP uses the same joint readout as SourceJointMAP. Exact equality to that matched control must not be presented as optimizer gain over nativeSource. Ours uses current user-selected kappa0, eight expert observations and zero optimizer steps; TTA uses zero expert and five update steps. These are different budgets, and the user selected kappa0 after prior full-result exposure. No independent unseen confirmation claim.\n\n## All paired effects and tails\n\n|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
    for k,v in report['paired'].items():
        z=all_tails[k];s+=f"|{k}|{fmt(v['t'])}|{fmt(v['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
    s+='\n## Duration groups\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
    for group,g in report['timegroups'].items():
        for k,v in g['paired'].items():s+=f"|{group} / {k}|{g['queries']}|{fmt(v['t'])}|{fmt(v['v'])}|\n"
    s+='\nShort is native-GT duration <=5s. Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Severe is a signed loss greater than5pp in t or v, union counted once; rescues use greater than5pp gain. All signed positive and negative queries and fixed-support/boundary metrics remain in report.json and both TSVs.\n\n## Actual costs and provenance\n\n'+json.dumps(costs,indent=2)+'\n\nNo private captions/media/coordinates/features/weights are published.\n';(folder/'ANALYSIS.md').write_text(s)
    receipt=dict(status='root_complete_full_TTA_analysis',dataset=ds,audit_sha256=sha(ap),score_sha256=sha(sp),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),code_sha256=sha(Path(__file__)),all_signed_tails_and_timegroups=True,matched_readout_explicit=True,new_model=0,new_GT_scoring=0)
    atomic(D/f'{ds}_root_export_receipt.json',receipt);atomic(folder/'root_export_receipt.json',receipt);assert not torch.cuda.is_initialized();print(ds,'full TTA root analysis exported')

def timing():
    ns.validate();g=K0/'deployment';q=ns.P/'deployment';seal=read(g/'sealed.json');export=read(g/'export_receipt.json');v=read(g/'measurements.json');paired=read(q/'paired_timing.json')
    assert seal['calls']==192 and seal['all_current_MAP_outputs_bitwise']and seal['measurement_sha256']==sha(g/'measurements.json')==export['measurement_sha256']
    assert paired['all192outputs_bitwise']and paired['seal_sha256']==sha(g/'sealed.json')and export['paired_report_sha256']==sha(q/'paired_timing.json')and export['calls_TSV_sha256']==sha(q/'all_measured_calls.tsv')
    assert all(x['prediction_bitwise_verified']and not x['GT_read']for x in v['items'])
    means=v['report']['means'];startup={ds:dict(Source_checkpoint_load_seconds=z['source_checkpoint_load_seconds'],warmup={a:dict(query_wall_seconds=w['query_wall_seconds'],cost=w['cost'])for a,w in z['warmup'].items()})for ds,z in v['report']['startup_and_warmup'].items()}
    atomic(q/'stage_costs.json',dict(measured_stage_means=means,startup_and_warmup_excluded=startup,worker=seal['worker'],measured_Source_calls=192,measured_expert_calls=96,warmup_Source_calls=4,warmup_expert_calls=2,scope='End-to-end query wall is authoritative. Expert forward is nested inside complete expert request; prototype/pooling/finalDP nested inside finish wall. Do not add nested stages or substitute expert forward for complete overhead. Startup/load and warmup separate.'))
    s='# Current kappa0 actual deployment timing\n\nActual192 measured Source calls and96 same-eight-observation expert requests, plus4Source/2expert warmup calls. Every output was bitwise equal to the current audited NoShrink prediction and its MAP interval. These are new current-kappa0 measurements, not renamed legacy kappa64 times. Root actually reran the read-only timing export and verified its seal, all-call TSV and paired statistics.\n\n|Domain|Independent parents/queries|Repeats|Source sec/query|Full sec/query|Extra sec [95% CI]|Full/Source ratio [95% CI]|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for ds,x in paired['statistics'].items():
        assert x['queries']==16 and x['repeats']==3
        s+=f"|{ds}|16|3|{x['mean_Source_seconds']:.9f}|{x['mean_FullMAP_seconds']:.9f}|{x['paired_overhead_seconds']:.9f} [{x['paired_overhead_95CI'][0]:.9f}, {x['paired_overhead_95CI'][1]:.9f}]|{x['mean_time_ratio']:.6f} [{x['ratio_95CI'][0]:.6f}, {x['ratio_95CI'][1]:.6f}]|\n"
    s+='\nEach query first averages its3repeats;10000 paired-parent bootstrap draws use16 independent units/domain, not48 independent queries. Confidence intervals are pointwise. Same RTX5090/i5-14600KF, native224FP32/two offsets/eight observations and fixed input cohort. Models resident with original CPU/GPU offload. No inter-query RGB/feature/capture/teacher/prediction cache. Own CPU experimental groups were paused during timing and resumed afterward; shared-host conditional measurement, not a full-domain throughput guarantee. Original startup and warmup are separate. Prediction-reference verification IO is outside the timer; decode, parse, transfers, native Source, complete expert request, geometry, auxiliary DP, actual ROI repooling, prototype update, jointMAP, final DP and cleanup are inside the query wall.\n\n## Mean stages in complete-method calls\n\n|Domain|RGB decode|Native Source|Complete expert request|Expert forward (nested)|Geometry|Aux DP|ROI pooling|Prototype|MAP readout|Final DP|Finish wall (overlapping)|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for ds,z in means.items():
        x=z['FullMAP'];c=x['components'];keys=['Source_seconds','expert_seconds','expert_forward_seconds','geometry_seconds','auxiliary_DP_seconds','region_pooling_seconds','prototype_seconds','MAP_readout_seconds','final_DP_seconds','finish_wall_seconds'];s+='|'+ds+'|'+f"{x['mean_RGB_seconds']:.6f}|"+'|'.join(f"{c[k]:.6f}"for k in keys)+'|\n'
    s+='\nTimes are seconds. Expert forward is inside expert request, and several stages are inside finish wall; these columns must not be summed. The all192callTSV also retains parsing/transfers/per-call variation. Current Source and Full calls share the same fixed hardware, warmup and no-query-cache rules.\n\nHardware and complete paired results are in paired_timing.json; original startup/warmup and nested stage means are in stage_costs.json. No new model or GT scoring in root export.\n';(q/'ANALYSIS.md').write_text(s)
    receipt=dict(status='root_verified_actual_current_kappa0_timing_analysis',measurement_sha256=sha(g/'measurements.json'),seal_sha256=sha(g/'sealed.json'),paired_timing_sha256=sha(q/'paired_timing.json'),all_calls_tsv_sha256=sha(q/'all_measured_calls.tsv'),analysis_sha256=sha(q/'ANALYSIS.md'),code_sha256=sha(Path(__file__)),new_model=0,new_GT_scoring=0)
    atomic(g/'root_export_receipt.json',receipt);atomic(q/'root_export_receipt.json',receipt);print('Current-kappa0 timing root analysis exported')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['tta','timing']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args();tta(a.dataset)if a.task=='tta'else timing()

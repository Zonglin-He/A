"""Export one fully sealed Lite cohort independently of subsequent cohorts.

Only saved scalar results, hashes, receipts and source text are read. No
inference, fitting, annotations, raw teacher responses or tensor loading.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import export_stvg_lite_batch001 as normalization
from scripts import publish_stvg_soft_ep_v1 as publisher
BASE=ROOT/'artifacts/stvg_lite_event_support_v1'
ARM_NAMES=('Frozen','Temporal','Spatial_native_clock','Full','Matched_original_OPD')
METRICS=normalization.METRICS
SOURCES={}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    SOURCES[str(p.relative_to(ROOT))]=sha(p)
    return json.loads(p.read_text())


def write(p,d):normalization.write(p,d)


def context(cohort):
    assert cohort in ('development','confirmation')
    batch='003' if cohort=='development' else '004'
    return dict(cohort=cohort,count=128 if cohort=='development' else 64,
        scope=BASE if cohort=='development' else BASE/'confirmation',batch=batch,
        out=ROOT/'results/stvg_lite_event_support/2026-10-16'/('batch'+batch),
        pub=BASE/'publication'/('batch'+batch),
        doc=ROOT/'docs'/('STVG_LITE_'+cohort.upper()+'_CLOSED_LOOP.md'))


def check_group(c):
    scope=c['scope'];n=c['count'];out=scope/'complete'
    cpu=read(out/'CPU_COMPLETION.json');assert cpu['status']=='complete' and cpu['count']==n and cpu['GT_read'] is True
    assert cpu['row_count']==5*n and cpu['bootstrap_executed'] is False and cpu['CURRENT_promoted'] is False
    assert cpu['new_selection_using_these_complete_outputs'] is False
    for filename,h in cpu['output_sha256'].items():assert sha(out/filename)==h
    pred=read(out/'GLOBAL_PREDICTION_BARRIER.json');assert pred['status']=='complete' and pred['count']==n and pred['GT_read'] is False
    plan=read(scope/'COMPLETE_PIPELINE_LOCK.json');assert plan['cohort']==c['cohort'] and plan['count']==n
    assert plan['trials']['full']!=plan['trials']['native'] and plan['no_further_method_selection'] is True
    assert pred['bindings']==cpu['bindings']
    for rel,h in cpu['bindings'].items():assert sha(ROOT/rel)==h
    assert cpu['bindings'][str((scope/'COMPLETE_PIPELINE_LOCK.json').relative_to(ROOT))]==sha(scope/'COMPLETE_PIPELINE_LOCK.json')
    recipe=read(BASE/'FINAL_RECIPE_LOCK.json');assert recipe['status']=='locked' and recipe['suite_final'] is False
    assert plan['recipe_sha256']==sha(BASE/'FINAL_RECIPE_LOCK.json')
    roots={};costs={};designs={}
    for kind in ('full','native'):
        b=scope/'trials'/plan['trials'][kind]
        r=read(b/'ROOT_READBACK.json');s=read(b/'STUDENT_GLOBAL_BARRIER.json');p=read(b/'GLOBAL_PREDICTION_BARRIER.json');d=read(b/'DESIGN_LOCK.json')
        assert r['status']=='pass' and s['status']==p['status']=='complete' and r['count']==s['count']==p['count']==n
        assert r['GT_read'] is s['GT_read'] is p['GT_read'] is False
        assert r['barrier_sha256']==p['student_barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json')
        assert p['root_readback_sha256']==sha(b/'ROOT_READBACK.json') and r['runtime_sha256']==sha(b/'RUNTIME.json')
        assert d['cohort']==c['cohort'] and d['reference_clock']==kind and d['recipe_sha256']==plan['recipe_sha256']
        assert d['recipe']==recipe['selected_spatial_recipe'] and d['response_scope']==recipe['selected_response_scope']
        for ds in ('hc2','vidstg'):assert d['configs'][ds]['M0']==d['configs'][ds]['M2']==recipe['configs'][ds]
        roots[kind]=r;designs[kind]={k:d[k] for k in ['trial_id','cohort','reference_clock','recipe','response_scope','configs','response_settings']}
        costs[kind]={k:s[k] for k in ['count','actual_fit_invocations','mode_fit_invocations','mode_optimizer_steps','source_capture_calls','full_readout_calls','reused_fit_count','wall_seconds'] if k in s}
    rows=read(out/'POSTSEAL_ROWS.json');summary=read(out/'POSTSEAL_SUMMARY.json');paired=read(out/'POSTSEAL_PAIRED.json')
    for d in (rows,summary,paired):assert d['bindings']==cpu['bindings'] and d['status']=='complete' and d['count']==n
    assert len(rows['rows'])==5*n and len({(r['case_id'],r['arm']) for r in rows['rows']})==5*n
    assert {r['arm'] for r in rows['rows']}==set(ARM_NAMES)
    assert all(r['status']=='scored' and set(r['metrics'])==set(METRICS) for r in rows['rows'])
    return rows['rows'],summary['summary'],paired['paired_contrasts'],recipe,roots,costs,designs


def aliases_for(rows,cohort):
    aliases={}
    for ds,prefix in [('hc2','HC'),('vidstg','VID')]:
        ids=sorted({r['case_id'] for r in rows if r['dataset']==ds})
        aliases.update({cid:('C' if cohort=='confirmation' else '')+f'{prefix}{i:03}' for i,cid in enumerate(ids,1)})
    return aliases


def anonymous(v,aliases):
    if isinstance(v,dict):return {aliases.get(k,k):anonymous(x,aliases) for k,x in v.items()}
    if isinstance(v,list):return [anonymous(x,aliases) for x in v]
    return aliases.get(v,v) if isinstance(v,str) else v


def export(c):
    o=c['out'];assert not (o/'BATCH_RECEIPT.json').exists()
    rows,summary,paired,recipe,roots,costs,designs=check_group(c);aliases=aliases_for(rows,c['cohort'])
    pubrows=normalization.sanitized_rows(rows,aliases)
    for a,b in zip(rows,pubrows):b['support_clock']=a['support_clock']
    write(o/'ROWS.json',dict(status='complete',cohort=c['cohort'],count=c['count'],row_count=len(rows),metrics_unit='fraction',rows=pubrows))
    write(o/'SUMMARY.json',dict(status='complete',summary=anonymous(summary,aliases)))
    write(o/'PAIRED.json',dict(status='complete',paired_contrasts=anonymous(paired,aliases),bootstrap_executed=False))
    write(o/'CONFIGURATION.json',dict(status='locked',selected_spatial_recipe=recipe['selected_spatial_recipe'],
          selected_response_scope=recipe['selected_response_scope'],selected_temporal=recipe['selected_temporal'],
          configs=recipe['configs'],response_settings=recipe['response_settings'],trials=designs,
          same_information_same_parameters_M0_control=True,online_LN_carry=False))
    write(o/'EXECUTION_READBACK.json',dict(status='pass',count=c['count'],
          trial_counters={k:r['counters'] for k,r in roots.items()},
          GT_only_after_both_clocks_and_all_predictions_sealed=True,
          independent_saved_mathematics_and_complete_output_readback=True,
          full_backward_independently_recomputed=False))
    # Parent stages are complete before this group's score, so no active or
    # subsequent-cohort stage is included in its cost evidence.
    stages=[]
    for base in [BASE/'execution_complete001',BASE/'execution_complete005']:
        if not base.exists():continue
        for p in sorted(base.iterdir()):
            if p.is_dir() and p.name.startswith(c['cohort']+'_') and (p/'LAUNCH.json').exists() and (p/'WORKER_EXIT.json').exists():
                a=read(p/'LAUNCH.json');z=read(p/'WORKER_EXIT.json')
                stages.append(dict(stage=p.name,exit_code=z['exit_code'],worker_wall_seconds=z['time']-a['time'],engineering_failed_attempt=z['exit_code']!=0))
    write(o/'MEASURED_COSTS.json',dict(status='complete',student_workers=costs,stage_workers=stages,
          scope='actual serial worker times; stages include CPU input and readback where labelled; excludes idle/root publication time; no full-set extrapolation',
          development_native_teacher_exact_reuse=c['cohort']=='development'))
    old=read(ROOT/'results/stvg_lite_event_support/2026-10-16/batch001/SOURCE_BINDINGS.json')
    existing={r['path']:r for r in old['source_snapshot']};runtime=read(BASE/'COMPLETE_DISPATCH_RUNTIME_TEMPLATE005.json');source=[]
    for rel,h in sorted(runtime['pins'].items()):
        if not rel.endswith('.py'):continue
        assert sha(ROOT/rel)==h
        if rel in existing and existing[rel]['original_sha256']==h:
            source.append(dict(path=rel,original_sha256=h,public_sha256=existing[rel]['public_sha256'],snapshot_batch='001'))
        else:
            text=normalization.public_source_text((ROOT/rel).read_bytes());dst=o/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text)
            source.append(dict(path=rel,original_sha256=h,public_sha256=sha(dst),snapshot_batch=c['batch']))
    write(o/'SOURCE_BINDINGS.json',dict(status='pass',measurement_sources=SOURCES,source_closure=source,
          prior_source_snapshot='../batch001/source_snapshot/',raw_runtime_or_teacher_assets_not_distributed=True))
    write(o/'ENGINEERING_RECOVERY.json',dict(status='resolved',originals_preserved=True,
          issue='reference builder unpacked two values from a three-value source helper',
          correction='separate builder002 captures third unused query; remaining44 functions AST-identical',
          synthetic_cohort_clock_cases=4,wrong_width_rejections=4,
          completed_development_teacher_jobs_preserved=512,new_development_teacher_GPU_reruns=0,
          scientific_algorithm_parameters_and_thresholds_unchanged=True))
    means={ds:{arm:100*math.fsum(r['metrics']['m_vIoU'] for r in rows if r['dataset']==ds and r['arm']==arm)/(c['count']//2) for arm in ARM_NAMES} for ds in ('hc2','vidstg')}
    write(o/'FIVE_ARM_COMPARISON.json',dict(status='complete',means_percent=means,
          matched_original_OPD_definition='same final parameters and teacher information as Full; not separately retuned optimum'))
    plot(o,means,c['cohort'])
    lines=['# Lite actual-clock '+c['cohort']+' closed loop — batch '+c['batch'],'',
      'This cohort completed both real Full and Native-clock Student pipelines, independent source/saved-mathematics/readout checks and a joint pre-GT seal before five-arm scoring. Subsequent cohorts are not required for this incremental publication. This is not whole-study FINAL or CURRENT promotion.','',
      '## Fixed method and cohort','',
      f'TA-STVG cross-domain Source checkpoints; {c["count"]} sources, {c["count"]//2} per target domain, one query per parent. '+('All128 sources were exposed development; no new selection is made from this closed-loop result.' if c['cohort']=='development' else 'These64 sources exclude the current128 development parents; earlier historical exposure is disclosed, so they are not claimed globally untouched. No confirmation-driven retuning.'),'',
      'Temporal-Lite: shared candidate supervision multiplied by offset count, no parameter L2, original full-grid per-offset trust and 512-parameter L-BFGS. Spatial: Lite-fullstep/full-grid decoder-response direction update, 1792 Query/LN parameters, episodic reset. HC lr0.0045/sigma0.025/tau0.1/5steps; Vid lr0.027/sigma0.4/tau0.5/3steps. Frozen R1 and existing weight-free candidates remain unchanged. No DINO, YOLO or SAM is added.','',
      'The Full branch reconstructs candidates and R1 references at the selected Temporal-Lite interval. Spatial-only uses actual Native-clock references, not Full-clock geometry with a different final interval. Matched_original_OPD uses the same Full references and final parameters as Full, with the original validated OPD update; it is not its separately retuned optimum. No GT enters candidates, references or online step acceptance.','',
      '## Complete five-arm result','', '| Method | HC m_vIoU (%) | VidSTG m_vIoU (%) |','|---|---:|---:|']
    lines += [f'| {arm} | {means["hc2"][arm]:.9f} | {means["vidstg"][arm]:.9f} |' for arm in ARM_NAMES]
    for baseline in ['Frozen','Temporal','Matched_original_OPD']:
        delta={ds:means[ds]['Full']-means[ds][baseline] for ds in means}
        lines+=['',f'Full minus {baseline}: HC {delta["hc2"]:+.6f} pp; VidSTG {delta["vidstg"]:+.6f} pp.']
    positive_frozen=all(means[ds]['Full']>means[ds]['Frozen'] for ds in means)
    below_matched=all(means[ds]['Full']<means[ds]['Matched_original_OPD'] for ds in means)
    interpretation=('Full improves over Frozen in both domains in this cohort.' if positive_frozen else 'Full does not improve over Frozen in both domains in this cohort.')
    interpretation+=(' It is below the same-parameter original OPD control in both domains.' if below_matched else 'Its relation to the matched original OPD control is reported separately for each domain above.')
    interpretation+=' All negative findings are retained and do not trigger another parameter search.'
    interpretation+=(' Development evidence is not independent generalization evidence.' if c['cohort']=='development' else 'This confirmation result does not retroactively select another method or parameter configuration.')
    reuse_statement=('Development Native teacher evidence is exact reuse of the prior sealed same-input run.' if c['cohort']=='development' else 'Confirmation Native teacher reuse, if any, must be established by its own actual receipts; development reuse is not assumed.')
    lines+=['',interpretation,'',
      'JSON scores are fractions, paired deltas percentage points. Every positive/negative/neutral/unknown source is retained with the saved1e-6 pp neutral tolerance. Native and Full supervision clocks differ: their supervised/unsupervised-frame means do not share identical supports and must not be mixed as a matched-support contrast. Dense GT-frame metrics and complete Tube metrics remain separately labelled. No bootstrap.','',
      'Costs contain actual worker elapsed times, including separately labelled failed engineering attempts if present, and Student fit/capture/readout counts. '+reuse_statement+' These are not cold-start deployment or official full-set speed measurements. The frozen external expert may carry target-domain prior; overlap-free external training is not established.','',
      'Videos, annotations, weights, raw tensor caches, real-query prompts, raw teacher responses and private conversation are excluded. Source snapshots inherit the previously audited complete batch001 closure and add only the necessary new engineering source files.','',
      '[All anonymous rows, summaries, paired cases, actual configuration, costs and source bindings](../results/stvg_lite_event_support/2026-10-16/batch'+c['batch']+'/)','','![Five-arm comparison](../results/stvg_lite_event_support/2026-10-16/batch'+c['batch']+'/FIVE_ARMS.png)','']
    c['doc'].write_text('\n'.join(lines))
    write(o/'BATCH_RECEIPT.json',dict(status='complete',cohort=c['cohort'],count=c['count'],logical_score_rows=len(rows),
          all_positive_negative_neutral_unknown_preserved=True,new_inference=False,new_GT_read=False,
          whole_study_FINAL_complete=False,CURRENT_promoted=False,source_closure_count=len(source),
          new_source_snapshot_count=sum(s['snapshot_batch']==c['batch'] for s in source)))


def audit(c):
    o=c['out'];raw=read(c['scope']/'complete/POSTSEAL_ROWS.json')['rows'];public=read(o/'ROWS.json')['rows'];aliases=aliases_for(raw,c['cohort']);n=c['count']//2
    assert len(raw)==len(public)==c['count']*5
    for r,p in zip(raw,public):
        assert p['case_id']==aliases[r['case_id']] and p['dataset']==r['dataset'] and p['arm']==r['arm']
        assert p['status']==r['status'] and p['metrics']==r['metrics'] and p['support_clock']==r['support_clock']
        assert set(p)=={'case_id','dataset','arm','status','metrics','support_counts','support_clock'}
    original_summary=read(c['scope']/'complete/POSTSEAL_SUMMARY.json')['summary']
    assert read(o/'SUMMARY.json')['summary']==anonymous(original_summary,aliases)
    for s in original_summary:
        metric=s['metric'];sub=[r for r in raw if r['dataset']==s['dataset'] and r['arm']==s['arm']];assert len(sub)==n
        if metric in METRICS:
            v=[r['metrics'][metric] for r in sub if r['metrics'][metric] is not None]
            assert len(v)==s['n_valid']
            if v:assert abs(math.fsum(v)/len(v)-s['mean'])<1e-12
        else:
            v=[r['metrics']['m_vIoU'] for r in sub if r['metrics']['m_vIoU'] is not None];threshold=float(metric.split('>=')[1]);assert abs(sum(x>=threshold for x in v)/len(v)-s['mean'])<1e-12
    pairs=read(o/'PAIRED.json')['paired_contrasts'];assert pairs==anonymous(read(c['scope']/'complete/POSTSEAL_PAIRED.json')['paired_contrasts'],aliases)
    for p in pairs:
        a,b=p['contrast'].split('_minus_');ds=p['dataset'];m=p['metric'];tol=p['neutral_tolerance_pp']
        x={r['case_id']:r['metrics'][m] for r in public if r['dataset']==ds and r['arm']==a};y={r['case_id']:r['metrics'][m] for r in public if r['dataset']==ds and r['arm']==b}
        delta={cid:100*(x[cid]-y[cid]) for cid in x if x[cid] is not None and y[cid] is not None}
        assert delta==p['per_source_delta_pp'] and len(delta)==p['n_pairs']
        assert set(p['unknown_case_ids'])==set(x)-set(delta)
        assert set(p['positive_case_ids'])=={k for k,v in delta.items() if v>tol}
        assert set(p['negative_case_ids'])=={k for k,v in delta.items() if v<-tol}
        assert set(p['neutral_case_ids'])=={k for k,v in delta.items() if abs(v)<=tol}
        if delta:assert abs(math.fsum(delta.values())/len(delta)-p['mean_delta_pp'])<1e-12
    sb=read(o/'SOURCE_BINDINGS.json')
    for rel,h in sb['measurement_sources'].items():assert sha(ROOT/rel)==h
    for r in sb['source_closure']:
        assert sha(ROOT/r['path'])==r['original_sha256']
        f=ROOT/'results/stvg_lite_event_support/2026-10-16'/('batch'+r['snapshot_batch'])/'source_snapshot'/r['path'];assert sha(f)==r['public_sha256']
    forbidden={'parent_sha256','query_text','caption','raw_output','video_path','physical_frame_ids','supervised_positions','box_xyxy','box_cxcywh','optimizer_state','GT_coordinates','weights','gradient'}
    def scan(v):
        if isinstance(v,dict):
            for k,x in v.items():assert k not in forbidden,k;scan(x)
        elif isinstance(v,list):
            for x in v:scan(x)
        elif isinstance(v,str):assert '/home/' not in v and '.codex/attachments' not in v
    for p in o.rglob('*'):
        if p.is_file():
            assert p.suffix in {'.json','.py','.png','.md'}
            if p.suffix=='.json':scan(read(p))
            if p.suffix in {'.py','.md'}:assert '/home/' not in p.read_text() and '.codex/attachments' not in p.read_text()
    proof=dict(status='pass',count=c['count'],rows_checked=len(public),metric_values_checked=len(public)*5,
          null_values_preserved=sum(v is None for r in public for v in r['metrics'].values()),summary_cells_checked=len(original_summary),
          paired_cells_checked=len(pairs),all_values_and_signs_exact=True,new_GT_read=False,new_inference=False)
    write(o/'ROOT_PUBLIC_SCALAR_AUDIT.json',proof);print(json.dumps(proof))


def plot(o,means,cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    labels=['Frozen','Temporal','Spatial Native','Full','Matched original OPD']
    for ax,ds in zip(axs,means):
        ax.bar(range(5),[means[ds][a] for a in ARM_NAMES],color=['#8b949e','#4c78a8','#72b7b2','#e45756','#f2a541'])
        ax.set_xticks(range(5),labels,rotation=28,ha='right');ax.set_ylabel('m_vIoU (%)');ax.set_title(ds+' '+cohort);ax.grid(axis='y',alpha=.2)
    fig.savefig(o/'FIVE_ARMS.png',dpi=160);plt.close(fig)


def paths(c):
    assert read(c['out']/'ROOT_PUBLIC_SCALAR_AUDIT.json')['status']=='pass'
    return {str(p.relative_to(ROOT)) for p in c['out'].rglob('*') if p.is_file()}|{str(c['doc'].relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))}


def main():
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['development','confirmation']);p.add_argument('mode',choices=['export','audit','stage','verify']);a=p.parse_args();c=context(a.cohort);c['pub'].mkdir(parents=True,exist_ok=True)
    if a.mode=='export':export(c);audit(c)
    elif a.mode=='audit':audit(c)
    else:
        publisher.BASE=c['pub'];publisher.expected_paths=lambda:paths(c)
        if a.mode=='stage':publisher.stage(publisher.DEFAULT_CHECKOUT,None,False)
        else:publisher.verify_stage(publisher.DEFAULT_CHECKOUT)

if __name__=='__main__':main()

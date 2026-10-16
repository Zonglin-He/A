"""Offline explicit export of sealed development measurements; never loads models/GT."""
import ast
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / 'artifacts/stvg_soft_ep_coordinate128_v1'
COMP = ROOT / 'artifacts/stvg_component_ablation128_v1'
TEMP = COMP / 'temporal_component/revision005'
NATIVE = COMP / 'native_clock/revision002'
OUT = ROOT / 'results/stvg_coordinate_component128/2026-10-16'
DOC = ROOT / 'docs/STVG_COORDINATE_COMPONENT128_V1.md'
SOURCE_HASHES = {}
ERRORS = []
METRICS = ['m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_supervised_GT','m_sIoU_unsupervised_GT']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCE_HASHES[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def check_complete(path, status='complete'):
    d = read(path)
    assert d['status'] == status, str(path)
    assert d['count'] == 128
    return d


def aggregate(rows, saved_summary):
    index = {(r['case_id'], r['arm']): r for r in rows}
    assert len(index) == len(rows)
    for arm in {r['arm'] for r in rows}:
        for ds in ['hc2','vidstg']:
            assert len([r for r in rows if r['dataset']==ds and r['arm']==arm]) == 64
    for r in saved_summary:
        if r['metric'] not in METRICS:
            continue
        values = [x['metrics'][r['metric']] for x in rows if x['dataset']==r['dataset'] and x['arm']==r['arm'] and x['metrics'][r['metric']] is not None]
        assert len(values)==r['n_valid']
        assert r['n_unknown']==64-len(values)
        error = abs(math.fsum(values)/len(values)-r['mean']) if values else 0.
        assert error < 1e-12
        ERRORS.append(error)
    return index


def export_rows(rows, aliases):
    return [dict(case_id=aliases[r['case_id']],dataset=r['dataset'],arm=r['arm'],status=r.get('status','scored'),metrics=r['metrics'],support_counts={k:v for k,v in r.get('support',{}).items() if isinstance(v,(int,float,bool))}) for r in rows]


def anon(value, aliases):
    if isinstance(value,dict):
        return {k:anon(v,aliases) for k,v in value.items()}
    if isinstance(value,list):
        return [anon(v,aliases) for v in value]
    if isinstance(value,str):
        return aliases.get(value,value)
    return value


def mean(index,ds,arm,metric='m_vIoU'):
    vals=[r['metrics'][metric] for r in index.values() if r['dataset']==ds and r['arm']==arm and r['metrics'][metric] is not None]
    return math.fsum(vals)/len(vals)*100


def contrast(a, arm_a, b, arm_b, label, aliases):
    out=[]
    for ds in ['hc2','vidstg']:
        aa={r['case_id']:r for r in a.values() if r['dataset']==ds and r['arm']==arm_a}
        bb={r['case_id']:r for r in b.values() if r['dataset']==ds and r['arm']==arm_b}
        assert aa.keys()==bb.keys() and len(aa)==64
        for metric in METRICS:
            rows=[]; unknown=[]
            for cid in sorted(aa):
                av,bv=aa[cid]['metrics'][metric],bb[cid]['metrics'][metric]
                if av is None or bv is None:unknown.append(aliases[cid])
                else:rows.append(dict(case_id=aliases[cid],delta_pp=(av-bv)*100))
            vals=[r['delta_pp'] for r in rows]
            out.append(dict(dataset=ds,contrast=label,metric=metric,n_pairs=len(rows),unknown_case_ids=unknown,mean_delta_pp=statistics.fmean(vals) if vals else None,positive=sum(x>1e-10 for x in vals),negative=sum(x<-1e-10 for x in vals),neutral=sum(abs(x)<=1e-10 for x in vals),rows=rows))
    return out


def load_trial(path, search=False, temporal=False):
    if temporal:
        cpu=check_complete(path/'SCORE_CPU_COMPLETION.json')
        root=check_complete(path/'ROOT_TEMPORAL_COMPONENT_READBACK.json','pass')
        prediction=check_complete(path/'PREDICTION_TEMPORAL_COMPONENT_BARRIER.json')
        assert prediction['root_temporal_component_readback_sha256']==digest(path/'ROOT_TEMPORAL_COMPONENT_READBACK.json')
        assert prediction['temporal_component_barrier_sha256']==digest(path/'TEMPORAL_COMPONENT_GLOBAL_BARRIER.json')
        prefix='TEMPORAL_COMPONENT_SCORE_'
    else:
        cpu=check_complete(path/'CPU_COMPLETION.json')
        root=check_complete(path/'ROOT_READBACK.json','pass')
        student=check_complete(path/'STUDENT_GLOBAL_BARRIER.json')
        prediction=check_complete(path/'GLOBAL_PREDICTION_BARRIER.json')
        assert root['barrier_sha256']==digest(path/'STUDENT_GLOBAL_BARRIER.json')
        prefix='POSTSEAL_SCORE_' if search else 'POSTSEAL_'
    assert cpu['GT_read'] is True and root['GT_read'] is False and prediction['GT_read'] is False
    if search:
        rowsdoc=read(path/'POSTSEAL_SCORE_ROWS.json')
        summary=read(path/'POSTSEAL_SCORE_SUMMARY.json')['summary']
        paired=read(path/'POSTSEAL_PAIRED_CONTRASTS.json')['paired_contrasts']
        assert cpu['bindings']==rowsdoc['bindings']
        bindings=rowsdoc['bindings']
        assert bindings['prediction_seal_sha256']==digest(path/'GLOBAL_PREDICTION_BARRIER.json')
        assert bindings['root_readback_sha256']==digest(path/'ROOT_READBACK.json')
        assert bindings['runtime_sha256']==digest(path/'RUNTIME.json')
    else:
        barname=prefix+'BARRIER.json' if temporal else 'POSTSEAL_SCORE_BARRIER.json'
        score=read(path/barname)
        assert score['status']=='complete'
        assert cpu['score_barrier_sha256']==digest(path/barname)
        for key in ['rows','summary','paired']:
            filename=prefix+key.upper()+'.json'
            assert score[key+'_sha256']==digest(path/filename)
        exposure='TEMPORAL_COMPONENT_GT_EXPOSURE.json' if temporal else 'POSTSEAL_GT_EXPOSURE.json'
        assert score['gt_exposure_sha256']==digest(path/exposure)
        rowsdoc=read(path/(prefix+'ROWS.json'))
        summary=read(path/(prefix+'SUMMARY.json'))['summary']
        paired=read(path/(prefix+'PAIRED.json'))['paired_contrasts']
    rows=rowsdoc['rows']
    index=aggregate(rows,summary)
    return dict(rows=rows,summary=summary,paired=paired,index=index,root=root,cpu=cpu)


def main():
    assert read(SEARCH/'SEARCH_COMPLETION006.json')['status']=='complete_pending_independent_root_view_public_archive'
    roster=read(SEARCH/'trials/D000/POSTSEAL_SCORE_ROWS.json')['rows']
    cases=sorted({r['case_id'] for r in roster})
    ds_by_case={r['case_id']:r['dataset'] for r in roster}
    aliases={}
    for ds,prefix in [('hc2','HC'),('vidstg','VID')]:
        ids=[c for c in cases if ds_by_case[c]==ds]; assert len(ids)==64
        aliases.update({c:f'{prefix}{i:03}' for i,c in enumerate(ids,1)})
    search_ids=[f'D{i:03}' for i in range(12)]+[f'E{i:03}' for i in range(12,15)]+[f'F{i:03}' for i in range(15,42)]
    datasets={};trials=[];rows_total=0
    for trial in search_ids+[f'A{i:03}' for i in range(16)]+['TEMPORAL','N000']:
        is_search=trial in search_ids
        path=SEARCH/'trials'/trial if is_search else (TEMP if trial=='TEMPORAL' else (NATIVE/'trials/N000' if trial=='N000' else COMP/'trials'/trial))
        d=load_trial(path,is_search,trial=='TEMPORAL');datasets[trial]=d
        public=OUT/'trials'/trial
        rows=export_rows(d['rows'],aliases)
        write(public/'ROWS.json',dict(status='complete',trial_id=trial,count=128,row_count=len(rows),all_sources_development=True,metrics_unit='fraction',GT_read_after_seal=True,bootstrap_executed=False,rows=rows))
        write(public/'SUMMARY.json',dict(status='complete',trial_id=trial,summary=anon(d['summary'],aliases)))
        write(public/'PAIRED.json',dict(status='complete',trial_id=trial,paired_contrasts=anon(d['paired'],aliases)))
        if trial!='TEMPORAL':
            design=read(path/'DESIGN_LOCK.json')
            keys=['status','trial_id','variant','ablation_flags','scope','coordinate','coordinate_value','configs','response_settings','family_reuse_from']
            write(public/'CONFIGURATION.json',{k:design[k] for k in keys if k in design})
        counters={k:v for k,v in d['root'].items() if k in ['status','count','scope','counters','max_errors','baseline_fit_exact_replay_count','fitted_case_count','explicit_no_fit_fallback_count']}
        write(public/'EXECUTION_READBACK.json',dict(root=counters,CPU_complete=d['cpu']['status'],GT_read_after_seal=True,actual_full_backward_independently_recomputed=False))
        trials.append(dict(trial_id=trial,count=128,row_count=len(rows),summary_cells=len(d['summary']),paired_cells=len(d['paired']),scope='coordinate_search' if is_search else ('conditional_temporal_deletion' if trial=='TEMPORAL' else 'true_native_clock_removal' if trial=='N000' else 'spatial_deletion')))
        rows_total+=len(rows)
    completed=read(SEARCH/'SEARCH_COMPLETION006.json')
    write(OUT/'SEARCH_SELECTION.json',{k:completed[k] for k in ['history','selections','best_trial','configs','response_settings','scope','independent_confirmation_complete','bootstrap_executed']})
    final=read(COMP/'FINAL_PARAMETER_LOCK.json')
    write(OUT/'FINAL_M2_CONFIGURATION.json',{k:final[k] for k in ['status','count','configs','response_settings','actual_baseline_trial','selection_used_development_GT','ablation_retuning_allowed','bootstrap_executed','CURRENT_promoted']})
    write(OUT/'ABLATION_DESIGN.json',{k:v for k,v in read(COMP/'DESIGN_LOCK.json').items() if k not in ['time','source_roster_sha256']})
    contrasts=[];baseline=datasets['A000']['index']
    for i in range(1,16):
        tid=f'A{i:03}';contrasts+=contrast(datasets[tid]['index'],'TS_M2',baseline,'TS_M2',tid+' minus A000',aliases)
    for arm in ['TS_no_grid_trust','TS_no_temporal_l2','TS_shared_offset_candidate','TS_projection_only']:
        contrasts+=contrast(datasets['TEMPORAL']['index'],arm,datasets['TEMPORAL']['index'],'TS_offset_trust',arm+' minus full_temporal',aliases)
    for a,aa,b,bb,label in [('N000','S_M2','A000','F','Native Spatial minus Frozen'),('A000','TS_M2','N000','S_M2','Full minus Native Spatial'),('A000','S_M2','N000','S_M2','QA-clock Spatial minus Native-clock Spatial'),('A000','TS_M2','A000','S_M2','Full minus QA-clock Spatial'),('A000','TS_M2','A000','T','Full minus Temporal')]:
        contrasts+=contrast(datasets[a]['index'],aa,datasets[b]['index'],bb,label,aliases)
    write(OUT/'CROSS_TRIAL_PAIRED.json',dict(status='complete',paired=contrasts,bootstrap_executed=False,matched_support_warning='supervised/unsupervised metrics across Native and QA reference clocks do not have identical support'))
    finalrows=[]
    for ds in ['hc2','vidstg']:
        for arm,idx,key in [('Frozen',baseline,'F'),('Temporal',baseline,'T'),('Spatial_Native_clock',datasets['N000']['index'],'S_M2'),('Spatial_QA_clock',baseline,'S_M2'),('Full',baseline,'TS_M2')]:
            finalrows.append(dict(dataset=ds,arm=arm,metrics_percent={m:mean(idx,ds,key,m) for m in METRICS[:3]}))
        for mode in ['M0','M1','M2']:
            tid=completed['best_trial'][ds][mode]
            finalrows.append(dict(dataset=ds,arm='Separately_tuned_'+mode,trial_id=tid,metrics_percent={m:mean(datasets[tid]['index'],ds,'TS_'+mode,m) for m in METRICS[:3]}))
    write(OUT/'FINAL_COMPARISONS.json',dict(status='complete',rows=finalrows))
    sensitivity=read(COMP/'analysis/PARAMETER_SENSITIVITY_1792106675.json')
    write(OUT/'PARAMETER_SENSITIVITY.json',dict(status='pass',results=sensitivity['results'],scope=sensitivity['scope'],global_importance_claim=False))
    recovery=[dict(stage='search',issue='CPU norm reader used fixed absolute threshold at large step',resolution='separate float64 norm/direction readback; saved GPU results retained, no reward/optimizer/threshold change'),dict(stage='temporal',issue='missing existing lock binding and empty-reference None schema',resolution='strict descriptor and no-reference Source fallback checks; no scientific change'),dict(stage='temporal',issue='cached fit replay depended on original BLAS lineage',resolution='111 exact baseline replays under original lineage-specific CPU environment; 17 explicit no-fit fallbacks; no relaxed tolerance'),dict(stage='native preparation',issue='legacy descriptor schema and helper/ABI namespaces',resolution='strict original SHA binding and isolated helper paths; no changed candidate or prompt')]
    write(OUT/'ENGINEERING_RECOVERY.json',dict(history=recovery,all_originals_preserved=True))
    runtime_paths=[SEARCH/'SEARCH_EXECUTION_RUNTIME006.json',COMP/'PREPARATION_RUNTIME.json',COMP/'TEMPORAL_COMPONENT_RUNTIME005.json',NATIVE/'EXECUTION_RUNTIME007.json']
    pins={}
    for p in runtime_paths:
        for rel,sha in read(p)['pins'].items():
            if rel.endswith('.py'):
                assert rel not in pins or pins[rel]==sha
                pins[rel]=sha
    source_manifest=[]
    for rel,expected in sorted(pins.items()):
        source=ROOT/rel;raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
        text=raw.decode()
        # Remove historical private attachment references, never attachment contents.
        text=re.sub(r'/home/[^/]+/\.codex/attachments/[^\s\'\"<>]+','PRIVATE_AUTHORITY_NOT_DISTRIBUTED',text)
        text=text.replace(ROOT.as_posix()+'/','./')
        ast.parse(text)
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        source_manifest.append(dict(path=rel,original_sha256=expected,public_snapshot_sha256=digest(target),private_path_references_redacted=raw!=text.encode(),role='runtime-pinned source closure; not model/data assets'))
    write(OUT/'SOURCE_BINDINGS.json',dict(status='pass',measurement_sources=SOURCE_HASHES,source_snapshot=source_manifest,source_count=len(source_manifest),source_snapshot_scope='Full runtime-pinned Python closure; five historical authority/workspace references normalized, algorithms unchanged. Private execution runtime/data are not distributed.'))
    write(OUT/'REPORT_RECEIPT.json',dict(status='complete',scope='measurement publication; no method promotion or independent confirmation',source_count=128,per_domain=64,coordinate_slots=42,spatial_trials=16,temporal_deletions=4,native_control=1,logical_score_rows=rows_total,trials=trials,summary_max_error=max(ERRORS),all_positive_negative_neutral_unknown_preserved=True,bootstrap_executed=False,whole_study_FINAL_complete=False,new_GT_read=False,new_inference=False))
    build_figures(datasets,finalrows,contrasts,sensitivity)
    build_report(finalrows,contrasts,sensitivity,rows_total)
    print(json.dumps(dict(status='exported',trial_blocks=len(trials),logical_rows=rows_total,summary_max_error=max(ERRORS),source_snapshot_files=len(source_manifest),private_path_normalized_files=sum(x['private_path_references_redacted'] for x in source_manifest))))


def build_figures(datasets, finalrows, contrasts, sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    names=['Frozen','Temporal','Spatial_Native_clock','Full'];labels=['Frozen','Temporal only','Spatial only\nNative clock','Temporal + Spatial']
    for ax,ds,title in zip(axes,['hc2','vidstg'],['HC-STVG-v2 (64 development sources)','VidSTG (64 development sources)']):
        vals=[next(r['metrics_percent']['m_vIoU'] for r in finalrows if r['dataset']==ds and r['arm']==name) for name in names]
        bars=ax.bar(range(4),vals,color=['#999999','#4477aa','#ee9977','#228866'])
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_ylabel('Mean vIoU (%)');ax.set_title(title,fontsize=11);ax.set_ylim(0,max(vals)*1.17)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.25,f'{v:.3f}',ha='center',fontsize=10)
    fig.suptitle('Sealed whole-branch control: rebuilding Native-clock spatial evidence',fontsize=13)
    fig.savefig(OUT/'WHOLE_BRANCH_COMPARISON.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
    variants=[read(COMP/'trials'/f'A{i:03}'/'DESIGN_LOCK.json')['variant'] for i in range(1,16)]
    for ax,ds in zip(axes,['hc2','vidstg']):
        vals=[next(r['mean_delta_pp'] for r in contrasts if r['dataset']==ds and r['contrast']==f'A{i:03} minus A000' and r['metric']=='m_vIoU') for i in range(1,16)]
        ax.barh(range(15),vals,color=['#228866' if v>=0 else '#bb5544' for v in vals]);ax.set_yticks(range(15),variants,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='#555555',lw=.8);ax.set_xlabel('Delta vIoU vs full M2 (percentage points)');ax.set_title(ds+' / 64 development sources')
    fig.suptitle('Spatial component deletions: fixed final parameters, all sources retained')
    fig.savefig(OUT/'SPATIAL_ABLATIONS.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6.7),layout='constrained')
    for ax,key in zip(axes.flat,['mu','lr','sigma','tau','steps','response_ratio']):
        for ds,color in [('hc2','#4477aa'),('vidstg','#bb5544')]:
            block=next(r for r in sensitivity['results'] if r['coordinate']==key and r['dataset']==ds)
            pts=block['points'];best=max(p['viou_percent'] for p in pts)
            ax.plot([p['value'] for p in pts],[p['viou_percent']-best for p in pts],'.-',label=ds,color=color,ms=4)
        if key not in ['steps','response_ratio']:ax.set_xscale('log')
        ax.set_title(key);ax.set_ylabel('vIoU minus coordinate best (pp)',fontsize=8);ax.legend(fontsize=8)
    fig.suptitle('Conditional sensitivity: different coordinate ranges, not global importance')
    fig.savefig(OUT/'PARAMETER_SENSITIVITY.png',dpi=180);plt.close(fig)


def build_report(finalrows,contrasts,sensitivity,rows_total):
    def fmt(ds,arm):return f"{next(r['metrics_percent']['m_vIoU'] for r in finalrows if r['dataset']==ds and r['arm']==arm):.6f}"
    lines=['# Coordinate128 parameter search and hierarchical component ablations','','Measurement publication, 2026-10-16. All 128 sources (HC64 + Vid64), one query per parent, have been used for development. These are not independent confirmation or Table 1 results. No bootstrap was run. No CURRENT promotion or automatic full evaluation was performed.','','## Whole-branch comparison','','| Arm | HC vIoU (%) | VidSTG vIoU (%) |','|---|---:|---:|']
    for name in ['Frozen','Temporal','Spatial_Native_clock','Spatial_QA_clock','Full','Separately_tuned_M0','Separately_tuned_M1','Separately_tuned_M2']:lines.append(f'| {name} | {fmt("hc2",name)} | {fmt("vidstg",name)} |')
    lines+=['','Spatial_Native_clock truly removes Temporal: it rebuilds candidate evidence at Source Native times, runs the same frozen R1 protocol, and performs the fixed final M2 update before full Student re-decoding. Spatial_QA_clock retains the original Temporal QA observation clock and only emits Native time; it is a conditional output control, not a complete removal of Temporal.','','Full minus true Native-clock Spatial is +5.231751 pp on HC and +0.262229 pp on VidSTG. On the original QA-clock spatial predictions, changing the output time contributes +5.075218 / -0.027099 pp. QA-clock versus Native-clock spatial evidence with Native output differs by +0.156533 / +0.289328 pp. These are descriptive matched-arm comparisons, not an additive unique causal identification. VidSTG Temporal output has no positive increment in this comparison; the observation clock still helps the spatial path on average.','','Native control executed 512 R1 jobs, produced 197 non-Source anchors across 95 sources, and performed 95 real M2 fits / 95 full final Student readouts / 349 optimizer steps. The other 33 sources retain the prescribed no-reference fallback. Four arm records per source are not four independent trainings.','','Full M2 versus Temporal-only improves +0.339858 / +1.172098 pp. Separately tuned M2 versus separately tuned M0 improves only +0.012469 / +0.213753 pp; HC is effectively a near tie in this development study, and no statistical significance claim is made. M0 uses the same current Weight-free/R1 references, not the historical DINO population.','','## Spatial ablations','','| Deletion / change | HC delta vIoU (pp) | Vid delta vIoU (pp) |','|---|---:|---:|']
    for i in range(1,16):
        variant=read(COMP/'trials'/f'A{i:03}'/'DESIGN_LOCK.json')['variant']
        vals=[next(r['mean_delta_pp'] for r in contrasts if r['dataset']==ds and r['contrast']==f'A{i:03} minus A000' and r['metric']=='m_vIoU') for ds in ['hc2','vidstg']]
        lines.append(f'| A{i:03} {variant} | {vals[0]:+.9f} | {vals[1]:+.9f} |')
    lines+=['','Deleting current analytic policy-KL and extra equal-norm diagnostics individually preserved all 128 saved boxes/intervals in the earlier root tensor readback. This does not prove arbitrary combined deletions equivalent. The flat-reward skip also preserved predictions but its removal added 39 proposals, so it should remain for efficiency. Equal reliability and no control variate change means by only about 1e-4 pp or less, not a meaningful improvement. Query/LN, paired samples, reference admission, and decoder-response direction have supporting deletion evidence in this scope. Runtime speedups from code deletion have not been benchmarked.','','## Temporal component controls','','These four variants condition on fixed full-M2 spatial boxes and the original QA-clock references; they replay the saved 512-parameter temporal interface on CPU, not a fresh visual-model forward per arm. The current interface is Offset-Trust / L-BFGS, not the historical Adam recipe. Original baseline replay is exact for 111 fitted caches; 17 no-fit cases remain explicit Native fallbacks.','','| Temporal deletion | HC delta vIoU (pp) | Vid delta vIoU (pp) |','|---|---:|---:|']
    for arm in ['TS_no_grid_trust','TS_no_temporal_l2','TS_shared_offset_candidate','TS_projection_only']:
        vals=[next(r['mean_delta_pp'] for r in contrasts if r['dataset']==ds and r['contrast']==arm+' minus full_temporal' and r['metric']=='m_vIoU') for ds in ['hc2','vidstg']]
        lines.append(f'| {arm} | {vals[0]:+.9f} | {vals[1]:+.9f} |')
    lines+=['','HC Projection-only is better than the full temporal adapter in this conditional comparison; VidSTG is worse. L2 and offset sharing are small and mixed across domains. Removing full-grid trust causes a substantial loss in both domains. The user-proposed combined simplification was explicitly held as an idea and has NOT been implemented or run.','','## Parameter sensitivity','','Each coordinate fixes the other parameters of its stage. The following range is max minus min across measured values, not a normalized global importance ranking.','','| Coordinate | HC measured spread (pp) | Vid measured spread (pp) |','|---|---:|---:|']
    for key in ['lr','response_ratio','steps','sigma','mu','tau']:
        vals=[next(r['spread_pp'] for r in sensitivity['results'] if r['coordinate']==key and r['dataset']==ds) for ds in ['hc2','vidstg']]
        lines.append(f'| {key} | {vals[0]:.6f} | {vals[1]:.6f} |')
    lines+=['','Vid learning rate is most sensitive in the measured ranges; rates up to 1.5 did not win. Sigma winners lie at the tested upper boundaries and are not global optima. All three applicable families receive the same coarse coordinate opportunities and frozen neighbor refinement. Configuration slots include exact fit reuse and must not be counted as 42 independent GPU reruns.','','## Protocol, evidence, and limitations','','Student: cross-domain TA-STVG from original source checkpoints, 256 Query residual + 1536 LN spatial parameters, episodic Source restore / Query and Adam reset. The writeback fields identify historical dataset configuration but do not enable cross-query LN carry here. Current teachers/candidate support remain fixed during coordinate search and spatial deletion; the Native control changes the observation clock by design. Frozen R1 evaluates Weight-free proposals; no new YOLO/SAM/DINO/tracker was introduced in this study.','','All candidate formation, teacher selection, parameter updates, and update acceptance are GT-free. Saved predictions are globally sealed and independent saved-witness checks passed before official labels were used for offline scoring. Development GT is used to choose hyperparameters and therefore this is target-development tuning, not a blind target evaluation. Saved source/state/reward/loss/Adam/solver readbacks do not repeat a full independent model backward.','','Five metrics and their unknown values are included for every logical row. Supervised/unsupervised sIoU compares the declared per-arm support; Native-clock and QA-clock reference supports differ, so those support-specific averages must not be presented as identical-support propagation comparisons. The complete source-level positive/negative/neutral/unknown data are retained. No cherry-picked query removal, GT-selected online steps, or GT-generated references were used.','','All engineering failures and partial outputs remain locally preserved. ENGINEERING_RECOVERY.json describes the corrected dispatch/binding/schema/CPU-lineage issues; none is converted into scientific success by a root pass. Earlier role-prompt P1 was separately completed and public; its P2 trigger did not fire, and the original P0 prompt remains the spatial search input.','','## Public package','','This package contains '+str(rows_total)+' logical scored arm rows across 42 coordinate slots, 16 spatial trials, 9 conditional temporal readouts, and 4 Native-clock readouts. Shared/reused baselines are present for audit and are not independent observations. It includes full per-domain summaries, paired contrasts with unknowns, final configurations, every trial configuration, execution counters, source bindings, and the runtime-pinned Python source snapshot. Five historical workspace/attachment path references in the broad source closure are normalized for privacy; original and public SHA256 are both recorded. No attachment contents, private media/annotations, weights, checkpoint tensors, prompts containing real query/caption text, or raw model responses are published.','','To inspect: start with FINAL_COMPARISONS.json, PARAMETER_SENSITIVITY.json, CROSS_TRIAL_PAIRED.json and each trials/*/ROWS.json. source_snapshot/ mirrors the original repository paths for source review; it is not a standalone data/checkpoint distribution. Restoring an execution environment still requires the original authorized private data, caches, locks and checkpoints. The exporter itself is a no-model/no-raw-GT reader and refuses incomplete measurement receipts.','','This is a completed measurement release. Full research closing and method qualification remain distinct; this publication does not claim independent confirmation, overall study FINAL, or production registration.','','![Whole-branch comparison](../results/stvg_coordinate_component128/2026-10-16/WHOLE_BRANCH_COMPARISON.png)','','![Spatial deletions](../results/stvg_coordinate_component128/2026-10-16/SPATIAL_ABLATIONS.png)','','![Conditional sensitivity](../results/stvg_coordinate_component128/2026-10-16/PARAMETER_SENSITIVITY.png)']
    DOC.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

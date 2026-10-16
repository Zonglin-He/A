"""Publish the frozen-recipe64 as its own complete group."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE as DEV,read,put,sha
BASE=DEV/'confirmation'
def main():
    from scripts.run_stvg_instance_confirmation64_v1 import module,verify
    verify();out=ROOT/'results/stvg_instance_evidence32/2026-10-16/confirmation001';assert not out.exists();out.mkdir(parents=True)
    d=read(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json');cpu=read(BASE/'EXPERIMENT_CPU_COMPLETION.json');assert cpu['status']=='complete'and cpu['count']==64 and cpu['result_sha256']==sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json')
    s,p=module('score_stvg_instance32_v1.py').aggregate(d['rows']);assert s==d['summary']and p==d['paired']and len(d['rows'])==320
    alias={c:f'C{i:03}'for i,c in enumerate(sorted({r['case_id']for r in d['rows']}),1)};pp=copy.deepcopy(p)
    for r in pp:r['per_source_delta_pp']={alias[k]:v for k,v in r['per_source_delta_pp'].items()}
    put(out/'RESULTS.json',dict(status='complete',count=64,rows=[dict(r,case_id=alias[r['case_id']])for r in d['rows']],summary=s,paired=pp,GT_read=True,scope='fixed64, current32-disjoint; previous project exposure disclosed; no confirmation retuning',bootstrap=False))
    usage=[dict(r,case_id=alias[r['case_id']])for r in d['usage']];geo=[dict(r,case_id=alias[r['case_id']])for r in d['geometry']]
    put(out/'REFERENCE_AND_UPDATE_DIAGNOSIS.json',dict(status='complete',count=64,usage=usage,geometry=geo,GT_read=True,identity='not independently annotated; box overlap does not certify identity'))
    actual=read(BASE/'ACTUAL_GRADIENT_SUPPORT_DIAGNOSIS.json')
    put(out/'ACTUAL_GRADIENT_SUPPORT.json',dict(status='complete',count=64,rows=[dict(r,case_id=alias[r['case_id']])for r in actual['rows']],summary=actual['summary'],GT_read=True,common_metric_note=actual['common_metric_note'],complement=actual['complement']))
    put(out/'COHORT_ADAPTER_READBACK.json',read(BASE/'COHORT_ADAPTER_READBACK.json'))
    put(out/'CONFIRMATION_DECISION.json',read(BASE/'CONFIRMATION_DECISION.json'))
    costs={f.parent.name:read(f)['wall_seconds']for f in sorted((BASE/'execution').glob('*/WORKER_EXIT.json'))}
    costs['final_binding002_CPU_recovery']=read(BASE/'recovery/final_binding002/CPU_COMPLETION.json')['wall_seconds']
    tb=read(BASE/'temporal/TEACHER_GLOBAL_BARRIER.json')
    put(out/'CONFIG_AND_COST.json',dict(status='complete',count=64,recipe=read(BASE/'FROZEN_RECIPE_LOCK.json'),actual_designs={t:read(BASE/'trials'/t/'DESIGN_LOCK.json')for t in ('E0','E1')},new_hypothesis=read(BASE/'EVIDENCE_MODIFICATION_LOCK.json'),stage_wall_seconds=costs,Temporal_caption_alignment_pairs=tb['response_count'],Temporal_new_pairs=tb['new_caption_alignment_pairs'],Temporal_exact_corrected_reused_pairs=tb['exact_P0_reused_pairs'],Temporal_actual_fits=read(BASE/'temporal/ROOT_READBACK.json')['fit_exact_count'],Temporal_original_fallbacks=read(BASE/'temporal/ROOT_READBACK.json')['no_fit_count'],spatial_teacher_requests=read(BASE/'spatial/ROOT_TEACHER_READBACK.json')['response_count'],actual_Spatial_fits={a:sum(r['actual_fit']for r in usage if r['arm']==a)for a in ('E0','E1','E2')},nonzero_Spatial_updates={a:sum(r['nonzero_parameter_update']for r in usage if r['arm']==a)for a in ('E0','E1','E2')},inherited_cost='Source and original caches are reused, not zero raw-deployment cost'))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=64,row_count=320,all_metrics_summaries_pairs_reaggregated=True,all_positive_negative_neutral_unknown_preserved=True,source_result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json'),prediction_barrier_sha256=sha(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json'),root_locked_fit_binding_sha256=sha(BASE/'ROOT_LOCKED_FIT_BINDING.json'),recipe_sha256=sha(BASE/'FROZEN_RECIPE_LOCK.json'),GT_read=True))
    coverage=[]
    for ds in ('hc2','vidstg'):
        for group in ('E0','E1'):
            allw=[w for w in geo if w['dataset']==ds and w['group']==group];known=[w for w in allw if w['known_GT']];pool=[max((c['GT_iou']for c in w['candidates']if c['role']=='proposal'and c['GT_iou']is not None),default=None)for w in known];selected=[c['GT_iou']for w in known for c in w['candidates']if c['selected']and c['GT_iou']is not None]
            source=[next(c['GT_iou']for c in w['candidates']if c['role']=='Source')for w in known];full=[max(c['GT_iou']for c in w['candidates']if c['GT_iou']is not None)for w in known]
            coverage.append(dict(dataset=ds,group=group,known_windows=len(known),unknown_windows=len(allw)-len(known),Source_ge_0_5=sum(v>=.5 for v in source),full_pool_ge_0_5=sum(v>=.5 for v in full),mean_full_pool_including_Source=math.fsum(full)/len(full)if full else None,nonSource_ge_0_5=sum(v is not None and v>=.5 for v in pool),mean_pool_max=math.fsum(v for v in pool if v is not None)/sum(v is not None for v in pool)if any(v is not None for v in pool)else None,mean_R1_selected=math.fsum(selected)/len(selected)if selected else None))
    put(out/'POOL_GEOMETRY_SUMMARY.json',dict(status='complete',count=64,records=coverage,GT_read=True,scope='exact common observation positions, not full Tube or identity truth'))
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for name in ('run_stvg_instance_confirmation64_v1.py','continue_stvg_instance_confirmation002.py','report_stvg_instance_confirmation64_v1.py','diagnose_stvg_instance_actual_support_v1.py'):
        f=ROOT/'scripts'/name;q=out/'code'/name;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(f.read_bytes()));bindings.append(dict(name=name,source_sha256=sha(f),public_sha256=sha(q)))
    put(out/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings,runtime_sha256=sha(BASE/'RUNTIME.json'),uses_original_published32_model_fitters_and_complete_readout=True))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,('hc2','vidstg')):
        for i,c in enumerate(('E0_minus_Temporal','E1_minus_Temporal','E1_minus_E0','E1_minus_E2')):
            row=next(r for r in p if r['dataset']==ds and r['comparison']==c and r['metric']=='m_vIoU');vv=list(row['per_source_delta_pp'].values());ax.scatter([i]*len(vv),vv,s=15,alpha=.6);ax.scatter([i],[row['mean_delta_pp']],marker='_',s=500,color='black',zorder=4)
        ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(4),['E0 - T','E1 - T','E1 - E0','E1 - E2']);ax.set_title(ds+' fixed32');ax.set_ylabel('Per-source vIoU change (pp)')
    fig.suptitle('Frozen recipe on64 fixed sources; no retuning');fig.tight_layout();fig.savefig(out/'confirmation_effects.png',dpi=160);plt.close(fig)
    table='\n'.join('|'+ds+'|'+a+'|'+f"{100*next(r['mean']for r in s if r['dataset']==ds and r['arm']==a and r['metric']=='m_vIoU'):.6f}"+'|'for ds in ('hc2','vidstg')for a in ('Frozen','Temporal','E0','E1','E2'))
    pairs='\n'.join('|'+r['dataset']+'|'+r['comparison']+'|'+f"{r['mean_delta_pp']:+.6f}"+'|'+f"{r['positive']}/{r['negative']}/{r['neutral']}"+'|'for r in p if r['metric']=='m_vIoU')
    doc=ROOT/'docs/STVG_INSTANCE_CONFIRMATION64_001.md';doc.write_text('''# Frozen instance-evidence recipe: fixed64-source confirmation

E1 was selected only on the previously sealed32 development screen. Before this run, the recipe, parameters, prompts, candidate mechanism, corrected time interface and64-source list were frozen. The64 parents are disjoint from that32, but have historical project exposure; this is not a globally blind benchmark or official fullset. No confirmation-driven tuning or per-domain method switch occurs.

Frozen and corrected Temporal share Source geometry. E0 uses old candidates/current response-guided Lite; E1 adds the single Source-seeded foreground hypothesis; E2 uses exactly E1 references with original validated OPD. HC lr .0045/Vid .027 stay fixed. Current Lite versus original OPD differs in multiple updater details, so E1-E2 is not an isolated projection-only ablation. Models are unchanged TA-STVG and frozen STVG-R1-7B. Temporal uses the same512-parameter Lite L-BFGS; Spatial stays episodic1792. No external detector or tracker is added.

All original eligible confirmation Temporal fits replay exactly before the corrected feedback run. Fresh corrected captions are never substituted from an old-time cache. Source/final model readout, saved OPD math, config and seed bindings are independently checked. All64 across both Spatial groups are jointly sealed before any new task GT score. No-reference/no-update/rejected and missing-GT cases remain. All320 logical arm rows and all candidate diagnostics are exported; these are not320 new vision-model training runs.

| Domain | Arm | Mean vIoU (%) |
|---|---|---:|
'''+table+'''

| Domain | Contrast | Mean delta (pp) | Positive/negative/unchanged |
|---|---|---:|---:|
'''+pairs+'''

All five metrics include dense and common-observation sIoU, plus tIoU and vIoU. In the main table, inherited supervised/unsupervised metric names split the four fixed observation positions versus all other known GT frames, including off-Source-grid frames. They do not imply all four positions were used for gradients. The separate actual-gradient-support diagnostic compares each Student to Temporal on its positive-weight training anchors and the complete known-GT complement. Missing GT stays unknown. Observation-level geometry is distinct from full-Tube scores and semantic identity. Pool improvement does not automatically imply that R1 selects it or that Student updates propagate safely.

Source prior failure, foreground/background merging and candidate-slot replacement remain limitations. R1 still only sees sparse, compressed candidate views; it is not an independent dense geometric oracle. Stage wall times, actual calls/fits/nonzero updates and the full negative tail are provided. Inherited Source/candidate caches do not imply zero deployment cost. No Bootstrap, CURRENT promotion or official full evaluation is performed.

The inherited writeback numbers identify the old configuration/dataset route; this run does not carry LN state across queries. Every Source state is restored. This matched E2 control is same-reference/same-parameter episodic validated OPD, not the independently tuned old DINO/online-carry fullset method.

The cohort adapter changes cardinality/layout and uses the original float64 output conversion qualified by the32-source reader recovery. Scientific model/fitter/readout expressions stay original. Optional old per-window mask tensor hashes may be absent or stale after prospective replacement; actual mask archive and rendered NPY bindings govern inference and are checked. A generic RGB/box proposal API alone is not an empirical second-backbone adaptation result. Private media, annotation coordinates, weights, tensors, actual Query-caption and raw R1 responses remain excluded.

After both complete Student groups and independent mathematical roots passed, the final CPU binding writer found that a copy of the already locked development-rule citation was missing. The original failure is preserved. Recovery002 copies that exact metadata file, verifies its hash against the frozen development decision and runs the unchanged final CPU reader/scorer. It does not apply a development rule to choose on confirmation, alter scientific expressions or rerun any GPU fit. Failed and recovery stage costs are both supplied.

## Frozen-recipe interpretation

E1-E0 remains positive in both domains(+.078/+ .359pp), but E1-Temporal is negative in both(-.089/-.098pp). New evidence reduces the old branch's harm; it does not establish a beneficial additional Spatial branch. E1-E2 is+.011pp in HC and-.074pp in Vid, so the development updater advantage is not stable across domains. The joint E1 model still exceeds Frozen by+3.333/+1.350pp, primarily through Temporal; that must not be attributed to a proven Spatial increment.

No confirmation-dependent recipe replacement or retuning is made. This specific Source-conditioned foreground recipe is not advanced to official fullset or empirical second-backbone efficacy expansion. Available TubeDETR weights and decoder shapes were inventoried on CPU only; no second-backbone adaptation result is claimed. This is a resource/qualification decision for one finite modification, not a universal impossibility result for Spatial OPD, pixel evidence or other backbones.
''')
    paths={str(f.relative_to(ROOT))for f in out.rglob('*')if f.is_file()}|{str(doc.relative_to(ROOT)),'scripts/run_stvg_instance_confirmation64_v1.py','scripts/report_stvg_instance_confirmation64_v1.py'}
    put(DEV/'publication/confirmation001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]));print(json.dumps(dict(status='complete',files=len(paths))))
if __name__=='__main__':main()

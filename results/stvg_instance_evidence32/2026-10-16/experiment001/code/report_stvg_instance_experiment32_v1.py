"""Complete anonymous E0/E1/E2 report, including all unselected candidates."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def main():
    out=ROOT/'results/stvg_instance_evidence32/2026-10-16/experiment001';assert not out.exists();out.mkdir(parents=True)
    d=read(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json');cpu=read(BASE/'EXPERIMENT_CPU_COMPLETION.json');assert cpu['status']=='complete'and cpu['result_sha256']==sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json')
    from scripts.score_stvg_instance32_v1 import aggregate
    s,p=aggregate(d['rows']);assert s==d['summary']and p==d['paired']and len(d['rows'])==160
    alias={c:f'E{i:03}'for i,c in enumerate(sorted({r['case_id']for r in d['rows']}),1)}
    pp=copy.deepcopy(p)
    for r in pp:r['per_source_delta_pp']={alias[k]:v for k,v in r['per_source_delta_pp'].items()}
    put(out/'RESULTS.json',dict(status='complete',count=32,rows=[dict(r,case_id=alias[r['case_id']])for r in d['rows']],summary=s,paired=pp,GT_read=True,scope='exposed development; no Bootstrap; same corrected Temporal and observation positions'))
    usage=[dict(r,case_id=alias[r['case_id']])for r in d['usage']];geo=[dict(r,case_id=alias[r['case_id']])for r in d['geometry']]
    put(out/'REFERENCE_AND_UPDATE_DIAGNOSIS.json',dict(status='complete',count=32,usage=usage,geometry=geo,GT_read=True,identity='not independently annotated; box IoU is geometric evidence only'))
    designs={tid:read(BASE/'trials'/tid/'DESIGN_LOCK.json')for tid in ('E0','E1')};costs={}
    for tid in ('E0','E1'):
        for stage in ('worker','root'):
            fp=BASE/'student_execution'/tid/stage/'WORKER_EXIT.json';new=BASE/'recovery/reader_precision002/execution'/tid/stage/'WORKER_EXIT.json'
            costs[tid+'_'+stage]=read(new if new.exists()else fp)['wall_seconds']
    costs['original_E0_reader_failed']=read(BASE/'student_execution/E0/root/WORKER_EXIT.json')['wall_seconds']
    costs['spatial_teacher_worker']=read(BASE/'spatial/execution/worker/WORKER_EXIT.json')['wall_seconds'];costs['spatial_teacher_root']=read(BASE/'spatial/execution/root/WORKER_EXIT.json')['wall_seconds']
    put(out/'CONFIG_AND_COST.json',dict(status='complete',count=32,designs=designs,new_hypothesis=read(BASE/'EVIDENCE_MODIFICATION_LOCK.json'),worker_wall_seconds=costs,teacher_requests=256,corrected_teacher_exact_reuse_count=read(BASE/'spatial/CORRECTED_TEACHER_REUSE_LOCK.json')['count'],actual_fits={a:sum(r['actual_fit']for r in usage if r['arm']==a)for a in ('E0','E1','E2')},nonzero_updates={a:sum(r['nonzero_parameter_update']for r in usage if r['arm']==a)for a in ('E0','E1','E2')},E2='exact E1 references with original validated M0',complete_decoding='one actual complete-model Student readout per fit; no-reference cases exactly fall back',inherited_cost='original Source checkpoints and prior cached proposal generation not fully remeasured; not zero deployment cost'))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=32,row_count=160,complete_metrics_summary_pairs_reaggregated=True,all_candidate_selected_unselected_unknown_kept=True,source_result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json'),prediction_barrier_sha256=sha(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json'),roots={t:sha(BASE/'trials'/t/'ROOT_READBACK.json')for t in ('E0','E1')},GT_read=True))
    mask=read(BASE/'spatial/ROOT_MASK_TENSOR_READBACK003.json')
    put(out/'ENGINEERING_READBACK.json',dict(status='pass',count=32,actual_mask_archive_and_center_geometry_exact=True,optional_mask_metadata=mask['optional_field_states'],mask_reader_sha256=sha(BASE/'spatial/ROOT_MASK_TENSOR_READBACK003.json'),note='Fresh45 windows omit the optional tensor hash. E1 carries128 stale inherited optional tensor hashes after replacement. Actual archive/file/rendered NPY hashes are independently bound and used for inference; optional fields are not reuse keys. All center boxes exactly match saved masks. Original metadata is preserved.',reader_failure='float32 versus original float64 box conversion; separate002 reader only, zero GPU rerun'))
    put(out/'ADVANCEMENT_RULE_LOCK.json',read(BASE/'ADVANCEMENT_RULE_LOCK.json'))
    from scripts.export_stvg_lite_batch001 import public_source_text
    names=('stvg_instance_spatial32_v1.py','run_stvg_instance_teacher32_v1.py','run_stvg_instance_student32_v1.py','score_stvg_instance_experiment32_v1.py','score_stvg_instance32_v1.py','report_stvg_instance_experiment32_v1.py','stvg_spatial_lite_fit_v1.py','stvg_weightfree_coordinate_fit002.py','run_stvg_soft_ep_coordinate_trial003.py','audit_stvg_spatial_lite_fit002.py','review_stvg_lite_complete002.py')
    bindings=[]
    for name in names:
        f=ROOT/'scripts'/name;q=out/'code'/name;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(f.read_bytes()));bindings.append(dict(name=name,source_sha256=sha(f),public_sha256=sha(q)))
    for name in ('review_stvg_instance_student32_002.py','continue_stvg_instance_student32_002.py'):
        f=ROOT/'scripts'/name;q=out/'code'/name;q.write_text(public_source_text(f.read_bytes()));bindings.append(dict(name=name,source_sha256=sha(f),public_sha256=sha(q)))
    put(out/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings,trial_runtime_sha256={t:sha(BASE/'trials'/t/'RUNTIME.json')for t in ('E0','E1')},reader_recovery_sha256=sha(BASE/'recovery/reader_precision002/RUNTIME.json')))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,('hc2','vidstg')):
        for i,c in enumerate(('E0_minus_Temporal','E1_minus_Temporal','E1_minus_E0','E1_minus_E2')):
            row=next(r for r in p if r['dataset']==ds and r['comparison']==c and r['metric']=='m_vIoU');vv=list(row['per_source_delta_pp'].values());ax.scatter([i]*len(vv),vv,s=16,alpha=.6);ax.scatter([i],[row['mean_delta_pp']],marker='_',s=500,color='black',zorder=4)
        ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(4),['E0 - T','E1 - T','E1 - E0','E1 - E2']);ax.set_title(ds+' development16');ax.set_ylabel('Per-source vIoU change (pp)')
    fig.suptitle('Matched evidence and updater contrasts; black marks are means');fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=160);plt.close(fig)
    selected=[r for r in p if r['metric']=='m_vIoU'];table='\n'.join('|'+r['dataset']+'|'+r['comparison']+'|'+f"{r['mean_delta_pp']:+.6f}"+'|'+f"{r['positive']}/{r['negative']}/{r['neutral']}"+'|'for r in selected)
    doc=ROOT/'docs/STVG_INSTANCE_EVIDENCE32_001.md';doc.write_text('''# One matched instance-evidence experiment

The whole historical candidate audit motivated one prospective Source-seeded RGB foreground hypothesis. E0 retains old proposals and current response-guided Lite OPD. E1 replaces the last proposal slot with the new hypothesis and keeps the same updater. E2 uses exactly E1 references with original validated OPD. Frozen and corrected Temporal are included. All three use the same corrected time interface, four physical centers, Source checkpoints, seed keys and fixed HC lr .0045/Vid .027; no parameter or prompt search is performed.

The new mask uses one locked1.25 support and central half-size foreground seed, three original GrabCut iterations, complete foreground and five-frame local DIS propagation. Source remains ID1 and other first-eight IDs stay unchanged. This is Source-conditioned evidence, not independent object discovery. It may merge background or remove a useful old last candidate, and cannot reliably recover a referent outside a wrong Source prior. Invalid candidates retain the original pool.

The32 sources are previously exposed development, including12 result-stratified diagnostic sources. Both complete Student groups pass independent saved rollout/Adam/solver/acceptance and actual full-model final-box checks before joint prediction sealing and GT scoring. No-reference/no-update sources remain in the denominator. Geometry measurements use exact observed GT positions only; missing annotations remain unknown. Candidate-pool quality is not full-Tube vIoU, and low box IoU is not an identity annotation.

## Complete vIoU paired results

| Domain | Contrast | Mean delta (pp) | Positive/negative/unchanged |
|---|---|---:|---:|
'''+table+'''

E1-E0 isolates this evidence modification; E1-E2 isolates the updater under identical new references. All160 five-arm rows, supervised/unsupervised metrics, all candidate scalar diagnostics, update admission/nonzero counts, costs, negative and unchanged sources are supplied. The independent saved mathematics audit does not claim a second complete backward pass. Historical source/proposal costs are inherited and are not zero raw-deployment cost.

E1-E2 is a complete updater comparison, not an isolated projection-direction ablation: current Lite also differs from original validated OPD in weighting/control-variate/acceptance details. Independent projection benefit cannot be attributed from this contrast alone.

The first saved-output reader converted float32 boxes directly while the original complete-model arm conversion uses float64. This caused an exact-array check failure after the E0 GPU group completed successfully. The original output, source and failure are preserved. A separate reader applies exactly the original float64 conversion; all other reader AST expressions and tolerances remain unchanged. E0 GPU is not rerun. The continuation executes E1 once and independently checks both complete groups.

Prospective input metadata also retains an unused optional old tensor hash in E1 windows. Independent readback records128 such stale optional hashes and45 fresh E0 windows where the optional hash is absent. All actual mask archives, rendered NPY file/tensor hashes and center geometry are checked. This metadata defect does not change generated inputs or reuse any response; original fields and the explicit correction receipt remain available.

No automatic confirmation, cross-backbone result or fullset claim follows from a completed measurement. The final decision receipt must state whether a shared recipe has evidence to justify confirmation. A generic RGB/box API is not an empirical second-backbone experiment. CURRENT remains unchanged. Private media, GT coordinates, weights, tensors, real Query-caption and raw Teacher responses are excluded. This is a finite evidence test, not a new tracking model or another optimization sweep.
''')
    paths={str(x.relative_to(ROOT))for x in out.rglob('*')if x.is_file()}|{str(doc.relative_to(ROOT)),'scripts/run_stvg_instance_student32_v1.py','scripts/run_stvg_instance_teacher32_v1.py','scripts/stvg_instance_spatial32_v1.py','scripts/score_stvg_instance_experiment32_v1.py','scripts/report_stvg_instance_experiment32_v1.py'}
    put(BASE/'publication/experiment001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=f,sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size)for f in sorted(paths)]));print(json.dumps(dict(status='complete',files=len(paths))))
if __name__=='__main__':main()

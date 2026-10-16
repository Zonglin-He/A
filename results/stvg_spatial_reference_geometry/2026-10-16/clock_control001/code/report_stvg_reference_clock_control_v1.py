"""Anonymous complete Spatial-clock control, including its negative outcome."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_geometry_audit_v1 import BASE as STUDY,read,put,sha
BASE=STUDY/'clock_control';OUT=ROOT/'results/stvg_spatial_reference_geometry/2026-10-16/clock_control001';DOC=ROOT/'docs/STVG_SPATIAL_CLOCK_CONTROL001.md'

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    cpu=read(BASE/'CPU_COMPLETION.json');score=read(BASE/'POSTSEAL_SCORE.json');pred=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');teacher=read(BASE/'ROOT_TEACHER_READBACK.json')
    assert cpu['status']=='complete'and cpu['count']==pred['count']==12 and cpu['GT_read']is True and cpu['score_sha256']==sha(BASE/'POSTSEAL_SCORE.json')
    assert score['prediction_barrier_sha256']==sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')and pred['root_sha256']==sha(BASE/'trials/C_CLOCK/ROOT_READBACK.json')and pred['teacher_root_sha256']==sha(BASE/'ROOT_TEACHER_READBACK.json')
    clock=read(STUDY/'CLOCK_INTERFACE_AUDIT.json');ids=sorted({r['case_id']for r in clock['records']if r['cohort']=='development'});alias={c:f'D{i:03}'for i,c in enumerate(ids,1)}
    rows=[dict(r,case_id=alias[r['case_id']])for r in score['rows']];paired=[]
    for r in score['paired']:
        q=dict(r,per_source_delta_pp={alias[c]:v for c,v in r['per_source_delta_pp'].items()});assert abs(math.fsum(q['per_source_delta_pp'].values())/q['count']-q['mean_delta_pp'])<1e-12;paired.append(q)
    for s in score['summary']:
        vals=[r['metrics'][s['metric']]for r in rows if r['dataset']==s['dataset']and r['arm']==s['arm']and r['metrics'][s['metric']]is not None]
        assert len(vals)==s['valid']and((s['mean']is None and not vals)or abs(math.fsum(vals)/len(vals)-s['mean'])<1e-12)
    put(OUT/'POSTSEAL_RESULTS.json',dict(status='complete',count=12,rows=rows,summary=score['summary'],paired=paired,GT_read=True,scope=score['selection_scope'],common_support=score['common_support'],legacy_M0_scope=score['legacy_M0_scope'],unknowns_preserved=True))
    selection=[{k:(alias[v]if k=='case_id'else v)for k,v in r.items()if k not in ('old_response_sha256','new_response_sha256')}for r in teacher['records']]
    put(OUT/'WHO_SELECTION_CONTROL.json',dict(status='complete',count=48,records=selection,actual_prefill_positions_changed=sum(r['time_positions_changed']for r in selection),ID_changes=sum(r['ID_changed']for r in selection),unchanged_prefill_same_ID=all(not r['ID_changed']for r in selection if not r['time_positions_changed']),GT_read=False,video_prompt_candidate_bindings_exact=True))
    t=read(BASE/'execution/WORKER_EXIT.json');s=read(BASE/'student_execution/worker_EXIT.json');r=read(BASE/'student_execution/root_EXIT.json');bar=read(BASE/'trials/C_CLOCK/STUDENT_GLOBAL_BARRIER.json');assert t['exit_code']==s['exit_code']==r['exit_code']==0
    costs=dict(status='complete',new_WHO_calls=48,teacher_worker_wall_seconds=t['wall_seconds'],actual_Student_fits=1,actual_full_Student_decodes=1,exact_geometry_source_parameter_reuses=11,accepted_optimizer_steps=bar['mode_optimizer_steps']['M2'],student_worker_wall_seconds=s['wall_seconds'],Student_model_worker_inner_wall_seconds=bar['wall_seconds'],student_root_wall_seconds=r['wall_seconds'],GT_Oracle_fits=0,wall_cost_scope='actual bounded workers; excludes prior candidate preparation and inherited Temporal; no matched deployment speed claim',geometry_variants_run=0,new_confirmation_sources_run=0)
    put(OUT/'ACTUAL_COST.json',costs)
    put(OUT/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=12,rows_checked=len(rows),all_summary_and_paired_reaggregated=True,unknown_metric_values=sum(v is None for r in rows for v in r['metrics'].values()),source_student_root=read(BASE/'trials/C_CLOCK/ROOT_READBACK.json'),teacher_root_sha256=sha(BASE/'ROOT_TEACHER_READBACK.json'),prediction_seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),score_sha256=sha(BASE/'POSTSEAL_SCORE.json'),GT_read=True,whole_method_frozen=False,geometry_patch_qualified=False))
    from scripts.export_stvg_lite_batch001 import public_source_text
    sources=[]
    for name in ['stvg_reference_clock_student_v1.py','score_stvg_reference_clock_control_v1.py','report_stvg_reference_clock_control_v1.py','audit_stvg_spatial_lite_fit002.py','stvg_spatial_lite_fit_v1.py','run_stvg_soft_ep_coordinate_trial003.py']:
        p=ROOT/'scripts'/name;dest=OUT/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(public_source_text(p.read_bytes()));sources.append(dict(name=name,original_sha256=sha(p),public_sha256=sha(dest)))
    put(OUT/'SOURCE_BINDINGS.json',dict(status='pass',records=sources,recipe='same original Temporal-Lite outputs and final Spatial Lite recipe; Spatial mRoPE dtype only',teacher_interface_source_commit='e1f0756b06ab5c4fcf9d8749695cddceaffa8ade',old_scientific_sources_unchanged=True))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));ds=['hc2','vidstg'];vals=[next(p['mean_delta_pp']for p in paired if p['dataset']==d and p['comparison'].endswith('minus_Legacy_Full'))for d in ds]
    ax.bar(['HC (6 sources)','VidSTG (6 sources)'],vals,color=['#b94d4d','#4c7d99']);ax.axhline(0,color='black',lw=.7);ax.set_ylim(-.15,.03);ax.set_ylabel('Corrected WHO clock − legacy Full vIoU (pp)');ax.set_title('Stratified diagnostic panel, not confirmation');fig.tight_layout();fig.savefig(OUT/'clock_control.png',dpi=160);plt.close(fig)
    DOC.write_text('''# Corrected Spatial WHO time: finite matched control

Forty-eight fresh STVG-R1 calls on twelve previously exposed development sources are complete. All historical videos, prompts, candidate IDs/geometry, FPS, weights and generation settings are fixed. Only the isolated fractional-time coercion changes. Historical Temporal-Lite output intervals and physical centers stay fixed. This is not a full-method corrected-Temporal comparison.

Actual prefill positions change for 36/48 requests. Only one selected ID changes: one HC background region switches to Source. All Vid selections stay the same. The 12 requests whose positions did not change also preserve the same selected ID. These actual-input controls do not establish that correcting Temporal caption generation has no effect; that was not measured here.

The unchanged eleven sources reuse their exact previous M2 fit and complete Student predictions after geometry/source/configuration/seed-key checks. They are not eleven fresh GPU runs. The single changed-reference source receives one real 1792-parameter fit with five accepted updates and one complete Student re-decode. Independent saved rollout/loss/Adam/response-solver/rollback checks pass, and the full model's final boxes exactly match the saved decoder output. The whole twelve-source group is sealed before this control's raw-GT scoring.

| Corrected Spatial clock Full minus legacy Full | Mean vIoU change | Paired signs |
|---|---:|---|
| HC, 6 sources | −0.121771 pp | 0 positive / 1 negative / 5 unchanged |
| VidSTG, 6 sources | 0.000000 pp | 0 positive / 0 negative / 6 unchanged |

The one changed HC source loses 0.730626 pp relative to its legacy Full result. Removing a visibly poor background reference does not guarantee better complete-Tube propagation under the remaining reference. A reference can be semantically poor while its parameter effect accidentally helps some geometry. This result does not validate learning from bad references, nor does it justify retaining defective time arithmetic for a higher score.

The panel contains two positive, two negative and two unchanged spatial outcomes per domain, selected from sealed development128. Its means are diagnostic, not a representative performance estimate or a replacement for the earlier fixed64 confirmation. Every Frozen, Temporal-Lite, legacy Full, corrected-clock Full and legacy M0 result, including negative/tied/null values, is exported. Legacy M0 uses the historical reference, so it is not claimed as a new corrected-reference same-information M0 control. All arms use the same four original observation positions when comparing supervised/unsupervised metrics.

Actual teacher worker wall is 59.0405 s for the 48 calls, including worker initialization. Actual one-source Student worker wall is 20.8669 s; the independent saved-math root worker is 1.5822 s. These costs exclude inherited candidates/Temporal and do not support a cold deployment speed claim. No new model, tracking algorithm, prompt/LR search, Oracle fit, Bootstrap, CURRENT promotion or official full-set run occurred.

## Decision

The historical geometry patch is not qualified: none of the 22 adopted component masks exactly replays, and visual decomposition does not show correct-instance truncation dominating errors. R0/R1/R2 historical fitting is therefore not run. The isolated Spatial time correction does not solve Vid identity selection or its spatial losses. There is no evidence-qualified new spatial recipe to freeze and expand to confirmation in this finite branch; no extra confirmation sources are consumed.

The unresolved next mechanism is obtaining instance-specific spatial evidence rather than a background/part region endorsed by an ID decision. No new candidate generator or identity prompt is silently chosen here. The earlier whole-scene and pairwise-rejection negative findings remain intact; this control is not a renamed rerun of either. The correction is an interface repair, not a new paper contribution. Historical old-protocol results remain labeled and preserved, with no cross-version cache relabeling.

See the preceding [geometry decomposition](STVG_SPATIAL_REFERENCE_GEOMETRY_AUDIT001.md) for the twelve-source visual review, all provenance failures and the broader time-interface audit. Private media, annotation coordinates, weights/tensors, actual Query/caption text and raw teacher outputs are not included in this publication.
''')
    paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()}|{str(DOC.relative_to(ROOT)),'scripts/stvg_reference_clock_student_v1.py','scripts/score_stvg_reference_clock_control_v1.py','scripts/report_stvg_reference_clock_control_v1.py'}
    put(STUDY/'publication/clock_control001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]))
    print(json.dumps({'status':'complete','files':len(paths),'rows':len(rows),'delta_pp':vals,'costs':costs}))

if __name__=='__main__':main()

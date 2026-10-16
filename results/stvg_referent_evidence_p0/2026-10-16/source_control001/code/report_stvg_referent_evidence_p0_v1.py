"""Anonymous complete-sign exports; no real text, media, vectors or GT boxes."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,CONTROL,PRIOR,read,put,sha
def write_common(group,out,doc,source_files):
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for name in source_files:
        p=ROOT/'scripts'/name;q=out/'code'/name;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(p.read_bytes()));bindings.append(dict(name=name,original_sha256=sha(p),public_sha256=sha(q)))
    put(out/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings))
    protocol=out/'PROTOCOL.md';protocol.write_text((ROOT/'protocols/stvg_referent_evidence_p0_v1.md').read_text())
    paths={str(p.relative_to(ROOT))for p in out.rglob('*')if p.is_file()}|{str(doc.relative_to(ROOT))}
    put(BASE/'publication'/group/'EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]))
    print(json.dumps(dict(status='exported_pending_pixel_privacy_remote',group=group,files=len(paths))))
def control():
    from scripts.review_stvg_referent_evidence_p0_v1 import summary_rows
    group='source_control001';d=read(CONTROL/'POSTSEAL_RESULTS.json');assert read(CONTROL/'CPU_COMPLETION.json')['result_sha256']==sha(CONTROL/'POSTSEAL_RESULTS.json')and read(CONTROL/'ROOT_MATCHED_SUPPORT.json')['status']=='pass'
    summary,paired=summary_rows(d['rows']);assert summary==d['summary']and paired==d['paired'];alias={cid:f'R{i:03}'for i,cid in enumerate(sorted({x['case_id']for x in d['rows']}),1)}
    rows=[dict(x,case_id=alias[x['case_id']])for x in d['rows']];pp=copy.deepcopy(paired)
    for p in pp:p['per_source_delta_pp']={alias[c]:v for c,v in p['per_source_delta_pp'].items()}
    out=ROOT/'results/stvg_referent_evidence_p0/2026-10-16'/group;out.mkdir(parents=True,exist_ok=False)
    put(out/'RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=pp,GT_read=True,aggregation='one Query per parent; same sixteen sources per domain; exposed development only',all_signs_nulls_preserved=True))
    cost=[dict(x,case_id=alias[x['case_id']])for x in d['cost']];trial=CONTROL/'trials/SOURCE';design=read(trial/'DESIGN_LOCK.json')
    put(out/'CONFIG_AND_COST.json',dict(status='complete',count=32,reference_count=read(CONTROL/'ROOT_MATCHED_SUPPORT.json')['reference_count'],actual_fit_count=sum(x['actual_fit']for x in cost),accepted_steps=sum(x['accepted_steps']for x in cost),nonzero_final_queries=sum(x['nonzero_parameter_update']for x in cost),per_source=cost,configs=design['configs'],response_settings=design['response_settings'],optimizer='episodic Adam; Query/LN1792; Lite fullstep/full_grid',LN_carry=False,writeback_field='legacy dataset/config label only; no cross-query writeback',changed='reference coordinates only; original E1 adopted support/weights/identity choice fixed',teacher_calls=0,temporal_calls=0,worker_wall_seconds=read(BASE/'execution/control_worker/WORKER_EXIT.json')['wall_seconds'],root_wall_seconds=read(BASE/'execution/control_root/WORKER_EXIT.json')['wall_seconds'],inherited_teacher_cost='not measured here; not zero deployment cost',GT_read=False))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=32,row_count=len(rows),source_result_sha256=sha(CONTROL/'POSTSEAL_RESULTS.json'),new_saved_math_root_sha256=sha(trial/'ROOT_READBACK.json'),source_support_root_sha256=sha(CONTROL/'ROOT_MATCHED_SUPPORT.json'),all_aggregate_pairs_recomputed=True,no_new_teacher_or_semantic_selection=True))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));arms=['Frozen','Temporal','E1','Source_control']
    for j,a in enumerate(arms):ax.bar([i+(j-1.5)*.19 for i in range(2)],[100*next(x['mean']for x in summary if x['dataset']==ds and x['arm']==a and x['metric']=='m_vIoU')for ds in ('hc2','vidstg')],.19,label=a)
    ax.set_xticks([0,1],['HC exposed development16','VidSTG exposed development16']);ax.set_ylabel('Mean vIoU (%)');ax.set_title('Same E1 adopted support and weights; only reference geometry changes');ax.legend();fig.tight_layout();fig.savefig(out/'source_reference_control.png',dpi=150);plt.close(fig)
    lines=['# Source-coordinate reference control on32 development sources','',
      'The previous Source-seeded foreground study reduced old Spatial harm but did not establish an additional Spatial gain on its fixed64 confirmation. This new control isolates the untested Source-like-reference explanation. It does not reopen LR or candidate search.','',
      'All original32 development sources are retained. At every adopted E1 anchor, only the geometric reference is replaced by the exact frozen Source box. E1 positions, admission, identity selection, visibility weight, random fit key and final M2 configuration remain fixed. This is intentionally a self-reference control, not independent expert evidence or a new deployed recipe. The1792-parameter optimization is actually invoked and the complete Student re-decoded; no box substitution is scored as TTA. All23 fits reject their first full-step proposal and leave zero final parameter displacement. Source reference agreement starts at1.0. The unchanged Lite implementation retains rejection-triggered early stopping, skipping68 later rows;91 saved rows are not91 actual candidate proposals.','',
      '| Arm | HC vIoU % | VidSTG vIoU % |','|---|---:|---:|']
    for a in ('Frozen','Temporal','E0','E1','E2','Source_control'):
        vals=[100*next(x['mean']for x in summary if x['dataset']==ds and x['arm']==a and x['metric']=='m_vIoU')for ds in ('hc2','vidstg')];lines.append(f'| {a} | {vals[0]:.6f} | {vals[1]:.6f} |')
    lines+=['','All per-source five-metric results, nulls and paired signs are provided. Common observation-support sIoU is not identical to actual-gradient-support sIoU. E2 is matched-reference original validated M0, not the separately tuned historical DINO-online baseline. This study uses episodic reset; legacy writeback fields are not LN carry.','',
      'Source self-reference may still move parameters through the Gaussian reward distribution and nonlinear decoder. Any change does not establish discovery of independent object geometry. E1 versus this coordinate-only control isolates geometry at adopted E1 positions, but does not separately identify every effect of changing candidate rendering or eligibility. The16+16 sources are already exposed development, not blind confirmation or an official fullset. No Bootstrap, CURRENT promotion or tuning occurs.','',
      'The separate16-source native evidence diagnostic has its own sealed results and conditional expansion rule. Completion of this control is not completion of that study. No private videos, GT boxes, actual queries/captions, raw Teacher responses, vectors or weights are published.']
    doc=ROOT/'docs/STVG_REFERENT_SOURCE_CONTROL001.md';doc.write_text('\n'.join(lines)+'\n')
    write_common(group,out,doc,['stvg_referent_evidence_p0_v1.py','review_stvg_referent_evidence_p0_v1.py','report_stvg_referent_evidence_p0_v1.py'])
def response():
    group='response001';d=read(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json');assert read(BASE/'RESPONSE_CPU_COMPLETION.json')['result_sha256']==sha(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json');decision=read(BASE/'PHASE2_DECISION.json')
    alias={c:f'P{i:03}'for i,c in enumerate(sorted({x['case_id']for x in d['source_rows']}),1)}
    source=[dict(x,case_id=alias[x['case_id']])for x in d['source_rows']];frames=[]
    fid_alias={}
    for cid in alias:
        for i,fid in enumerate(sorted({x['physical_frame_id']for x in d['frame_rows']if x['case_id']==cid}),1):fid_alias[cid,fid]=f'F{i:03}'
    for x in d['frame_rows']:
        row={k:v for k,v in x.items()if k!='physical_frame_id'};row['case_id']=alias[x['case_id']];row['frame_alias']=fid_alias[x['case_id'],x['physical_frame_id']];frames.append(row)
    # Exact physical IDs are not exported; merged-cell/candidate membership is
    # sufficient to inspect anonymous scores, but does not reconstruct raw data.
    pairs=[dict(x,case_id=alias[x['case_id']])for x in d['pair_rows']];clips=[dict(x,case_id=alias[x['case_id']])for x in d['clips']]
    out=ROOT/'results/stvg_referent_evidence_p0/2026-10-16'/group;out.mkdir(parents=True,exist_ok=False)
    put(out/'RESULTS.json',dict(status='complete',count=16,source_rows=source,frame_rows=frames,pair_rows=pairs,clips=clips,summary=d['summary'],GT_read=True,aggregation='source macro after candidate exposures averaged within each known physical frame; adjacent transitions are candidate-local',all_unknown_preserved=True))
    put(out/'DECISION.json',decision)
    bar=read(BASE/'RESPONSE_GLOBAL_BARRIER.json');lens=[]
    import numpy as np
    for r in bar['records']:
        with np.load(ROOT/r['path'],allow_pickle=False)as a:lens.append(dict(case_id=alias[r['case_id']],candidate_id=r['candidate_id'],query_token_count=len(a['query_embedding']),caption_token_count=len(a['caption_embedding']),visual_tokens=int(a['features'].shape[0]*a['features'].shape[1]),retained_per_merged_cell=int(a['budget_per_cell']),wall_seconds=r['wall_seconds']))
    put(out/'CONFIG_AND_COST.json',dict(status='complete',count=16,actual_visual_encoder_passes=bar['visual_encoder_passes'],new_caption_alignment_calls=0,new_student_updates=0,model_load_seconds=bar['model_load_seconds'],visual_loop_wall_seconds=bar['wall_seconds'],per_clip=lens,retention_ratio=.125,retention_is_diagnostic_only=True,changed_native_sequence=False,Student_internal_attention_used=False,actual_attention_ER_CS=False,semantic_identity_verified=False,inherited_caption_generation='reused exact corrected captions; their deployment cost is not free',official_comparison='released Qwen2.5 embedding similarity used as inspiration; unlike official long-text>50 pooling this diagnostic keeps all token rows, so it is NOT an official SemVID A run',GT_read=False))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=16,clip_count=bar['visual_encoder_passes'],root_sha256=sha(BASE/'ROOT_RESPONSE_READBACK.json'),result_sha256=sha(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json'),all_scalar_signs_unknown_preserved=True,semantic_audit='fixed two SHA-ordered cases per domain; separate private review receipt'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4));strategies=['query','caption','consensus','disagreement','connection','query_swap']
    for i,(key,title)in enumerate([('target_occupancy','Selected displayed-area target occupancy'),('adjacent_supported_pair_coverage','Adjacent sampled-cell target-supported proxy pair')]):
        for j,ds in enumerate(('hc2','vidstg')):
            vv=[next(x['mean']for x in d['summary']if x['dataset']==ds and x['strategy']==s and x['metric']==key)for s in strategies];axes[i].bar(np.arange(6)+(j-.5)*.36,[100*x if x is not None else float('nan')for x in vv],.36,label=ds)
        axes[i].set_xticks(np.arange(6),strategies,rotation=30,ha='right');axes[i].set_title(title,fontsize=9);axes[i].set_ylabel('%; rectangle diagnostic, not identity');axes[i].legend()
    fig.tight_layout();fig.savefig(out/'native_response_proxy.png',dpi=150);plt.close(fig)
    lines=['# Native referent-evidence prerequisite P0','',
      'Sixteen previously exposed development sources, eight per domain, are fixed by parent/query hashes among cases with original candidate inputs. All64 original eight448 marked candidate clips and already sealed corrected-clock captions are preserved. Fresh native frozen R1 visual encoder+merger vectors are computed; no new decoder generation, Student update, pruning or bbox proposal occurs in this diagnostic.','',
      'Q/C are native visual/text embedding cosine proxies, not contextual attention or two independent expert truths. Scores are per-cell percentile ranks; consensus=min(Q,C), disagreement=abs(Q-C). All strategies retain the same diagnostic12.5% per merged temporal cell. One merged token spans two physical sampled frames, not one; letterboxing and native28px merged spatial cells are accounted for. Source marks can influence these vectors.','',
      'Connection uses reciprocal projected-feature nearest neighbours, geometric proximity and a fixed shot-break proxy. It is neither official SemVID CS nor verified same-instance association. The official analysis uses attention-instrumented Qwen3, while this model is Qwen2.5 R1. The released official long-text embedding pooling also differs from this all-token diagnostic; no result here is labelled original SemVID performance.','',
      '| Domain | Consensus minus better single-text occupancy (pp) | Connection minus query supported-pair coverage (pp) | Known sources | Gate |','|---|---:|---:|---:|---|']
    for r in decision['gate']:lines.append(f"| {r['dataset']} | {100*r['consensus_minus_best_single']:.6f} | {100*r['connection_minus_query']:.6f} | {r['known_sources']} | {r['pass_gate']} |")
    lines+=['',f"Prelocked development resource decision: **{decision['status']}**. The gate is not a significance test or a proof that all connectivity mechanisms work/fail. No result-driven gate, parameter, candidate or Prompt search was allowed.",'',
      'Target rectangle occupancy and supported feature pairs do not establish referent identity, actor/patient/recipient roles, event correctness or causal TTA gains. Query-swap and all positive/negative/unknown rows are reported. Per-source physical-frame averages avoid counting repeated candidate exposures as independent sources. Pair metrics remain local to candidate clips, with missing exact-GT times explicitly unknown. Fixed four-case private panels are reviewed separately and contain no publicly exported media or text.','',
      'The previous Source-seeded recipe stays closed. A/B/C would require positive prerequisite evidence plus qualified native token-pruning positions/cache interfaces with equal budgets. RGB masking or larger coverage would not qualify as a connectivity-only ablation. No such new method has been silently substituted.']
    doc=ROOT/'docs/STVG_REFERENT_RESPONSE_P0_001.md';doc.write_text('\n'.join(lines)+'\n')
    write_common(group,out,doc,['stvg_referent_evidence_p0_v1.py','stvg_referent_response_v1.py','review_stvg_referent_evidence_p0_v1.py','report_stvg_referent_evidence_p0_v1.py'])
if __name__=='__main__':{'control':control,'response':response}[sys.argv[1]]()

"""Report the completed bounded four-cell experiment, including null and negative results."""
import collections
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_ecld_common_v1 import BASE, read, write, sha
OUT=ROOT/'results/stvg_evidence_conditioned/2026-10-15/p1'


def main():
    assert read(BASE/'ROOT_STUDENT_READBACK.json')['status']=='pass'
    assert read(BASE/'ROOT_POSTSEAL_SCALAR_READBACK.json')['status']=='pass'
    assert read(BASE/'ACTUAL_MEASUREMENT_HANDOFF.json')['status']=='complete_pending_root_review_publication'
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['POSTSEAL_ROWS.json','POSTSEAL_SUMMARY.json','PARENT_PAIRED_DELTAS.json',
                 'ROOT_STUDENT_READBACK.json','ROOT_POSTSEAL_SCALAR_READBACK.json',
                 'TEACHER_DIAGNOSTIC_READBACK.json']:
        value=read(BASE/name)
        if name=='POSTSEAL_ROWS.json':
            for row in value['rows']:
                row['support']={k:v for k,v in row['support'].items() if not isinstance(v,list)}
            value['public_projection']='all metrics and support counts retained; physical frame/position lists remain private'
        # All exported records are anonymous scalar outcomes, no model inputs,
        # teacher raw text, annotations, media, weights or optimizer tensors.
        write(OUT/name,value)
    rows=read(BASE/'POSTSEAL_ROWS.json')['rows']
    summary=read(BASE/'POSTSEAL_SUMMARY.json')['summary']
    paired=read(BASE/'PARENT_PAIRED_DELTAS.json')['records']
    teacher=read(BASE/'TEACHER_DIAGNOSTIC_READBACK.json')
    index={(r['dataset'],r['arm'],r['metric']):r['mean'] for r in summary}
    table=[]
    for arm in ['F','T','TS_historical','TS_A','TS_B','TS_C','TS_D']:
        table.append(dict(arm=arm,hc_vIoU_percent=index['hc2',arm,'m_vIoU']*100,
                         vid_vIoU_percent=index['vidstg',arm,'m_vIoU']*100))
    write(OUT/'MAIN_TABLE.json',dict(status='complete',development_sources_per_domain=16,
        independent_confirmation=False,GT_read=True,rows=table))
    reasons=collections.Counter()
    by_control={c:collections.Counter() for c in ['base','id_permutation','visual_null']}
    for p in (BASE/'teacher').glob('*.json'):
        r=read(p)
        if r['parsed']['status']=='invalid':
            reasons[r['parse_diagnostic']]+=1
            by_control[r['control']][r['parse_diagnostic']]+=1
    write(OUT/'INVALID_PROTOCOL_OUTPUTS.json',dict(status='complete',GT_read=False,
        total=dict(reasons),by_control={k:dict(v) for k,v in by_control.items()},
        limitation='ABSTAIN evidence-label contract failures are not proof of wrong referent identification; contradictory KEEP/REPLACE and IDs were not silently repaired'))
    first=read(BASE/'execution/teacher/LAUNCH.json')
    failed_exit=read(BASE/'execution/teacher/WORKER_EXIT.json')
    handoff=read(BASE/'ACTUAL_MEASUREMENT_HANDOFF.json')
    cost=dict(status='complete',GT_read=False,actual_generations=320,accepted_prefix_reused=1,
        new_generations_in_revision002=319,cold_model_loads=2,
        revision002_pipeline_elapsed_seconds=handoff['elapsed_seconds'],
        failed_first_worker_elapsed_seconds=failed_exit['time']-first['time'],
        overall_wall_from_first_worker_launch_seconds=read(BASE/'SEQUENCE_STATUS.json')['time']-first['time'],
        student_stage_wall_seconds=read(BASE/'STUDENT_GLOBAL_BARRIER.json')['wall_seconds'],
        teacher_generate_and_request_cost=teacher['cost'],
        actual_fit_invocations=12,fits_with_accepted_optimizer_steps=10,accepted_optimizer_steps=42,
        source_capture_and_complete_decoder_readouts_included_in_student_stage=True,
        reused_stages=['weight-free candidate generation and exact input arrays','existing Temporal teacher feedback and Offset-Trust results','historical comparison predictions'],
        full_raw_video_to_final_method_cold_latency_remeasured=False,
        limitation='These are complete costs of this actual bounded cache-reusing experiment, including failed-attempt disclosure. They are not a fresh end-to-end method latency or a full Table1 estimate.')
    write(OUT/'EXECUTION_COST.json',cost)
    decision=dict(status='stop_current_P1_recipe_before_confirmation_or_full',scope='current matched closed-schema teacher plus evidence-aware OPD on exposed development32',
        implementation_readback_pass=True,new_method_qualified=False,new_teacher_correction_evidence_sufficient=False,
        freeze_new_recipe=False,confirmation_triggered=False,full13785_triggered=False,
        CURRENT_promoted=False,bootstrap_executed=False,
        reasons=['Student-conditioned B/D supplies no HC reference and only one Vid reference; no vIoU increment over Temporal in either domain',
                 'Neutral A/C produces one HC negative source; evidence-aware C reduces aggregate damage but retains the negative HC mean',
                 'Positive HC14 geometry propagation and V04 dense spatial gains remain; V04 has zero temporal overlap so full vIoU stays zero',
                 'Old global flat-reward skip already exists; this audit does not justify adding cold-start',
                 'No isolated candidate-coverage evidence here triggers new proposal generation'],
        not_a_route_level_impossibility_claim=True,time=time.time())
    write(BASE/'TECHNICAL_DECISION.json',decision)
    write(OUT/'TECHNICAL_DECISION.json',decision)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.9))
    for ax,ds,title in zip(axes,['hc2','vidstg'],['HC-STVG-v2 (16 sources)','VidSTG (16 sources)']):
        vals=[next(r['mean_delta_pp'] for r in paired if r['dataset']==ds and r['left']=='TS_'+v and r['right']=='T') for v in 'ABCD']
        bars=ax.bar(list('ABCD'),vals,color=['#ba6d57','#527da5','#b89453','#568876'])
        ax.axhline(0,color='#555555',lw=.8)
        ax.set_ylim(-.17,.045)
        for bar,value in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,value+(.006 if value>=0 else -.012),f'{value:+.3f}',ha='center',fontsize=10)
        ax.set_title(title);ax.set_ylabel('vIoU change vs Temporal-only (pp)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Matched P1: no positive dual-domain full-tube increment')
    fig.tight_layout();fig.savefig(OUT/'FOUR_CELL_RESULTS.png',dpi=170);plt.close(fig)
    groups=teacher['main_summary'];fig,ax=plt.subplots(figsize=(9.5,4.2));bottom=np.zeros(4)
    for name,color in [('KEEP','#527da5'),('REPLACE','#568876'),('ABSTAIN','#b89453'),('invalid','#ba6d57')]:
        vals=np.array([g['decisions'].get(name,0) for g in groups]);ax.bar(range(4),vals,bottom=bottom,label=name,color=color)
        for i,val in enumerate(vals):
            if val:ax.text(i,bottom[i]+val/2,str(val),ha='center',va='center',color='white',fontsize=10)
        bottom+=vals
    ax.set_xticks(range(4),['HC neutral','HC conditioned','Vid neutral','Vid conditioned']);ax.set_ylim(0,70)
    ax.set_ylabel('Main teacher requests (64 / group)');ax.set_title('Teacher decisions and closed-protocol failures')
    ax.spines[['top','right']].set_visible(False);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.14));fig.tight_layout()
    fig.savefig(OUT/'TEACHER_DECISIONS.png',dpi=170);plt.close(fig)
    lines=['# Evidence-Conditioned STVG: bounded P0/P1 completed', '',
        '**Decision: this fixed recipe is not frozen or expanded.** Complete numerical/readback checks passed, but the matched experiment did not establish useful dual-domain spatial transfer. No new confirmation or 13,785-query run was triggered; CURRENT and the paused old full run remain protected.', '',
        'The cohort is the same 32 historically exposed development sources (16/domain), one registered query per source. These are development means, not independent confirmation or Table 1 results. Temporal Offset-Trust, exact candidate geometry/pixels/clocks, source weights, 1,792 spatial parameters, five steps and the prior domain-specific settings were fixed. No new detector, segmentation model, tracker, bootstrap or GT-oracle fit was used.', '',
        '## Actual full-tube results', '',
        '| Arm | HC vIoU (%) | VidSTG vIoU (%) |', '|---|---:|---:|']
    labels={'F':'Frozen','T':'Temporal-only','TS_historical':'Historical top-1 + original OPD (separate output protocol)',
        'TS_A':'A: neutral + original OPD','TS_B':'B: conditioned + original OPD','TS_C':'C: neutral + evidence-aware OPD','TS_D':'D: conditioned + evidence-aware OPD'}
    for r in table:lines.append(f"| {labels[r['arm']]} | {r['hc_vIoU_percent']:.6f} | {r['vid_vIoU_percent']:.6f} |")
    lines += ['', 'Historical top-1 is an explicit separate control: its task/output schema differs, so its gap against A is not a one-factor student-conditioning effect. A versus B and C versus D are matched.', '',
        '![Matched outcomes](../results/stvg_evidence_conditioned/2026-10-15/p1/FOUR_CELL_RESULTS.png)', '',
        'A adds -0.135494 pp on HC relative to Temporal; C adds -0.092132 pp. B/D add exactly zero full-tube vIoU in both domains, which does not mean their parameters never changed. Both perform two accepted updates on V09, whose time interval has zero overlap and whose dense spatial score worsens. Do not call this successful safe adaptation.', '',
        'HC02 remains negative: A/C full-tube -2.181364 pp, despite reference-anchor agreement improving; supervised sIoU increases about 1.418 pp while unobserved-frame sIoU falls about 2.277 pp. This shows the present pseudo-reference gate does not protect propagation. HC14 is a retained positive: C versus Temporal +0.707253 pp, versus A +0.693798 pp; dense sIoU changes from A -0.208292 pp to C +1.076504 pp relative to Frozen. V04 retains a substantial spatial improvement (A dense sIoU +11.774665 pp; C +11.763339 pp), but both Native and fixed Temporal have tIoU=0, so final vIoU remains zero. These are distinct spatial and temporal readouts, not interchangeable scores.', '',
        '## Teacher feedback and input controls', '',
        '320 actual generations completed: 256 main requests and 64 fixed controls; one accepted prefix response was reused exactly and 319 were generated after an engineering-only cache fix. Main outputs contain 201 valid / 55 invalid observations, with no token-cap truncation. Neutral A produces 8 references across 5 sources; conditioned B produces one across one source and none on HC. The same references feed C/D respectively.', '',
        '![Teacher decisions](../results/stvg_evidence_conditioned/2026-10-15/p1/TEACHER_DECISIONS.png)', '',
        'Across all main/control outputs, 90 violate the closed protocol: 54 ABSTAIN responses do not set every evidence label to unclear, 25 KEEP responses choose a non-Source ID, and 11 REPLACE responses choose Source. Abstention-label violations are not proof of referent errors. Contradictory decisions and IDs were retained as invalid rather than repaired after viewing scores. This format/task burden and the conditioned prompt’s conservatism limit interpretation of the teacher; they do not establish that all student-conditioned feedback is useless.', '',
        'Of 32 ID-permutation comparisons, 19 have valid outputs on both sides; all 19 preserve the stable candidate. The other 13 are unknown/invalid, not counted as consistent. Visual-null produces 9 valid abstentions and 23 invalid outputs; none asserts all three grounded labels. Only 7 base/null pairs are valid on both sides, so the control supports a limited visual-sensitivity observation rather than a calibrated semantic guarantee. Four-frame context pacing retains the previously disclosed mean-gap approximation.', '',
        '## Actual optimization and independent checks', '',
        '12 fitter invocations completed; 10 have accepted updates, totalling 42 actual accepted Adam steps. Reuse/no-reference and zero-accepted-step cases remain in all 32 denominators. Complete Student decoder readouts match the cached decoder, frozen source weights restore, all 32 predictions seal before GT, and independent saved arithmetic checks 2,496 sampled rewards (maximum IoU difference 8.282e-7). No independent decoder backward replay is claimed. All 384 case-arm rows, 120 summary scalars and 40 source-paired comparisons are retained; root reaggregation checked 760 values with zero difference. Fixed-observation/non-observation sIoU uses the union of A–D reference positions, not a claim that every such frame received a nonzero gradient.', '',
        'P0 independently audited 49 old anchors / 241 anchor rounds: 214 have varying rewards and 27 zero overlap; 95 proposals were accepted and one rejected. Original code already centers weights by 1/32, skips globally flat reward rounds and applies reliability outside loss. These are not new fixes. Four missing historical transitive source pins remain a disclosed old-provenance limitation. No cold-start was added because this evidence does not show a dominant lack of reward support.', '',
        '## Cost and engineering history', '',
        f"Revision002’s complete teacher → references → Student → independent readback → scoring chain took {cost['revision002_pipeline_elapsed_seconds']:.3f} seconds. The failed first worker adds {cost['failed_first_worker_elapsed_seconds']:.3f} seconds of actual execution; wall time from its launch through the final handoff was {cost['overall_wall_from_first_worker_launch_seconds']:.3f} seconds including the engineering repair gap. Student stage took {cost['student_stage_wall_seconds']:.3f} seconds. All 320 generation calls sum to {teacher['cost']['total_generate_seconds']:.3f} model seconds / {teacher['cost']['total_request_seconds']:.3f} request seconds; both cold model loads are disclosed separately. Total input/visual/output tokens were {teacher['cost']['total_input_tokens']}/{teacher['cost']['total_visual_tokens']}/{teacher['cost']['total_output_tokens']}; maximum allocated teacher GPU memory was {teacher['cost']['maximum_GPU_peak_bytes']} bytes.", '',
        'This is the complete actual cost of the bounded cache-reusing experiment, not a fresh full-method cold latency: candidate generation, Temporal feedback/adaptation and historical predictions were reused. It cannot be extrapolated into a reliable Table1 estimate or advertised as the whole method’s runtime. No OOM or resource-guard stop occurred. The first launch failed because tensor nbytes was compared to NPY file bytes including its header; original code/log/runtime and the one response are preserved, and revision002 only fixes that check while retaining disk SHA checks. Old full/background queues were not restarted.', '',
        '## Scope decision', '',
        'Do not freeze this D recipe or open confirmation/full evaluation: its Teacher correction and usable full-tube increment have not both been demonstrated. Keep HC14 and V04 as positive mechanism evidence, HC02/V09 as failures, all no-reference and unknown cases, and the format failures. The present results do not isolate candidate coverage as the main bottleneck, so they do not trigger new proposal generation, cold-start, broad parameter search or a new teacher. Boundary verification remains a separate unexecuted mechanism, not a claimed component of this tested method.', '',
        '[Protocol](../protocols/stvg_evidence_conditioned_v1.md) · [All anonymous P1 outcomes](../results/stvg_evidence_conditioned/2026-10-15/p1) · [P0 source-support audit](../results/stvg_evidence_conditioned/2026-10-15/p0).', '']
    (ROOT/'docs/STVG_EVIDENCE_CONDITIONED_V1.md').write_text('\n'.join(lines))
    print('Report and complete anonymous outcomes ready',len(rows),len(summary),len(paired))


if __name__=='__main__':
    main()

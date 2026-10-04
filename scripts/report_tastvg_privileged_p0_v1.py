"""Durable report and publication figures from sealed anonymous P0 results."""
import sys, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_privileged_p0_common_v1 import *
import numpy as np


def pp(m):return f"{100*m['mean']:+.3f} [{100*m['ci95'][0]:+.3f}, {100*m['ci95'][1]:+.3f}]"


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.score_audit_tastvg_privileged_p0_v1 import public_check
    public_check(PUB);summary=read(PUB/'SUMMARY.json');rows=read(PUB/'ROWS.json');diag=read(PUB/'DIAGNOSTICS.json');decision=read(PUB/'DECISION.json');cost=read(PUB/'COST.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    labels=['Vid development','Vid confirmation','HC development','HC confirmation'];panels=[summary[d][s]['corruption'] for d in DATASETS for s in ['search','confirm']]
    fig,ax=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    for a,k,title in zip(ax,['delta_v','delta_s'],['Full-tube vIoU','Full-GT spatial IoU']):
        for i,p in enumerate(panels):
            m=p['metrics'][k];v=100*m['mean'];ci=100*np.array(m['ci95'])
            a.errorbar(v,i,xerr=[[v-ci[0]],[ci[1]-v]],fmt='o',capsize=3,color='#285F9C' if i<2 else '#BA6437',ms=6,lw=1.5)
        a.set_yticks(range(4),labels);a.invert_yaxis();a.axvline(0,color='.5',ls='--',lw=.8)
        a.set_xlabel('Privileged − ordinary (pp), paired source 95% CI');a.set_title(title);a.grid(axis='x',alpha=.15)
    for ext in ['png','pdf']:fig.savefig(PUB/f'P0_ADVANTAGE.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10.2,3.6),layout='constrained')
    for a,ds in zip(ax,DATASETS):
        rr=sorted([r for r in rows if r['dataset']==ds and r['condition']!='clean'],key=lambda r:r['delta_v'])
        values=100*np.array([r['delta_v'] for r in rr]);a.bar(range(len(rr)),values,color=np.where(values>=0,'#285F9C','#BA6437'),width=1)
        a.axhline(0,color='.3',lw=.7);a.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2');a.set_xlabel('Corruption cells, sorted by paired change');a.set_ylabel('vIoU change (pp)');a.grid(axis='y',alpha=.15)
    for ext in ['png','pdf']:fig.savefig(PUB/f'P0_NEGATIVE_TAILS.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    diagnostics={}
    for ds in DATASETS:
        diagnostics[ds]={}
        for split in ['search','confirm']:
            rr=[d for d in diag if d['dataset']==ds and d['split']==split and d['condition']!='clean']
            related=[r for r in rows if r['dataset']==ds and r['split']==split and r['condition']!='clean']
            mean=lambda vv:float(np.mean([v for v in vv if v is not None])) if any(v is not None for v in vv) else None
            q=[q for r in rr for q in r['observed_expert_quality']]
            diagnostics[ds][split]=dict(
                observation_event_fraction=sum(r['observed_GT_event_frames'] for r in related)/max(sum(r['requested_frames'] for r in related),1),
                active_evidence_frames=sum(r['active_evidence_frames'] for r in related),requested_frames=sum(r['requested_frames'] for r in related),
                mean_observed_GT_box_delta=mean([r['observed_GT_event_box_IoU_delta'] for r in rr]),
                mean_unobserved_GT_box_delta=mean([r['unobserved_GT_event_box_IoU_delta'] for r in rr]),
                observed_expert_best_IoU=mean([x['best_IoU'] for x in q]),observed_expert_weighted_IoU=mean([x['weighted_IoU'] for x in q]),
                changed_unobserved_frames=sum(r['changed_unobserved_frames'] for r in rr),
                attention={a:{k:mean([r['attention'][a][k] for r in rr]) for k in ['observed_visual_mass','unobserved_visual_mass','evidence_mass']} for a in ['ordinary','privileged']})
    write(PUB/'DIAGNOSTIC_SUMMARY.json',diagnostics)
    coverage=dict(empty_evidence_cells=sum(r['active_evidence_frames']==0 for r in rows),
        empty_phrase_cells=sum(r['raw_proposals']==0 for r in rows),cells=192)
    concentration={}
    for ds in DATASETS:
        values=summary[ds]['confirm']['corruption']['metrics']['delta_v']['source_values']
        top=max(values,key=values.get);positive=sum(max(v,0) for v in values.values());net=sum(values.values())
        concentration[ds]=dict(source_id=int(top),gross_positive_source_fraction=values[top]/positive if positive else None,
            net_source_fraction=values[top]/net if net else None,posthoc_descriptive=True)
    cases={ds:{name:sorted([r for r in rows if r['dataset']==ds and r['condition']!='clean'],key=lambda r:r['delta_v'],reverse=name=='gain')[:5] for name in ['gain','harm']} for ds in DATASETS}
    write(PUB/'CASES.json',cases)
    lines=['# Event-scoped privileged spatial attention: P0 review','',
        '**Actual P0 completed: 192 paired frozen-model inputs.** '+('The predeclared P0 mean-direction gate passed; this is a conditional small-panel result.' if decision['P0_gate_pass'] else 'The predeclared P0 gate failed. Do not run OPD or LN consolidation for this fixed attention-prior implementation.'),'',
        'This experiment tests whether an external spatial field gives the same frozen STVG model a usable advantage. It does not train a new model, imitate DINO box coordinates, run online adaptation, or modify the deployed registry.','',
        '## Inputs and intervention','',
        'Each dataset has 8 development and 8 source-disjoint but historically exposed confirmation sources, one query per source, clean plus five 5% corruptions: 96 unique inputs each. The sources are original C1 ordinals 0–7 and 32–39, fixed before GT. Each P0 input requests up to four existing sampled frames inside its frozen native interval. This is an every-query expert diagnostic, not A’s 25% expert-arrival flow. Two orders would duplicate identical frozen predictions, so none are counted as extra evidence.','',
        'The baseline is native Frozen TA-STVG with the official same-domain EMA checkpoint, original Paper48 pixels/grid and two-offset decoding. A8/UniversalVTG and the previous online C1 outputs are not the baseline. Grounding-DINO tiny is frozen; the existing static referent phrase is used. All positive-area clipped raw proposals with phrase score ≥ .35 are kept, including duplicates. No top-K/NMS/margin winner enters the field. Weights are softmax(score/1); each Gaussian uses half the proposal width/height as standard deviation. Alpha=1 and epsilon=1e-6 are fixed defaults, not searched.','',
        'The field adds log(M+epsilon) to valid image-key logits in all six blocks of the final native spatial PosDecoder only. Native first-pass gates/query inputs and time are frozen. RGB, token content and global context remain present. Text/padded keys receive zero bias; image-vs-text mass can change. The prior is soft evidence, not calibrated object correctness. Cross-frame self attention can change unobserved frames despite an observed-only logit bias.','',
        '## Main paired results','',
        'Corruption means are source macro averages. Difference and paired 95% intervals are in percentage points; 10,000 whole-source draws, seed 20261004. Each split contains only 8 sources, with historical exposure. Intervals are pointwise, not simultaneous multiple-testing guarantees.','',
        '| Panel | Ordinary vIoU (%) | Privileged vIoU (%) | ΔvIoU pp [95% CI] | ΔsIoU pp [95% CI] | >5pp harm / cells |','|---|---:|---:|---|---|---:|']
    for ds in DATASETS:
        for split in ['search','confirm']:
            s=summary[ds][split]['corruption'];m=s['metrics'];lines.append(f"| {ds} {split} | {100*m['ordinary_v']['mean']:.3f} | {100*m['privileged_v']['mean']:.3f} | {pp(m['delta_v'])} | {pp(m['delta_s'])} | {s['harm_gt5pp']}/{s['cells']} |")
    lines+=['','![P0 advantage](../results/tastvg_privileged_attention_p0/2026-10-04/P0_ADVANTAGE.png)','','Clean control:','','| Panel | ΔvIoU pp [95% CI] | ΔsIoU pp [95% CI] | >5pp harm |','|---|---|---|---:|']
    for ds in DATASETS:
        for split in ['search','confirm']:
            s=summary[ds][split]['clean'];lines.append(f"| {ds} {split} | {pp(s['metrics']['delta_v'])} | {pp(s['metrics']['delta_s'])} | {s['harm_gt5pp']}/{s['cells']} |")
    lines+=['','tIoU is exactly invariant because I0 is fixed. Dense sIoU is over all annotated GT frames, independent of predicted time; ΔsIoU is not a change of temporal evaluation support. All five corruption panels and pooled source means are in SUMMARY.json.','','## Scope and evidence diagnostics','',
        '| Panel | Native observations inside GT event | Nonempty evidence frames / requested | Observed GT-frame IoU Δ pp | Unobserved GT-frame IoU Δ pp |','|---|---:|---:|---:|---:|']
    fmt=lambda x:'undefined' if x is None else f'{100*x:+.3f}'
    for ds in DATASETS:
        for split in ['search','confirm']:
            d=diagnostics[ds][split];lines.append(f"| {ds} {split} | {100*d['observation_event_fraction']:.2f}% | {d['active_evidence_frames']}/{d['requested_frames']} | {fmt(d['mean_observed_GT_box_delta'])} | {fmt(d['mean_unobserved_GT_box_delta'])} |")
    lines+=['','These are descriptive cell/frame summaries, not new primary tests or GT-based online gates. Missing GT outside an event is not assigned a made-up box. The prior can misidentify an instance, be sparse, or alter cross-modal attention mass; these diagnostics do not uniquely identify one causal failure. Proposal-best and mixture-weighted GT IoU, attention visual/evidence mass and unobserved frame changes are preserved in DIAGNOSTIC_SUMMARY.json.', '',
        f"Evidence coverage is a material limitation: {coverage['empty_evidence_cells']}/192 cells have no valid field on any requested frame, including {coverage['empty_phrase_cells']} cells with an empty conservative phrase. These remain in the paired evaluation as exact no-ops, not removed using GT. In HC development only 21/160 requested corruption observation frames receive a nonempty field. The result tests this explicit extraction/threshold/field interface, not an ideal all-query evidence teacher.", '',
        f"The Vid confirmation positive signal is concentrated: source 37 contributes {100*concentration['vidstg']['gross_positive_source_fraction']:.2f}% of positive source-mean gains ({100*concentration['vidstg']['net_source_fraction']:.2f}% of net gain). This is post-hoc concentration analysis, not a source deletion or new gate. The positive dense-sIoU CI in this panel is retained, but is insufficient to establish a shared two-dataset teacher advantage.", '',
        'The field changes actual attention allocation: observed-frame conditional evidence mass rises from .433 to .541 in Vid confirmation and from .580 to .632 in HC confirmation. This confirms an active intervention; the much smaller box/tube changes show that attention allocation improvement under this field is not itself a localization-quality guarantee.', '',
        '## Work and failure cases','',
        '| Dataset | Source | Condition | Ordinary vIoU % | Privileged vIoU % | Δ pp | Active evidence frames |','|---|---:|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for name in ['gain','harm']:
            for r in cases[ds][name][:2]:lines.append(f"| {ds} | {r['source_id']} | {r['condition']} | {100*r['ordinary_v']:.2f} | {100*r['privileged_v']:.2f} | {100*r['delta_v']:+.2f} | {r['active_evidence_frames']} |")
    lines+=['','![Paired negative tails](../results/tastvg_privileged_attention_p0/2026-10-04/P0_NEGATIVE_TAILS.png)','','## Validation, costs and decision','',
        f"10 CPU contracts passed without initializing CUDA. Two fixed clean inputs passed real full-model ordinary/privileged bitwise reinsertion and unchanged temporal logits. All 192 native ordinary tubes match their cache bitwise; all model/expert weights stayed unchanged. All returned attention probabilities, additive priors and dense metrics were independently recomputed on CPU after the global prediction barrier. Root audit: {read(PUB/'ROOT_AUDIT.json')['checks']:,} checks; public aggregate audit: {read(PUB/'PUBLIC_AUDIT.json')['checks']:,} checks.",'',
        f"New DINO forwards: {cost['new_DINO']} (cap 768); GPU worker wall {cost['GPU_worker_seconds']:.3f}s, CPU scoring/audit {cost['CPU_score_audit_seconds']:.3f}s. Zero backwards and zero parameter updates. Worker wall includes checkpoint loading, input hashing, decoding and IO; it is not pure GPU kernel time. Empty phrases/evidence are recorded as no-op rather than removed.",'',
        'The P0 gate was locked before GT: corrupt vIoU means must be positive in all four development/confirmation dataset panels. '+('Proceed only to a separately locked episodic residual-distillation test; this gate does not establish online utility or cross-query persistence.' if decision['P0_gate_pass'] else 'That condition did not hold, so the implementation stops here. No new strength sweep, Sa2VA swap, OPD, LN consolidation or old queue is started. This is not proof that every expert/evidence/prior interface is impossible.'),'',
        'Two methodological boundaries remain even if a later version succeeds: native attention probabilities are not a tube-generating policy likelihood; at a single layer with unchanged QK logits, the residual mixture A0^(1−beta)Aplus^beta is algebraically a scaled additive prior. That identity does not itself create an independent innovation or a guarantee of coordinate/localization quality.','',
        'Checkpoint training provenance: the [official Grounding-DINO repository](https://github.com/IDEA-Research/GroundingDINO) lists the published training sources. This is not an audited guarantee of zero underlying-media overlap with these historically exposed research datasets.','',
        'All implementation/protocol/anonymous metrics/negative cases and PNG/PDF figures are public-export eligible. Private videos, annotations, weights, raw boxes, spatial fields and attention/H tensors remain excluded. CURRENT_METHOD is unchanged.']
    f=ROOT/'docs/TA_PRIVILEGED_ATTENTION_P0_REVIEW.md';f.parent.mkdir(exist_ok=True);f.write_text('\n'.join(lines)+'\n')
    status(BASE/'STATUS.json',dict(status='reported_pending_root_visual_publication',cells=192,P0_gate_pass=decision['P0_gate_pass'],time=time.time()))
    print('REPORT_FIGURES_CREATED',decision,flush=True)


if __name__=='__main__':run()

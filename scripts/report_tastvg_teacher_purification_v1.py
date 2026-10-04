"""Actual measured audit report and exportable scientific charts."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,csv,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
PUB=ROOT/'results/tastvg_teacher_purification/2026-10-04'
REPORT=ROOT/'docs/TA_TEACHER_PURIFICATION_REVIEW.md'
LABELS={'vidstg':'VidSTG','hc2':'HC-STVG-v2'}
def fmt(m,unit=100):
    return f'{unit*m["mean"]:+.4f} [{unit*m["ci95"][0]:+.4f}, {unit*m["ci95"][1]:+.4f}]'
def main():
    saved=read(PUB/'SUMMARY.json');decision=read(PUB/'DECISION.json');resource=read(PUB/'RESOURCES.json')
    audit=read(PUB/'ROOT_AUDIT.json');public=read(PUB/'PUBLIC_AUDIT.json');cases=read(PUB/'CASES.json');rows=read(PUB/'ROWS.json')
    primary=[(ds,sp,saved[ds][sp]['corrupt']) for ds in ['vidstg','hc2'] for sp in ['search','confirm']]
    text=['# Teacher purification: raw-proposal consensus medoid audit', '',
        f'**{decision["status"]}:** the CPU teacher-quality audit is complete. The prelocked condition requires positive paired lower95% bounds in all four corrupt expert panels. No adaptation or R2c was run. Spatial A and the production registry are unchanged.', '',
        '## Matched setting and scope', '',
        'Reuse the original historically exposed 32 search +16 confirmation sources per dataset, one query/source, two orders, clean +five 5% corruption conditions, fixed 25% expert schedule and exact R2 raw UniversalVTG support. Only 288 scheduled expert observations (240 corrupt,48 clean) are evaluated; the 864 nonexpert cells are not scored here. Independent expert sources are Vid16/8 and HC14/7. Repeated orders/conditions do not create independent videos. No video, weight, hidden-feature or model load was used.', '',
        'Confidence is the first raw confidence argmax. Consensus is the first argmax mean interval IoU with all other raw rows, excluding self and **not using confidence**. All duplicates/order/fractional endpoints are retained. Support range26–312 and235 cache files match R2. Oracle is the first GT-best proposal, diagnostic only. This is a proposal-level teacher experiment, not a STVG readout or an online adaptation result.', '',
        'Confidence and Consensus were fixed for all288 cells in a GT-read-guarded subprocess and sealed before the scoring subprocess read pinned GT spans and previous scored controls. Teacher tIoU uses continuous half-open physical frame coordinates, exactly R2 support-level scoring; it is not the dense integer-truncated tube metric. Primary aggregation: within-source condition/order means then equal source macro; paired10000 source bootstrap, seed20261004.', '',
        '## Corrupted expert teacher quality', '',
        '| Dataset/panel | Cells/sources | Confidence tIoU (%) | Consensus tIoU (%) | Oracle tIoU (%) | Consensus−Confidence (pp, paired95% CI) |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,sp,z in primary:
        m=z['metrics'];text.append(f'| {LABELS[ds]} {sp} | {z["cells"]}/{z["sources"]} | {100*m["Confidence_t"]["mean"]:.4f} | {100*m["Consensus_t"]["mean"]:.4f} | {100*m["Oracle_t"]["mean"]:.4f} | {fmt(m["Consensus_minus_Confidence_t"])} |')
    text += ['', '| Dataset/panel | Confidence P(tIoU>.5) | Consensus P(tIoU>.5) | Oracle P(tIoU>.5) | Confidence disjoint | Consensus disjoint | Oracle disjoint |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for ds,sp,z in primary:
        m=z['metrics'];vals=[100*m[a+'_'+metric]['mean'] for metric in ['success_gt_05','disjoint'] for a in ['Confidence','Consensus','Oracle']]
        text.append('| '+LABELS[ds]+' '+sp+' | '+' | '.join(f'{v:.2f}%' for v in vals)+' |')
    text += ['', 'P(tIoU>.5) is strict **>**, not R2b’s raw-support >=.5 counter. Severe wrong event here means tIoU=0 (disjoint or touching). These rates use source macro; the following counts are cell counts.', '',
        '| Dataset/panel | Changed choice | Improved/worsened | Correct teacher destroyed/rescued | New disjoint/rescued disjoint | Positive/negative/unchanged sources |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,sp,z in primary:
        c=z['counts'];text.append(f'| {LABELS[ds]} {sp} | {c["choices_changed"]}/{z["cells"]} | {c["teacher_improved"]}/{c["teacher_worsened"]} | {c["confidence_success_destroyed"]}/{c["confidence_failure_rescued"]} | {c["new_disjoint_event"]}/{c["disjoint_event_rescued"]} | {z["positive_sources"]}/{z["negative_sources"]}/{z["unchanged_sources"]} |')
    text += ['', '## Clean, orders and source sensitivity', '',
        '| Dataset/panel | Clean Consensus−Confidence (pp,95% CI) | Corrupt order1 (pp,95% CI) | Corrupt order2 (pp,95% CI) | Corrupt leave-one-source-out mean range (pp) |',
        '|---|---:|---:|---:|---:|']
    for ds,sp,z in primary:
        clean=saved[ds][sp]['clean'];lo=list(z['leave_one_source_out_delta'].values())
        text.append(f'| {LABELS[ds]} {sp} | {fmt(clean["metrics"]["Consensus_minus_Confidence_t"])} | {fmt(z["orders"]["order1"]["metrics"]["Consensus_minus_Confidence_t"])} | {fmt(z["orders"]["order2"]["metrics"]["Consensus_minus_Confidence_t"])} | [{100*min(lo):+.4f},{100*max(lo):+.4f}] |')
    text += ['', 'All source values, full per-proposal peer/confidence scores and postseal GT-quality scalars are saved, so macro means, bootstrap, choices and tails can be independently recomputed. Full raw interval coordinates/GT are not exported.', '',
        '## Positive and failure cases', '', '| Dataset/panel | Case/source/condition/order | Confidence→Consensus tIoU (%) | Delta (pp) | Confidence→Consensus peer agreement | Oracle tIoU (%) |',
        '|---|---|---:|---:|---:|---:|']
    for ds,sp,z in primary:
        subset=[c for c in cases if c['dataset']==ds and c['split']==sp]
        for tag in ['worst','best']:
            c=next(c for c in subset if c['case']==tag)
            text.append(f'| {LABELS[ds]} {sp} | {tag}/source{c["source_id"]}/{c["condition"]}/{c["order"]} | {100*c["Confidence_t"]:.2f}→{100*c["Consensus_t"]:.2f} | {100*c["Consensus_minus_Confidence_t"]:+.2f} | {c["selected_Confidence_agreement"]:.4f}→{c["selected_Consensus_agreement"]:.4f} | {100*c["Oracle_t"]:.2f} |')
    text += ['', 'Repeated-source cases are explanations, not independent evidence. Higher peer overlap need not identify the query event: duplicates and correlated wrong intervals can agree, and broad intervals can overlap many modes. This audit can show the mismatch concretely, but does not uniquely establish which mechanism generated every bad proposal.', '',
        '## Decision and limits', '',
        f'The four-panel pass flags are `{decision["panel_pass"]}`. The registered continuation criterion is {"met" if decision["status"]=="GO_TEACHER_ONLY" else "not met"}. No teacher is promoted and no DTA is launched.', '',
        ('The medoid is a teacher candidate for a separately executed matched R2c, but teacher-quality improvement alone is not adaptation gain.' if decision['status']=='GO_TEACHER_ONLY' else
         'Stop this confidence-free raw medoid purification route; do not continue internal aggregation/weight/LR/K/PoE variants from this result. Additional independent evidence is a potential next research direction, not executed here. The negative audit does not prove that every single-expert method is impossible.'), '',
        'The old T0 consensus heuristic was confidence×peer overlap applied to routing/native readout and failed. This audit isolates a different, confidence-free raw teacher selection; its conclusion is limited to that fixed rule on the exposed panel. Oracle shows support capacity, not a deployable selection or adaptation upper bound achievable without labels. The proposal errors and R2b mixture loss need not share a single cause.', '',
        '## Verification and cost', '',
        f'Root audit: {audit["checks"]}, max absolute error{audit["max_absolute_error"]:.3g}; independent public scalar/statistical audit: {public["checks"]}, max error{public["max_absolute_error"]:.3g}. Confidence and Oracle exactly reproduce all prior R2 choices and quality values. Input/cache/state/production hashes match, and seal timing is checked.', '',
        f'Selection CPU wall {resource["selection_CPU_wall_seconds"]:.4f}s; scoring/statistics CPU wall {resource["scoring_CPU_wall_seconds"]:.4f}s; root audit wall{audit["CPU_wall_seconds"]:.4f}s. Times exclude preparation hash validation, report generation and publication, and are not GPU kernel time. Model/GPU/backbone/expert/backward/head update/spatial update/persistent temporal write counts are all zero; no torch/tensorflow/jax framework loaded.', '',
        'The canonical publication and exact remote content receipts are recorded in the private FINAL_COMPLETION and RESEARCH_HISTORY after upload. Public exports exclude raw proposals/GT/media/checkpoints/hidden states/private caches.', '',
        '![Teacher quality](../results/tastvg_teacher_purification/2026-10-04/teacher_quality.png)', '',
        '![Paired teacher change](../results/tastvg_teacher_purification/2026-10-04/paired_teacher_change.png)', '',
        '![Success and wrong-event rates](../results/tastvg_teacher_purification/2026-10-04/teacher_success_failure.png)', '']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(text))
    csv_fields=['cell_key','dataset','split','source_id','condition','order','proposal_count','unique_intervals','Confidence_index','Consensus_index','Oracle_index',
        'Confidence_t','Consensus_t','Oracle_t','Consensus_minus_Confidence_t','selected_Confidence_agreement','selected_Consensus_agreement']
    with (PUB/'ROWS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields);writer.writeheader();writer.writerows({k:r[k] for k in csv_fields} for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    names=[LABELS[ds]+'\n'+sp for ds,sp,_ in primary];colors=['#536878','#217a98','#ce9738']
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');xx=np.arange(4)
    for j,arm in enumerate(['Confidence','Consensus','Oracle']):
        m=[z['metrics'][arm+'_t'] for _,_,z in primary];vals=np.array([v['mean']*100 for v in m]);ci=np.array([v['ci95'] for v in m])*100
        ax.bar(xx+(j-1)*.24,vals,.22,label=arm,color=colors[j],yerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),capsize=3)
    ax.set_xticks(xx,names);ax.set_ylabel('Teacher tIoU (%)');ax.set_ylim(0,100);ax.legend(ncol=3,loc='upper left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for suffix in ['png','pdf']:fig.savefig(PUB/('teacher_quality.'+suffix),dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    m=[z['metrics']['Consensus_minus_Confidence_t'] for _,_,z in primary];vals=np.array([v['mean']*100 for v in m]);ci=np.array([v['ci95'] for v in m])*100
    axes[0].errorbar(vals,np.arange(4),xerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),fmt='o',color=colors[1],capsize=4)
    axes[0].axvline(0,color='#555',lw=1);axes[0].set_yticks(np.arange(4),[n.replace('\n',' ') for n in names]);axes[0].invert_yaxis();axes[0].set_xlabel('Consensus − Confidence tIoU (pp)')
    for i,(_,_,z) in enumerate(primary):
        v=list(z['metrics']['Consensus_minus_Confidence_t']['source_values'].values())
        offsets=np.linspace(-.12,.12,len(v));axes[1].scatter(np.array(v)*100,np.full(len(v),i)+offsets,s=23,color=colors[1],alpha=.85)
    axes[1].axvline(0,color='#555',lw=1);axes[1].set_yticks(np.arange(4),[n.replace('\n',' ') for n in names]);axes[1].invert_yaxis();axes[1].set_xlabel('Each source mean change (pp)')
    for ax in axes:ax.grid(axis='x',alpha=.18)
    for suffix in ['png','pdf']:fig.savefig(PUB/('paired_teacher_change.'+suffix),dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,field,title in zip(axes,['success_gt_05','disjoint'],['Correct teacher: tIoU > .5','Wrong event: tIoU = 0']):
        for j,arm in enumerate(['Confidence','Consensus','Oracle']):
            ax.bar(xx+(j-1)*.24,[100*z['metrics'][arm+'_'+field]['mean'] for _,_,z in primary],.22,color=colors[j],label=arm)
        ax.set_xticks(xx,names);ax.set_title(title);ax.set_ylabel('Source-macro rate (%)');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(ncol=3,fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(PUB/('teacher_success_failure.'+suffix),dpi=220)
    plt.close(fig)
    write(PUB/'REPORT_COMPLETION.json',dict(status='completed',report='docs/TA_TEACHER_PURIFICATION_REVIEW.md',charts=3,PNG_PDF_files=6,csv_rows=288,time=time.time()))
    print('TEACHER_PURIFICATION_REPORT_COMPLETED',decision['status'])
if __name__=='__main__':main()

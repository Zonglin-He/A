"""Generate a teacher-only report and paired-interval figure from audited rows."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
BASE=ROOT/'artifacts/tastvg_spatial_guided_temporal_p0_v1';PUB=ROOT/'results/tastvg_spatial_guided_temporal_p0/2026-10-04'
def run():
    import numpy as np,matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    assert read(PUB/'ROOT_AUDIT.json')['status']=='pass';s=read(PUB/'SUMMARY.json');rows=read(PUB/'ROWS.json');d=read(PUB/'DECISION.json');resource=read(PUB/'RESOURCES.json')
    panels=[(ds,sp) for ds in ['vidstg','hc2'] for sp in ['search','confirm']]
    lines=['# Spatially guided temporal expert P0 review','',
        'The matched three-arm expert-input audit is complete and independently audited. This evaluates confidence-top1 teacher intervals; it makes no STVG adaptation or deployment claim.','',
        'Each dataset uses the fixed 32 search +16 within-batch source-disjoint confirmation design, one query/source, two orders, clean plus five existing 5% corruptions, 25% expert schedule. Only288 scheduled expert cells are evaluated (240corrupt/48clean). Independent expert sources are Vid16/8 and HC14/7; all have historical exposure. A is the sealed pre-current-update spatial trajectory, not a fresh TA-STVG inference. Full is the exact old R2 raw teacher.','',
        '| Dataset / panel | Full tIoU | A-ROI tIoU | GT-ROI tIoU | A-ROI − Full (pp,95%CI) | GT-ROI − Full (pp,95%CI) |','|---|---:|---:|---:|---:|---:|']
    fmt=lambda m:f'{100*m["mean"]:+.4f} [{100*m["ci95"][0]:+.4f}, {100*m["ci95"][1]:+.4f}]'
    for ds,sp in panels:
        z=s[ds][sp]['corrupt']['metrics'];lines.append(f'| {ds}/{sp} | {100*z["Full_t"]["mean"]:.4f}% | {100*z["A_ROI_t"]["mean"]:.4f}% | {100*z["GT_ROI_t"]["mean"]:.4f}% | {fmt(z["A_ROI_minus_Full_t"])} | {fmt(z["GT_ROI_minus_Full_t"])} |')
    lines+=['','Source-macro paired10000-source bootstrap, seed20261004. Repeated corruption/order cells are grouped by source; the intervals are descriptive and not multiplicity corrected. Neither panel names nor source-disjoint-in-this-batch status imply a fresh test.','',
        '**GT-ROI is a spatial-box-extension diagnostic.** Spatial labels are absent outside the event for Vid138/144 and HC144/144 scheduled cells. The fixed crop pipeline interpolates coordinates and extends nearest boxes outside support, preserving every original video frame. There is no temporal event-window crop or event-based black screen. This prevents that direct timing shortcut but cannot provide perfect target tracking outside the annotated event. GT spatial annotation availability and coordinates are privileged; do not claim that this arm is deployable or a complete spatial oracle.','',
        'The square context ratio1.5 was fixed before predictions. Border pixels are edge-replicated; invalid A boxes use full-frame fallback. Crops alter identity focus, apparent scale/motion and available action/context together. A positive crop result would not uniquely establish identity confusion as the cause; a negative result would not disprove all spatial-to-temporal evidence mechanisms.','',
        '| Dataset / panel | Arm | >.5 teacher success | zero-overlap | >5pp harms | >20pp harms | rescued / destroyed successes |','|---|---|---:|---:|---:|---:|---:|']
    for ds,sp in panels:
        z=s[ds][sp]['corrupt']
        for a in ['A_ROI','GT_ROI']:
            t=z['tails'][a];lines.append(f'| {ds}/{sp} | {a} | {100*z["metrics"][a+"_success"]["mean"]:.2f}% | {100*z["metrics"][a+"_disjoint"]["mean"]:.2f}% | {t["severe_harm_gt5pp"]} | {t["severe_harm_gt20pp"]} | {t["success_rescued"]} / {t["success_destroyed"]} |')
    lines+=['','Clean and order controls are fully published in SUMMARY.json. Clean teacher differences:']
    for ds,sp in panels:
        z=s[ds][sp]['clean']['metrics'];lines.append(f'- {ds}/{sp}: A-ROI−Full {fmt(z["A_ROI_minus_Full_t"])}pp; GT-ROI−Full {fmt(z["GT_ROI_minus_Full_t"])}pp.')
    cases={}
    for ds,sp in panels:
        q=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean']
        for a in ['A_ROI','GT_ROI']:
            rr=sorted(q,key=lambda r:r[a+'_minus_Full_t']);cases[ds+'/'+sp+'/'+a]=dict(worst=rr[:2],best=rr[-2:])
    write(PUB/'CASES.json',cases)
    lines+=['','Decision: **'+d['status']+'**. The strict rule requires positive paired lower95CI for A-ROI−Full in all four corrupt panels. No crop-ratio tuning, DTA, new expert, source resampling, parameter update or method promotion is performed. Retain A and the old Full temporal expert as the control. Positive and negative examples, teacher-support oracle values and severe failure counts are preserved.','',
        f'New expert-input calls: A-ROI {resource["A_ROI"]["new_calls"]}, GT-ROI {resource["GT_ROI"]["new_calls"]}; original Full235 unique cached inputs reused, plus2 Full reencoding smoke controls. GPU worker wall: A-ROI {resource["A_ROI"]["GPU_worker_wall_seconds"]:.2f}s, GT-ROI {resource["GT_ROI"]["GPU_worker_wall_seconds"]:.2f}s. Wall includes model loading/CPU decoding/cropping/IO and is not pure GPU kernel time. Zero backbone calls, spatial expert calls, backwards or parameter updates.','',
        'Root audit reconstructs all A box conversions, GT spatial interpolations/extensions, crop-window rules, state/pixel/cache hashes, same time grids, raw confidence choices and continuous physical teacher metrics. Portable public audit recomputes scalar differences, source bootstrap, orders, counts, CSV and seal chronology. Two predetermined smoke samples reproduce original Full proposals/top1 within the saved FP16 tolerance. This confirms the interface, not teacher correctness.','',
        'The CVPR2023 [Collaborative Static and Dynamic Vision-Language Streams](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Collaborative_Static_and_Dynamic_Vision-Language_Streams_for_Spatio-Temporal_Video_Grounding_CVPR_2023_paper.html) uses learned spatial attention to guide a dynamic stream; that trained architecture motivates the question but is not evidence that this frozen expert crop must work. Earlier pooled ROI-cosine critics and joint8×9 oracle diagnoses are different interventions.','',
        '![Paired teacher differences](../results/tastvg_spatial_guided_temporal_p0/2026-10-04/teacher_differences.png)','',
        'Implementation/protocol and anonymous results are public; original frames, GT boxes/spans, captions, physical raw proposals, model weights, cropped pixels and features remain private.']
    (ROOT/'docs/TA_SPATIALLY_GUIDED_TEMPORAL_P0_REVIEW.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,4.2))
    for j,(a,color) in enumerate([('A_ROI','#247b92'),('GT_ROI','#cc8a2f')]):
        m=[s[ds][sp]['corrupt']['metrics'][a+'_minus_Full_t'] for ds,sp in panels]
        x=np.arange(4)+(j-.5)*.17;y=np.array([100*z['mean'] for z in m]);ci=np.array([z['ci95'] for z in m])*100
        ax.errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o',color=color,label=a.replace('_','-'),capsize=4,markersize=6)
    ax.axhline(0,color='#666',linewidth=.8);ax.set_xticks(range(4),[ds+'\n'+sp for ds,sp in panels]);ax.set_ylabel('Teacher tIoU difference vs Full (pp)');ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(PUB/'teacher_differences.png',dpi=200);fig.savefig(PUB/'teacher_differences.pdf');plt.close(fig)
    write(PUB/'ROOT_REVIEW.json',dict(status='pass',decision=d['status'],DTA_started=False,GT_extension_scope_disclosed=True,time=time.time()))
    print(d['status'],flush=True)
if __name__=='__main__':run()

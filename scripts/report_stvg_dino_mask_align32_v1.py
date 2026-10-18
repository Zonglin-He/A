"""Actual matched-mask and one-update results, including all signed outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_mask_align32_v1 import D,P,PRE,ARMS,verify


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    p=verify();r=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    cfg={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']}
    cfg.update(source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/dino_conditional32/CONFIGURATION.json')['source_checkpoints'],DINO_weight_SHA256='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3',
        covariance='unit Gaussian in source-scaled shared4 coordinates; same original24 noise',pretrained_parameter_updates=0,source_E_cost_excluded=True)
    put(P/'CONFIGURATION.json',cfg);put(P/'CPU_PROPERTY_COMPLETION.json',read(D/'CPU_PROPERTY_COMPLETION.json'))
    cost=collections.Counter();quals=[]
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));cost.update(z['compute']);quals.extend(z['qualifications'])
    put(P/'ACTUAL_COSTS.json',dict(**dict(cost),GPU_lease_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_seconds'],qualification=quals,first2=read(D/'PILOT2_TIMING.json'),existing_E_Source_cost_excluded=True,not_complete_first_online_latency=True))
    lines=['# Content-source attention exclusion: original32 matched experiment','','Same high native target-score content, exact same fixed24 student references, original4 observations, image/text/native900, .35 legality and score rules. Only the excluded native-attention slot changes: A geometry-nearest; B content-source. Six-layer readout boxes/scores, actual propagated references/hidden and native initial900 references saved privately. Last layer is fixed; no layer chosen by GT.','','Both conditions actually perform mu_new=mu+sum24(y-a)/24 from E, preserving exact no-feedback no-op. This is one external Gaussian-mean correction, not network-weight training. E, Source, MBR and old predictions unchanged. All32 complete outputs sealed before new GT; original exposed HC16/Vid16, not independent confirmation. Four unsupported Vid queries retained.','','| Domain | Source vIoU% | E vIoU% | A unaligned once% | B aligned once% |','|---|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():lines.append('| '+ds+' | '+' | '.join(f'{100*g["metrics"][a]["v"]:.6f}'for a in ARMS)+' |')
    def fmt(v):return 'unknown'if v['mean']is None else f'{v["mean"]:+.6f} [{v["CI95"][0]:+.6f},{v["CI95"][1]:+.6f}]'
    lines+=['','| Domain | Comparison | Full vIoU pp [95% query CI] | Unobserved actual Decoder sIoU pp [95% CI] | Loss >5pp queries |','|---|---|---:|---:|---:|']
    for ds,g in r['groups'].items():
        for name,pair in g['paired'].items():lines.append(f'| {ds} | {name} | {fmt(pair["v"])} | {fmt(pair["unobserved_actual_s"])} | {pair["v"]["severe_loss_below_minus5pp"]}/16 |')
    lines+=['','## Feedback coverage','','| Domain | Interface | Admitted / returned anchors | Candidates with feedback | Queries with feedback |','|---|---|---:|---:|---:|']
    for ds,g in r['groups'].items():
        for m,c in g['feedback'].items():lines.append(f'| {ds} | {m} | {c.get("accepted_refs",0)}/{c.get("returned_refs",0)} | {c.get("candidates_with_feedback",0)} | {c["queries_with_feedback"]}/16 |')
    lines+=['','## Where improvement is used or lost','','All-position gated accounting retains illegal/no-feedback samples as zero correction and query as unit. Accepted-only quantities additionally report their available query counts in JSON. Unknown GT is excluded, not scored as failure. Teacher-vs-E evaluates whether feedback contains information beyond recovery of exploration noise. Compression compares the SAME accepted local boxes with the shared4 target they induced. Actual final once update is separately evaluated over complete tube and unobserved frames. These are different supports and not an additive proof of causality.','','| Domain | Interface | Local teacher - sampled input pp | Local teacher - E pp | Shared4 - E pp | Shared4 - local teacher pp |','|---|---|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        for m,a in g['accounting'].items():lines.append('| '+ds+' | '+m+' | '+' | '.join(fmt(a[k])for k in ['gated_teacher_minus_sample_pp','gated_teacher_minus_E_pp','gated_shared_minus_E_pp','gated_compression_minus_local_pp'])+' |')
    lines+=['','## Layer evidence','','| Domain | Interface | Mean target scores at layer1..6 (query means) |','|---|---|---|']
    for ds,g in r['groups'].items():
        for m,s in g['layers'].items():lines.append(f'| {ds} | {m} | '+', '.join('unknown'if v is None else f'{v:.6f}'for v in s)+' |')
    lines+=['','Per-layer head readouts and actual next-layer references are distinct arrays. Native replay and cloned original reference/content/exclusion are qualified across all6 layers. Native initial reference is now measured, not inferred from a postprocessed final box. Layer traces diagnose response; they do not license best-layer selection or replacing the final prediction. Classification is hidden/text matching, not geometric correctness.','','## Decision','',r['decision']['status']+'. Prelocked continuation requires B coverage>A and B once full vIoU>E in BOTH domains. No target-selected thresholds, layer choice, state dimensions or propagation changes. If qualified, only original32 dynamic3x8 is authorized, with each round correction SUM divided by24, matching explicit cumulative weight to Fixed24. Teacher is reevaluated at new states; no path splicing.','','All signed rows, noops, uncertainty and negative tails remain public. No geometric IoU-only identity claims. No new training/backbone/QR/LN/Qwen/SAM/historical replay. Raw media, text/parent identifiers, GT/coordinates, weights and layer tensors remain private. Costs exclude old Source/E and are not complete first online latency.']
    if (P/'DYNAMIC_RESULTS_PUBLIC.json').exists():
        dr=read(P/'DYNAMIC_RESULTS_PUBLIC.json');assert read(P/'DYNAMIC_ROOT_AUDIT.json')['status']=='pass'
        lines+=['','## Conditional matched-weight dynamic result','','| Domain | E% | Fixed24 B% | Dynamic3x8% | Dynamic - Fixed pp [95% CI] |','|---|---:|---:|---:|---:|']
        for ds,g in dr['groups'].items():lines.append(f'| {ds} | {100*g["metrics"]["E_MBR"]["v"]:.6f} | {100*g["metrics"]["B_aligned_once"]["v"]:.6f} | {100*g["metrics"]["B_aligned_dynamic"]["v"]:.6f} | {fmt(g["paired"]["B_aligned_dynamic_minus_B_aligned_once"]["v"])} |')
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('See REPORT.md, RESULTS_PUBLIC.json, CONFIGURATION.json, DECISION.json, ROOT_INDEPENDENT_AUDIT.json and ACTUAL_COSTS.json.\n')
    review=D/'root_review';review.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));fig.subplots_adjust(bottom=.2,wspace=.3)
    for ax,(ds,g)in zip(axs,r['groups'].items()):
        names=ARMS[1:];ax.bar(['E','A once','B aligned once'],[100*g['metrics'][a]['v']for a in names],color=['#537b8c','#8c79a6','#bf6849']);ax.set_title(ds+'16 exposed development');ax.set_ylabel('Complete tube vIoU (%)')
    fig.text(.08,.03,'Only attention exclusion differs; high content / student reference / .35 threshold / MBR fixed.\nActual external shared4 once update; frozen pretrained weights. All outputs before GT.',fontsize=8)
    fig.savefig(P/'MASK_MATCHED_UPDATE.svg');fig.savefig(review/'MASK_MATCHED_UPDATE.png',dpi=130);plt.close(fig)
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    cache=collections.OrderedDict();panels=[]
    for ds in r['groups']:
        rec=next(x for x in p['records']if x['dataset']==ds);z=load(D/'predictions'/(rec['alias']+'.pt'));row=read_row(ds,rec['query_ordinal']);frames,_,_=frames_for(ds,row,cache)
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        for ax,pos in zip(axs,z['planned_positions']):
            rgb=frames[pos];h,w=rgb.shape[:2];ax.imshow(rgb)
            for a,col in [('E_MBR','lime'),('A_unaligned_once','cyan'),('B_aligned_once','magenta')]:
                cx,cy,bw,bh=z['outputs'][a][pos];ax.add_patch(Rectangle(((cx-bw/2)*w,(cy-bh/2)*h),bw*w,bh*h,fill=False,color=col,lw=1.4))
            ax.set_title('anchor '+str(pos));ax.axis('off')
        fig.suptitle(rec['alias']+' E green / A once cyan / B once magenta; no identity labels');f=review/(ds+'_actual_pixels.png');fig.savefig(f,dpi=100);plt.close(fig)
        panels.append(dict(alias=rec['alias'],dataset=ds,path=str(f.relative_to(ROOT)),sha256=sha(f),anchors=4))
    put(D/'ACTUAL_PIXEL_PANELS.json',dict(status='pending_root_view',panels=panels,new_semantic_labels=0))
    own=['vg_tta/stvg_dino_mask_align32_v1.py','scripts/run_stvg_dino_mask_align32_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','scripts/audit_stvg_dino_mask_align32_v1.py','scripts/report_stvg_dino_mask_align32_v1.py','scripts/publish_stvg_dino_mask_align32_v1.py','protocols/stvg_dino_mask_align32_v1.md']
    files=['RESULTS_PUBLIC.json','CONFIGURATION.json','DECISION.json','ROOT_INDEPENDENT_AUDIT.json','ACTUAL_COSTS.json','CPU_PROPERTY_COMPLETION.json','REPORT.md','README.md','MASK_MATCHED_UPDATE.svg']
    if (P/'DYNAMIC_RESULTS_PUBLIC.json').exists():files+=['DYNAMIC_RESULTS_PUBLIC.json','DYNAMIC_ROOT_AUDIT.json']
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=own+[str((P/f).relative_to(ROOT))for f in files]));print(dict(status='report_ready',files=len(own)+len(files)))


if __name__=='__main__':main()

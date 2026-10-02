"""Standalone scientific figures and a source-backed research qualification report."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from tastvg_reference_common_v1 import *
import numpy as np

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    summary=read(PUBLIC/'SUMMARY.json');resource=read(PUBLIC/'RESOURCES.json');root=read(PUBLIC/'ROOT_READBACK.json')
    colors={'vidstg':'#315D95','hc2':'#BA7635'};names={'vidstg':'VidSTG','hc2':'HC-STVG-v2'}
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.3),layout='constrained')
    specs=[('delta_event_precision','Event-frame precision change (pp)'),('delta_fixed_pairwise','Pairwise accuracy change (pp)'),('delta_fixed_v','Selected-candidate vIoU change (pp)')]
    for ax,(field,label) in zip(axs,specs):
        for j,ds in enumerate(DATASETS):
            m=summary[ds]['corruption']['metrics'][field];mu=100*m['mean'];lo,hi=np.asarray(m['ci95'])*100
            ax.errorbar(mu,1-j,xerr=[[mu-lo],[hi-mu]],fmt='o' if j==0 else 's',color=colors[ds],capsize=4,elinewidth=1.5,markersize=6)
        ax.axvline(0,color='#555555',lw=1,ls='--');ax.set_yticks([1,0],['VidSTG','HC-STVG-v2']);ax.set_xlabel(label);ax.set_ylim(-.55,1.55);ax.grid(axis='x',alpha=.15)
    for extension in ['png','pdf','svg']:fig.savefig(PUBLIC/f'qualification.{extension}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    def mean(ds,group,key):return summary[ds][group]['metrics'][key]['mean']*100
    def ci(ds,group,key):
        m=summary[ds][group]['metrics'][key];return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
    text=['# Student-routed Sa2VA: candidate judgment improves in HC, Vid remains uncertain',
      '', 'The small matched frame-acquisition qualification is complete. Student support places more reference frames inside the event in both datasets. On the same nine sealed A spatial candidates, HC ranking and selected-candidate utility improve; Vid has small positive means with intervals crossing zero. Keep original spatial A and temporal critic. The HC signal justifies considering the separately proposed reference-position-only online comparison; no new online trajectory was run or automatically started.',
      '', '## Fixed setting and scope', '',
      'Each dataset: ten source-hash-selected sources from its original 32 historically exposed development pool, one query per source, one original A order at a scheduled expert arrival, clean plus two balanced transient 5% corruptions per source. Thirty matched cells per dataset: 20 corrupted and 10 clean; every corruption family has four cells. This is a development qualification, not fresh generalization or full evaluation.',
      '', 'Uniform reuses the original Sa2VA five-frame cache. Student-routed observes five distinct CDF 10/30/50/70/90% frames of the existing student temporal-candidate consensus. Model, prompt, BF16, preprocessing and frame budget are identical. Original A Rank-RKL, learning rate/temperature, Vid K1 and HC K8, 1792 parameters, original checkpoint/sampling/pixels and temporal critic remain fixed. Nine candidates come from the first sealed inner step; HC later steps are not independent observations.',
      '', 'New calls: 60 routed plus two uniform bitwise parity checks, 62 total, no repeated calls. All specialist outputs and GT-free rewards/selections were globally sealed before diagnostic labels were interpreted. No student forward, backward or persistent-state update occurred.',
      '', '## Corrupted expert cells: primary matched results', '',
      '| Dataset | Uniform event-frame precision | Routed precision | Uniform pairwise accuracy | Routed accuracy | Routed−Uniform candidate vIoU (pp, paired 95% CI) | Oracle regret Uniform→Routed (pp) |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        text.append(f"| {names[ds]} | {mean(ds,'corruption','Uniform_event_precision'):.2f}% | {mean(ds,'corruption','Routed_event_precision'):.2f}% | {mean(ds,'corruption','Uniform_fixed_pairwise'):.2f}% | {mean(ds,'corruption','Routed_fixed_pairwise'):.2f}% | {ci(ds,'corruption','delta_fixed_v')} | {mean(ds,'corruption','Uniform_fixed_regret'):.4f} → {mean(ds,'corruption','Routed_fixed_regret'):.4f} |")
    text += ['', 'Primary vIoU evaluates every candidate at the SAME original A native interval, isolating spatial judgment. Candidate-own interval is a prespecified secondary readout; all 540 candidate intervals actually match their arrival native interval, so these two utility readouts coincide. These numbers are the utility of direct critic selection from fixed candidates; they are not an adapted model’s gain versus Frozen.',
      '', f"HC paired ranking improvement: {ci('hc2','corruption','delta_fixed_pairwise')} pp; Vid: {ci('vidstg','corruption','delta_fixed_pairwise')} pp. The ten sources are the bootstrap units, keeping two corruptions together; 10000 paired resamples, seed20261001. Vid ranking is defined on 16/20 corrupted cells and 8/10 sources: four cells have all nine GT utilities tied at zero. HC ranking uses all20 cells/10 sources. All36 candidate pairs enter the check; GT ties are excluded, expert ties/empty feedback earn chance .5, and decisive coverage is published. Do not count pairs or frames as independent videos.",
      '', '## Evidence failures and candidate headroom', '',
      '| Dataset | Empty arrivals Uniform→Routed | No event-inside valid reference Uniform→Routed | Valid observed masks Uniform→Routed | Changed candidate choices | Positive/negative utility cells |',
      '|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        z=summary[ds]['corruption'];a,b=z['counts']['Uniform'],z['counts']['Routed']
        text.append(f"| {names[ds]} | {a['empty_arrivals']} → {b['empty_arrivals']} /20 | {a['no_event_valid']} → {b['no_event_valid']} /20 | {a['valid_frames']} → {b['valid_frames']} /100 | {z['changed_selection']}/20 | {z['positive_cells']} / {z['negative_cells']} |")
    text += ['', 'HC reduces arrivals without event-inside valid evidence from6 to2, including a reduction among nonempty feedback from4/18 to1/19. Vid remains mixed: no event-inside valid evidence4→5 and one new empty arrival. More event hits do not guarantee a correct object mask or improved candidate judgment.',
      '', 'All actual routed inputs have five distinct frames; raw quantile duplicates were zero in all60 cells. The deterministic fill rule was locked and analytically tested before inference. Actual event endpoints preserve the previous half-open convention; HC differences from inclusive final GT-box support are recorded in ROOT_READBACK rather than changing the old metric.',
      '', 'At vIoU>.3 the candidate set contains a correct candidate in only3/20 Vid cells and8/20 HC cells; both strategies select a correct candidate in every such cell, with zero native-correct destruction. At>.5 support is3/20 Vid and2/20 HC. The measured benefit is continuous utility/ranking, not a new threshold-correction gain. Current probe support remains narrow; its oracle ceiling and residual regret are published.',
      '', '## Positive and negative examples', '']
    for ds in DATASETS:
        rows=[r for r in read(PUBLIC/ds/'ROWS.json') if r['condition']!='clean'];pos=max(rows,key=lambda r:r['delta_fixed_v']);neg=min(rows,key=lambda r:r['delta_fixed_v'])
        text.append(f"{names[ds]}: largest positive is anonymous source{pos['source_id']}/{pos['condition']}/{pos['order']}, candidate{pos['Uniform_selected']}→{pos['Routed_selected']}, ΔvIoU{100*pos['delta_fixed_v']:+.4f}pp. Largest negative is source{neg['source_id']}/{neg['condition']}/{neg['order']}, candidate{neg['Uniform_selected']}→{neg['Routed_selected']}, ΔvIoU{100*neg['delta_fixed_v']:+.4f}pp. Full rows, pair differences and CASES retain both directions.")
        text.append('')
    text += ['## Clean and unchanged temporal-readout control', '',
      f"Clean candidate-utility difference: Vid {ci('vidstg','clean','delta_fixed_v')}pp; HC {ci('hc2','clean','delta_fixed_v')}pp. HC clean pairwise gain is{ci('hc2','clean','delta_fixed_pairwise')}pp. Thus the mechanism is not established as corruption-specific.",
      '', 'After inspecting the primary results, a clearly labeled supplementary CPU readout evaluates exactly the same candidates at the unchanged original A final temporal-critic interval. It changes no observations, rewards or decisions and does not replace the predeclared primary endpoint. See FINAL_INTERVAL_CONTROL.json for all values and denominators. This is a sensitivity check, not a separately pre-registered test.', '']
    for ds in DATASETS:
        m=read(PUBLIC/ds/'FINAL_INTERVAL_CONTROL.json')['summary']['corruption']['delta_v']['metrics']['delta_v']
        text.append(f"{names[ds]} at the original A final interval: routed−uniform {100*m['mean']:+.4f}pp [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}].")
    text += ['', '## Resources, verification and engineering record', '',
      f"Finite specialist worker wall: {resource['worker_wall_seconds']:.4f}s including IO/loading and the saved engineering attempt; peak allocated VRAM {resource['peak_vram_bytes']/2**30:.3f}GiB. This is not pure GPU-kernel time or end-to-end online-method throughput. Specialist calls62, student forward0, backward0. Independent root checks:60 sealed A states,60 routers,300 literal CDF quantiles,4320 dense metric scalar comparisons,1026 reward scalars and120 selections; maximum dense error {root['max_error']:.4g}. Public audit independently verifies all pair signs, candidate selection/utility/regret, macro and paired bootstrap.",
      '', 'At61/62 the final HC freeze donor input failed the old pixel-hash check. The isolated runner had bound the main HC decode correctly but left the global donor decoder using Vid vsync0. Restore the ORIGINAL HC global decode binding; all four selected HC freeze inputs then exactly match old pixels. Original failed code/lock/log/receipts are preserved with additive revision001. Reuse the matching61 outputs and complete only the remaining call; no GT was read during repair, no sample or method changed, no extra specialist call was used. Two uniform controls reproduce input digest, masks, boxes and prediction text bitwise.',
      '', '## Decision', '',
      'HC passes this limited candidate-judgment qualification: improvement extends beyond event coverage to ranking, selected utility and oracle regret on the frozen development set. Vid evidence is inconclusive: coverage improves, but judgement/utility intervals cross zero and evidence failures do not decrease. This does not demonstrate future nonexpert transfer, online vIoU improvement, or a universal advantage of student routing. Retain spatial A and the original temporal critic. No E/SE/QC, geometry weighting, temporal parameter changes, memory, new tuning, full H-full or historical full-query queue is started. A reference-position-only online comparison is the next conditional research step, not a completed result or automatic production promotion.',
      '', '![Candidate qualification](qualification.png)', '']
    report='\n'.join(text);(PUBLIC/'REPORT.md').write_text(report);(ROOT/'docs/TA_REFERENCE_SELECTION_REVIEW.md').write_text(report.replace('](qualification.png)','](../results/tastvg_reference_selection/2026-10-02/qualification.png)'))
    print('REPORT_AND_FIGURES',PUBLIC)
if __name__=='__main__':run()

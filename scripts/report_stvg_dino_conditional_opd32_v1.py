"""Anonymous measured report plus private actual-pixel readback panel."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_conditional_opd32_v1 import D,P,ARMS,verify


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    p=verify(True);r=read(P/'RESULTS_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
    cfg={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']}
    cfg.update(backbone='TA-STVG frozen original cross-domain checkpoints, cached output-only reuse',
        source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/query_e_response64/CONFIGURATION.json')['source_checkpoints'],
        DINO_snapshot='IDEA-Research/grounding-dino-tiny a2bb814dd30d776dcf7e30523b00659f4f141c71',
        DINO_weight_SHA256='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3',
        original_official_source='https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/transformer.py')
    put(P/'CONFIGURATION.json',cfg)
    costs=dict(GPU_lease_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_seconds'],
        CPU_score_seconds=read(D/'CPU_COMPLETION.json')['score_seconds'],first2=read(D/'PILOT2_TIMING.json'),
        new_Source_calls=0,new_DINO_encoded_frames=sum(x['compute']['full_image_text_encodings']for x in r['rows']),
        encoding_full_forward_seconds=sum(x['compute']['encoding_full_forward_seconds']for x in r['rows']),
        conditional_decoder_seconds=sum(x['compute']['conditional_decoder_seconds']for x in r['rows']),
        conditional_decoder_calls=sum(x['compute']['conditional_decoder_calls']for x in r['rows']),
        logical_reference_queries=sum(x['compute']['logical_candidate_queries']for x in r['rows']),
        unique_reference_queries=sum(x['compute']['unique_candidate_queries']for x in r['rows']),
        per_arm_distill_and_conditional_seconds={a:sum(load(D/'predictions'/(x['alias']+'.pt'))['fits'][a]['seconds']for x in p['records'])for a in ARMS[2:]},
        qualifier_decoder_passes_included_separately_not_charged_as_adaptation=True,
        encoding_forward_contains_native_decoder=True,old_DINO_E_generation_cost_not_remeasured=True,
        complete_first_online_total_cost_not_claimed=True,new_Qwen_SAM_training=0)
    put(P/'ACTUAL_COSTS.json',costs)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4));fig.subplots_adjust(bottom=.27,wspace=.3)
    for ax,ds in zip(axs,['hc2','vidstg']):
        g=r['groups'][ds];vals=[100*g['metrics'][a]['v']for a in ARMS]
        ax.bar(np.arange(5),vals,color=['#999999','#528b99','#d8984c','#7564a5','#ba5353']);ax.set_xticks(range(5),['Source','E','One','Fixed24','Dynamic3'],rotation=20);ax.set_ylabel('Complete tube vIoU (%)');ax.set_title(ds+'16 (exposed development)')
    fig.text(.07,.04,'All methods share final-MBR. One/fixed/dynamic use the same logical24 references per anchor; one deduplicates repeated references.\nDynamic is three rounds of frozen state-conditioned DINO feedback + exact Gaussian mean distillation; no network weights change.',fontsize=8)
    review=D/'root_review';review.mkdir(exist_ok=True);fig.savefig(P/'MEASURED_RESULTS.svg');fig.savefig(review/'MEASURED_RESULTS.png',dpi=140);plt.close(fig)
    lines=['# Student-conditioned DINO trajectory distillation:32-query pilot','','All32 predictions sealed before new GT. HC16/Vid16 are deterministic subsets of the exposed original64, not independent confirmation. E and time final-MBR remain fixed. Three new arms use frozen DINO conditional feedback and a four-dimensional episodic external correction distribution. No QR/LN/backbone update.','','| Domain | Source vIoU% | E vIoU% | One correction% | Fixed24% | Dynamic3% |','|---|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():lines.append('| '+ds+' | '+' | '.join(f'{100*g["metrics"][a]["v"]:.6f}'for a in ARMS)+' |')
    lines+=['','| Domain | Contrast | Complete vIoU pp [95% CI] | Unobserved actual sIoU pp [95% CI] | Loss >5pp /16 |','|---|---|---:|---:|---:|']
    for ds,g in r['groups'].items():
        for name in ['One_correction_minus_E_MBR','Fixed_sample_distill_minus_E_MBR','On_policy_distill_minus_E_MBR','On_policy_distill_minus_Fixed_sample_distill','On_policy_distill_minus_One_correction']:
            v=g['paired'][name]['v'];u=g['paired'][name]['unobserved_actual_s'];fmt=lambda x:'unknown'if x['mean']is None else f'{x["mean"]:+.6f} [{x["CI95"][0]:+.6f},{x["CI95"][1]:+.6f}]'
            lines.append(f'| {ds} | {name} | {fmt(v)} | {fmt(u)} | {v["severe_loss_below_minus5pp"]}/16 |')
    lines+=['','## Mechanism actually run','','Native900 context queries are retained. Each student reference takes the content query of the nearest original encoder proposal by IoU. An additive attention mask isolates student candidates and removes the matched native slot from that added query\'s support. Ordinary native replay was exact; cloned references passed the prelocked1e-4 box tolerance in both domains. These are engineering checks, not grounding-quality proof.','','The teacher corrects each sampled state at the original four images. Target token score>=.35 supplies an anchor correction; no score calibration or identity-probability claim. Shared4 teacher means are averages over accepted anchors, then the student Gaussian mean is the exact forward-KL mixture projection. This tests on-policy teacher-conditioned distillation, not the old fixed-E IoU-reward/Adam OPD objective. Fixed and dynamic sample24 actual states each; one-correction allocates24 repeated logical references and computes duplicates only once. Candidate processing count and decoder passes are different.','','No accepted anchor returns its input sampled action as target. With finite Gaussian samples, averaging identity targets can still introduce sampling drift; unchanged feedback is not automatically unchanged distribution mean. Invalid normalized references get whole-action no-op, no clipping or replacement. These counts and all signed/unknown cases are retained in RESULTS_PUBLIC.json.','','## Decision','','`'+r['decision']['status']+'`. No automatic expansion or LR rescue. A negative result applies to this content initialization, frozen DINO reference interface, shared4 representation and Gaussian mean distillation; it does not invalidate every form of OPD or iterative correction. Sparse4 images with many candidate states are not dense independent full-video evidence. Time/tIoU is identical across all arms.','','## Actual cost','',f'GPU lease {costs["GPU_lease_seconds"]:.3f}s; original-anchor full image/text encoding-forward {costs["encoding_full_forward_seconds"]:.3f}s (includes native decoder); additional conditional decoder {costs["conditional_decoder_seconds"]:.3f}s. New encoded frames {costs["new_DINO_encoded_frames"]}, logical references {costs["logical_reference_queries"]}, unique references {costs["unique_reference_queries"]}. No Source forward, source training, Qwen, SAM or new backbone. Existing E generation cost excluded; this is not complete first online deployment latency.','','## Evidence and limitations','','All five outputs and all signed paired query-level bootstrap intervals are public. Missing GT stays unknown, all no-support/no-op cases remain in denominator, no GT step selection. Private RGBs, original Query/parent identifiers, model weights, raw tensors and GT coordinates are excluded. Source/E exact prior metrics, all saved teacher-target-to-student updates and metrics were independently recomputed. No new identity labels.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('See [REPORT.md](REPORT.md), [RESULTS_PUBLIC.json](RESULTS_PUBLIC.json), [CONFIGURATION.json](CONFIGURATION.json), [ROOT_INDEPENDENT_AUDIT.json](ROOT_INDEPENDENT_AUDIT.json) and [ACTUAL_COSTS.json](ACTUAL_COSTS.json).\n')
    # Fixed first query/domain: actual pixels + outputs, private only, no identities.
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    framecache=collections.OrderedDict();pixelread=[]
    for ds in ['hc2','vidstg']:
        rec=next(x for x in p['records']if x['dataset']==ds);z=load(D/'predictions'/(rec['alias']+'.pt'));row=read_row(ds,rec['query_ordinal']);frames,ids,_=frames_for(ds,row,framecache)
        fig,axs=plt.subplots(1,4,figsize=(16,4));fig.subplots_adjust(top=.85,bottom=.05)
        for ax,pos in zip(axs,z['planned_positions']):
            rgb=frames[pos];h,w=rgb.shape[:2];ax.imshow(rgb)
            for arm,col in [('Source_MBR','white'),('E_MBR','lime'),('On_policy_distill','magenta')]:
                cx,cy,bw,bh=z['outputs'][arm][pos];ax.add_patch(Rectangle(((cx-bw/2)*w,(cy-bh/2)*h),bw*w,bh*h,fill=False,color=col,lw=1.4))
            ax.set_title('anchor '+str(pos));ax.axis('off')
        fig.suptitle(rec['alias']+' actual original4: white Source, green E, magenta dynamic (no identity annotation)');name=review/(ds+'_actual_pixels.png');fig.savefig(name,dpi=100);plt.close(fig)
        pixelread.append(dict(dataset=ds,alias=rec['alias'],path=str(name.relative_to(ROOT)),sha256=sha(name),anchors=len(z['planned_positions']),new_semantic_labels=0))
    put(D/'ACTUAL_PIXEL_PANEL.json',dict(status='generated_pending_actual_root_view',panels=pixelread,new_semantic_labels=0))
    files=['vg_tta/stvg_dino_conditional_opd32_v1.py','scripts/run_stvg_dino_conditional_opd32_v1.py','scripts/score_stvg_dino_conditional_opd32_v1.py','scripts/audit_stvg_dino_conditional_opd32_v1.py','scripts/report_stvg_dino_conditional_opd32_v1.py','scripts/publish_stvg_dino_conditional_opd32_v1.py','protocols/stvg_dino_conditional_opd32_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS_PUBLIC.json','CONFIGURATION.json','DECISION.json','ROOT_INDEPENDENT_AUDIT.json','ACTUAL_COSTS.json','REPORT.md','README.md','MEASURED_RESULTS.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(dict(status='report_and_private_panels_ready',files=len(files)))


if __name__=='__main__':main()

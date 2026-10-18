"""Matched content-only validation report; anonymous scalars, no raw boxes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_content_match32_v1 import D,P,verify


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    p=verify();r=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    cfg={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']}
    cfg.update(source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/dino_conditional32/CONFIGURATION.json')['source_checkpoints'],
        DINO_weight_SHA256='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3',pretrained_weight_updates=0,
        stage1_student_updates=0,original_frame_budget=4,no_new_Source_calls=True)
    put(P/'CONFIGURATION.json',cfg)
    cost=collections.Counter();quals=[]
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));cost.update(z['compute']);quals.extend(z['qualifications'])
    put(P/'ACTUAL_COSTS.json',dict(**dict(cost),GPU_lease_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_seconds'],
        qualification=quals,first2=read(D/'PILOT2_TIMING.json'),new_Source_calls=0,old_Source_E_generation_excluded=True,
        not_first_online_end_to_end_latency=True,conditional_calls_include_qualification=True))
    lines=['# Same student reference, different content initialization: original32','','Only the appended learned content vector changes. Same references, native900 context and geometry-nearest attention exclusion, target tokens, .35 output gate, original four frames and frozen detector. Native target-high is the first argmax of the SAME native forward target-token score. This is not an ROI content initializer or a calibrated identity posterior.','','Original exposed HC16/Vid16 development. Exact previous fixed24 actions are reused; invalid whole-trajectory references stay no-op. Four unsupported Vid queries remain in the denominator. All32 outputs sealed before new GT. No OPD/Student update in this validation.','','| Domain | Content | Admitted / returned anchors | Queries with feedback /16 | Known-anchor gated correction pp [95% query CI] | Raw correction pp |','|---|---|---:|---:|---:|---:|']
    def fmt(q):return f'{q["mean"]:+.6f} [{q["CI95"][0]:+.6f}, {q["CI95"][1]:+.6f}]'
    for ds,g in r['groups'].items():
        for m in ['nearest','target_high']:
            c=g[m]['coverage'];lines.append(f'| {ds} | {m} | {c.get("accepted_refs",0)}/{c.get("returned_refs",0)} | {g[m]["queries_with_feedback"]}/16 | {fmt(g[m]["gated_delta"])} | {g[m]["raw_delta"]["mean"]:+.6f} |')
    lines+=['','Correction quality is mean conditional output IoU minus mean identical input-reference IoU at known GT planned positions. Below-threshold or invalid candidates contribute zero correction; raw correction additionally shows returned boxes without threshold. Missing GT is unknown and does not count as failure. Each query contributes one mean, then domain means/paired query bootstrap; thousands of reference pairs are not thousands of independent samples. Unsupported queries contribute zero movement. This is a sparse-anchor interface test, not complete Student tube performance or an OPD score.','','| Domain | Target-high minus nearest gated quality pp [95% CI] | Prelocked gate |','|---|---:|---|']
    for ds,g in r['groups'].items():lines.append(f'| {ds} | {fmt(g["high_minus_nearest_gated_delta"])} | {g["gate_pass"]} |')
    lines+=['','Decision: `'+r['decision']['status']+'`. Coverage must improve in each domain AND admitted-policy quality must improve over unchanged input and nearest in each domain before bounded repaired OPD. Confidence intervals are reported, not used as an arbitrary significance gate. No new content rule, threshold tuning or expansion is authorized by a negative result.','','The nearest arm is checked against old fixed24 on exactly the same actions, scores, boxes and admission. All non-content decoder tensors, reference and attention-mask hashes match within each pair; cloned native-query box and target-score qualifications cover one original query per domain. Copying a high-score native content can still read a reference that points at a wrong object; a score gain alone is not a localization gain. Geometric IoU does not prove identity.','','Costs: see ACTUAL_COSTS.json. Existing Source and E generation excluded. Source captures0, pretrained updates0, new training0. Both conditions share one image/text encoding per anchor. Raw media, Query/parent IDs, coordinates, GT and weights stay private. Fixed actual-pixel panels only verify alignment, with no new semantic labels. CURRENT and paused historical queues unchanged.']
    P.mkdir(parents=True,exist_ok=True);(P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('See REPORT.md, RESULTS_PUBLIC.json, DECISION.json, CONFIGURATION.json, ROOT_INDEPENDENT_AUDIT.json and ACTUAL_COSTS.json.\n')
    review=D/'root_review';review.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,(ds,g)in zip(axs,r['groups'].items()):
        vals=[g[m]['gated_delta']['mean']for m in ['nearest','target_high']]
        ax.bar(['Nearest content','Target-high content'],vals,color=['#64788d','#b05c48']);ax.axhline(0,color='black',lw=.6);ax.set_title(ds+'16 matched references');ax.set_ylabel('Known-anchor gated correction (IoU pp)')
    fig.tight_layout();fig.savefig(P/'MATCHED_CONTENT.svg');fig.savefig(review/'MATCHED_CONTENT.png',dpi=130);plt.close(fig)
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    from vg_tta.stvg_error_state_correction_v1 import encode,decode
    cache=collections.OrderedDict();panels=[]
    for ds in r['groups']:
        rec=next(x for x in p['records']if x['dataset']==ds);z=load(D/'predictions'/(rec['alias']+'.pt'));e=load(ROOT/rec['prior_prediction']);row=read_row(ds,rec['query_ordinal']);frames,_,_=frames_for(ds,row,cache)
        k=next(i for i,d in enumerate(z['matched']['nearest']['details'])if d['legal']);refs=decode(encode(np.asarray(e['outputs']['Source_no_correction']))+z['actions'][k]*np.array(p['scale'][ds]))
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        for ax,pos in zip(axs,z['planned_positions']):
            image=frames[pos];h,w=image.shape[:2];ax.imshow(image)
            for m,col in [('reference','white'),('nearest','cyan'),('target_high','magenta')]:
                box=refs[pos]if m=='reference'else z['matched'][m]['details'][k]['anchors'][pos]['box'];cx,cy,bw,bh=box
                ax.add_patch(Rectangle(((cx-bw/2)*w,(cy-bh/2)*h),bw*w,bh*h,fill=False,color=col,lw=1.5))
            ax.set_title('anchor '+str(pos));ax.axis('off')
        fig.suptitle(rec['alias']+' same reference white / nearest cyan / target-high magenta; raw responses, not GT');f=review/(ds+'_actual_pixels.png');fig.savefig(f,dpi=100);plt.close(fig)
        panels.append(dict(alias=rec['alias'],dataset=ds,path=str(f.relative_to(ROOT)),sha256=sha(f),anchors=4))
    put(D/'ACTUAL_PIXEL_PANELS.json',dict(status='pending_root_actual_view',panels=panels,new_semantic_labels=0))
    own=['vg_tta/stvg_dino_content_match32_v1.py','scripts/run_stvg_dino_content_match32_v1.py','scripts/score_stvg_dino_content_match32_v1.py','scripts/audit_stvg_dino_content_match32_v1.py','scripts/report_stvg_dino_content_match32_v1.py','scripts/publish_stvg_dino_content_match32_v1.py','protocols/stvg_dino_content_match32_v1.md']
    files=['RESULTS_PUBLIC.json','DECISION.json','ROOT_INDEPENDENT_AUDIT.json','CONFIGURATION.json','ACTUAL_COSTS.json','REPORT.md','README.md','MATCHED_CONTENT.svg']
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=own+[str((P/f).relative_to(ROOT))for f in files]));print(dict(status='report_ready',files=len(own)+len(files)))


if __name__=='__main__':main()

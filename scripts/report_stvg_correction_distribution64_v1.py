"""Anonymous signed report and fixed private engineering pixel panels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load
from scripts.run_stvg_correction_distribution64_v1 import D,P,PRE,PAUSED,ARMS,verify


def main():
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from scripts.run_decota_paper_main_v1 import frames_for
    p=verify();assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';r=read(P/'RESULTS.json');pause=read(PAUSED/'PAUSE_RECEIPT.json')
    put(P/'ACTUAL_COSTS.json',dict(CPU_readout_wall_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['CPU_wall_seconds'],
        CPU_score_seconds=read(D/'CPU_COMPLETION.json')['CPU_score_seconds'],CPU_readout_case_seconds=sum(v['CPU_readout_seconds']for v in r['rows']),
        new_Source_forwards=0,new_model_calls=0,new_DINO_frames=0,new_Student_updates=0,new_training=0,
        cached_expert_planned_slots=256,baseline_Source_and_E_reused=64,
        previous_paused_parameter_experiment=dict(complete_queries=pause['complete_queries'],completed_fit_slots=pause['completed_fit_slots'],optimizer_steps=pause['completed_optimizer_steps'],target_GT_scored=False,included_in_current_CPU_readout_cost=False),
        cached_cost_not_first_online_end_to_end_latency=True))
    put(P/'ACTUAL_PROTOCOL.json',dict(status='complete',count=64,parents_per_domain=32,backbone='TA-STVG frozen original cross-domain Source',
        checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/source_expanded256/ACTUAL_PROTOCOL.json')['source_checkpoints'],
        frozen_E_beta=1/16,source_calibration_sha256=p['source_calibration_sha256'],
        selection=p['selection'],exact_predecessor_roster_plan_sha256=p['exact_predecessor_roster_plan_sha256'],
        weight_energy='Conditional LS assignment joint E energy; outlier cutoff + retained relative DINO support',
        weight_scale='Existing source cutoff, temperature multiplier1 fixed before new output/GT',
        assignment_mean_not_full_continuous_posterior=True,calibrated_identity_probability=False,
        duplicate_four_parameter_corrections_assignment_mass_summed=True,all_outlier_canonical_Source=True,
        mean_in_correction_chart_not_mean_frame_boxes=True,MBR_returns_one_actual_complete_candidate=True,
        MBR_utility=p['utility'],raw_boxes_not_clamped=True,final_MBR_time_exact_all_outputs=True,
        all64_output_seal_before_new_GT=True,GT_not_used_for_selection_scoring_or_readout=True,
        historically_exposed_development=True,independent_confirmation=False,
        old_source_calibration_parent_overlap_disclosed_in_Source256_report=True,
        LN_predecessor_paused_by_user=True,second_backbone_explicitly_not_started=True,
        CURRENT_MAIN_PAPER_unchanged=True,old_negative_results_preserved=True,no_new_model_or_optimizer=True))
    root=D/'root_review';root.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(12,7.5));fig.subplots_adjust(left=.1,right=.97,bottom=.15,top=.86,wspace=.32,hspace=.4)
    for i,ds in enumerate(['hc2','vidstg']):
        for j,ep in enumerate(['v','unobserved_actual_s']):
            ax=axs[i,j]
            for k,a in enumerate(ARMS[2:]):
                q=r['summary'][ds]['paired'][a+'_minus_E_MBR'][ep];m=q['mean'];lo,hi=q['CI95'];ax.errorbar(k,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#ac7145','#397c91'][k])
            ax.axhline(0,color='.5',lw=.8);ax.set_xlim(-.6,1.6);ax.set_xticks([0,1],['Correction mean','Trajectory MBR']);ax.set_ylabel('Change vs deterministic E (pp)');ax.set_title(f'{ds}: '+('full vIoU'if ep=='v'else'unobserved actual-frame sIoU'),loc='left',fontsize=10);ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Frozen correction distribution: same 32 parents per domain',fontsize=14);fig.text(.1,.035,'Same finite assignments and working weights; all output states frozen before GT.\nPaired parent bootstrap95% intervals; exposed development. No model or expert calls; no optimizer.',fontsize=9)
    fig.savefig(P/'CORRECTION_DISTRIBUTION.svg');fig.savefig(root/'CORRECTION_DISTRIBUTION.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));fig.subplots_adjust(bottom=.23,wspace=.28)
    for ax,ds in zip(axs,['hc2','vidstg']):
        rr=[v for v in r['rows']if v['dataset']==ds];x=[100*(v['values']['E_MBR']['v']-v['values']['Source_MBR']['v'])for v in rr]
        for a,color,mark in [('Correction_mean','#ac7145','o'),('Correction_MBR','#397c91','x')]:
            y=[100*(v['values'][a]['v']-v['values']['Source_MBR']['v'])for v in rr];ax.scatter(x,y,s=22,alpha=.75,color=color,marker=mark,label=a)
        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],lw=.8,color='.5');ax.axhline(0,lw=.7,color='.7');ax.axvline(0,lw=.7,color='.7');ax.set_title(ds);ax.set_xlabel('E − frozen Source full vIoU (pp)');ax.set_ylabel('New readout − frozen Source (pp)');ax.legend(fontsize=8)
    fig.text(.1,.035,'All 64 fixed queries retained. This sampled HC32 has E below Source on average; do not substitute the full128 mean.',fontsize=9)
    fig.savefig(P/'ALL_SIGNED_PARENTS.svg');fig.savefig(root/'ALL_SIGNED_PARENTS.png',dpi=140);plt.close(fig)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};views=[]
    colors=dict(Source_MBR='#2687de',E_MBR='#25b660',Correction_mean='#e0a035',Correction_MBR='#cc46b1')
    for ds in ['hc2','vidstg']:
        rec=next(v for v in p['records']if v['dataset']==ds);z=load(D/'predictions'/(rec['alias']+'.pt'));oldsource=load(ROOT/rec['prior_Source']);row=plans[ds][rec['query_ordinal']]
        frames,ids,_=frames_for(ds,row,collections.OrderedDict());assert ids==z['frame_ids'];oldinput=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2/inputs'/('vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg')/'clean'/f'{rec["query_ordinal"]:05}.pt'
        import torch,hashlib
        inp=torch.load(oldinput,map_location='cpu',weights_only=False);assert hashlib.sha256(frames.tobytes()).hexdigest()==inp['pixel_sha256'];assert sha(oldinput)==oldsource['original_input_sha256']
        panels=[]
        for slot,pos in enumerate(z['planned_positions']or[0]):
            im=Image.fromarray(frames[pos]).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im)
            for arm,color in colors.items():
                b=z['outputs'][arm][pos];draw.rectangle(((b[0]-b[2]/2)*w,(b[1]-b[3]/2)*h,(b[0]+b[2]/2)*w,(b[1]+b[3]/2)*h),outline=color,width=3)
            im=im.resize((480,max(1,round(h*480/w))));panel=Image.new('RGB',(480,im.height+45),'white');panel.paste(im,(0,45));dr=ImageDraw.Draw(panel);dr.text((5,3),f'{rec["alias"]} slot{slot} frame{ids[pos]}',fill='black');dr.text((5,23),'Blue Source | Green E | Orange mean | Magenta MBR',fill='black');panels.append(panel)
        sheet=Image.new('RGB',(480*len(panels),max(v.height for v in panels)),'white')
        for k,v in enumerate(panels):sheet.paste(v,(k*480,0))
        dest=root/(ds+'_fixed_input_output.png');sheet.save(dest);views.append(dict(alias=rec['alias'],dataset=ds,frames=len(panels),pixel_SHA256=inp['pixel_sha256'],path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
    put(D/'PIXEL_PANEL_GENERATION.json',dict(status='generated_pending_actual_root_view',count=sum(v['frames']for v in views),views=views,new_identity_labels=0))
    lines=['# Frozen output-space correction distribution64','','Latest user instruction pauses the former LN/query-residual update. Its two completed queries/four fits/160 optimizer steps are preserved and unscored. This experiment uses only existing frozen Source/E caches and the identical preselected HC32/Vid32 roster. No model, expert or optimizer call. The second backbone was explicitly deferred by the human.','','## Full-tube result','','All arms use the exact same final-distribution MBR time. Source is an existing same-clock reference, not an extra new variant.','','| Target | Frozen Source vIoU % | E vIoU % | Correction mean % | Trajectory MBR % | Mean−E pp | MBR−E pp |','|---|---:|---:|---:|---:|---:|---:|']
    for ds,g in r['summary'].items():
        vals=[100*g['query_mean'][a]['v']for a in ARMS]+[g['paired'][a+'_minus_E_MBR']['v']['mean']for a in ARMS[2:]];lines.append('| '+ds+' | '+' | '.join(f'{v:.6f}'for v in vals)+' |')
    lines+=['','| Target | Mean−E unobserved actual s pp | MBR−E unobserved actual s pp | MBR−E full v95% CI pp | Mean <−5pp vs E | MBR <−5pp vs E |','|---|---:|---:|---|---:|---:|']
    for ds,g in r['summary'].items():
        a,b=[g['paired'][x+'_minus_E_MBR']for x in ARMS[2:]];ci=b['v']['CI95'];lines.append(f'| {ds} | {a["unobserved_actual_s"]["mean"]:+.6f} | {b["unobserved_actual_s"]["mean"]:+.6f} | [{ci[0]:+.6f},{ci[1]:+.6f}] | {a["v"]["severe_loss_below_minus5pp"]}/32 | {b["v"]["severe_loss_below_minus5pp"]}/32 |')
    lines+=['','## Decision and scope','','Neither new readout improves deterministic E in both domains. Stop this fixed correction-distribution readout; retain E. Do not search its temperature, mixture weights or new readout rules. This is not evidence against all no-network-update methods or all correction distributions. It is not OPD: it changes output-space inference only. No second backbone, network adaptation or cross-video cache starts automatically.','','This random HC32 subset has E−Source = −0.446148pp, unlike the full128 positive mean; the entire fixed subset is retained. Vid E−Source = +5.836399pp. The overall256 result is not substituted into this64 table. Targets and prior calibration are historically exposed; no independent confirmation claim. Eight Vid queries have no admitted support and remain exact no-ops. Missing GT stays unknown. All negative/neutral outcomes and severe tails are in RESULTS.json.','','## Exact construction and limitations','','Every assignment chooses at most one retained DINO candidate per admitted frame, or pays the original outlier cutoff. Its conditional LS delta and joint assignment energy define a working Gibbs weight with the frozen source cutoff as scale. The all-rejected state has canonical Source delta0. Exact identical deltas sum their assignment mass. The global minimum matches deterministic E, but mass allocation changes the decision. There is no continuous posterior volume/Laplace normalization or calibrated identity probability.','','The mean is in (cx,cy,logw,logh) correction coordinates, not averaged frame boxes. MBR selects one existing complete trajectory with maximum expected physical-cell-weighted IoU on the shared predicted event support and original sampled grid. This approximates full-tube overlap; it is not the exact dense benchmark risk. Both use precisely the same candidates and weights; no per-frame identity switching. This supplies no new visual evidence and cannot repair arbitrary local identity changes.','','Mean maximum working mass/ESS:']
    for ds in ['hc2','vidstg']:
        rr=[v for v in r['rows']if v['dataset']==ds];lines.append(f'- {ds}: maximum mass {np.mean([v["distribution"]["maximum_weight"]for v in rr]):.6f}; ESS {np.mean([v["distribution"]["ESS"]for v in rr]):.6f}.')
    lines+=['','These are descriptive distribution diagnostics, not an independent causal diagnosis of semantic errors or a license to retune scale.','','## Verification and cost','',f'CPU readout wall {read(D/"GLOBAL_PREDICTION_BARRIER.json")["CPU_wall_seconds"]:.6f}s; zero new Source forward, expert frame, network update or training. Previously paused fits are excluded from this CPU cost and disclosed separately. Cached readout cost is not first online end-to-end latency. All64 output barriers preceded new GT evaluation; old Source/E baseline metrics and all tIoUs match exactly. Independent assignment/mass/mean/utility/metric/bootstrap audits passed. Fixed original pixel panels and aggregate plots require actual root view before publication.','','[All signed results, strata and CIs](RESULTS.json) · [Decision](DECISION.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Costs](ACTUAL_COSTS.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Paired intervals](CORRECTION_DISTRIBUTION.svg) · [All parents](ALL_SIGNED_PARENTS.svg).']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# Frozen correction distribution64\n\n[Report](REPORT.md), [all signed results](RESULTS.json), [protocol](ACTUAL_PROTOCOL.json).\n\nReproduction requires private original Source/E/DINO cache receipts: run_stvg_correction_distribution64_v1.py prepare then predict; score_stvg_correction_distribution64_v1.py; audit_stvg_correction_distribution64_v1.py; report_stvg_correction_distribution64_v1.py; actual root pixel/privacy review; publish_stvg_correction_distribution64_v1.py stage, authenticated two-ref CAS and verify. No backbone gradient or new expert. Previous LN experiment remains paused and unscored.\n')
    put(P/'VERIFICATION.json',dict(status='pass',count=64,independent=read(P/'ROOT_INDEPENDENT_AUDIT.json'),pixel_panels_pending_actual_view=True,media_original_query_parent_GTcoordinates_weights_tensors_excluded=True,old_Source_E_exact=True,all_time_invariants_exact=True))
    files=['vg_tta/stvg_correction_distribution64_v1.py','scripts/run_stvg_correction_distribution64_v1.py','scripts/score_stvg_correction_distribution64_v1.py','scripts/audit_stvg_correction_distribution64_v1.py','scripts/report_stvg_correction_distribution64_v1.py','scripts/publish_stvg_correction_distribution64_v1.py','protocols/stvg_correction_distribution64_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS.json','REPORT.md','README.md','DECISION.json','ACTUAL_PROTOCOL.json','ACTUAL_COSTS.json','ROOT_INDEPENDENT_AUDIT.json','VERIFICATION.json','CORRECTION_DISTRIBUTION.svg','ALL_SIGNED_PARENTS.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(status='report_generated',public_files=len(files),actual_pixels_pending=True)))


if __name__=='__main__':main()

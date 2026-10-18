"""Fixed-stream conditional report, source-only image views and export allowlist."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,collections,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_external_opd20_online64_v1 import D,P,verify
from vg_tta.stvg_external_opd20_v1 import ARMS,CONFIG


def main():
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from scripts.run_decota_paper_main_v1 import frames_for
    p=verify();audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass';r=read(P/'RESULTS.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');inputs=read(D/'INPUT_BARRIER.json');access=read(D/'ACCESS_READBACK.json')
    stats={ds:dict(expert=16,no_expert=16,valid_expert=sum(v['valid_expert']is True for v in r['rows']if v['dataset']==ds and v['role']=='expert'))for ds in ['hc2','vidstg']}
    put(P/'ACTUAL_PROTOCOL.json',dict(status='complete',count=64,primary_queries_per_domain=16,expert_queries_per_domain=16,
        arms=ARMS,external_variables=20,features='1 plus mean Source cx cy w h over all original decoder frames',configurations=CONFIG,
        role_order=[{k:v for k,v in q.items()if k in ['alias','dataset','role','domain_arrival']}for q in p['records']],
        exact_original64_plan_sha256=p['exact_prior64_plan_sha256'],split_rule=p['split_rule'],split_seed=p['split_seed'],counts=stats,
        W_zero_per_domain_independent_per_branch=True,W_only_cross_query_carry=True,Adam_reset_per_expert=True,
        action='One4d standardized Gaussian action transforms the entire tube; source scale from existing opposite-source E calibration',
        expert_outputs_identical_E=True,current_no_expert_teacher_unavailable=True,no_expert_optimizer_steps=0,
        input_projection='Only Source/logits ZIP storages0,1,2 and byteorder; no mixed-source pickle; source-derived clock and planned positions',
        raw_storage_Source_time_positions_independently_exact_on_all32_experts=True,old_A_baseline_metrics_exact_all64=True,
        numerical_dtype='CPU float64 external learner, original FP32 Source storages exactly converted',
        Source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/source_expanded256/ACTUAL_PROTOCOL.json')['source_checkpoints'],
        CI='paired parent bootstrap10000 conditional on one fixed SAVED online state history; not stream-order uncertainty',
        historically_exposed_development=True,independent_confirmation=False,all64_predictions_sealed_before_new_GT=True,
        CURRENT_MAIN_PAPER_unchanged=True,paused_LN_and_second_backbone_not_resumed=True,
        CPU_audit_numerical_recovery=read(D/'recovery/audit_precision002/RECOVERY.json')))
    put(P/'INPUT_ACCESS_AUDIT.json',dict(status='pass',count=64,Source_only_inputs=64,expert_only_inputs=32,
        no_expert_teacher_files=0,expert_open_events=len(access['expert_open_events']),
        expert_open_aliases=sorted({v['alias']for v in access['expert_open_events']}),
        nonexpert_current_teacher_opens=access['nonexpert_teacher_opens'],original_mixed_payload_opens=access['original_mixed_payload_opens'],
        model_GT_asset_opens=access['backbone_or_GT_asset_opens'],future_expert_input_hashes_not_pre_read=True,
        storage_projection={'allowed_members':['byteorder','data/0 Source','data/1 logits offset0','data/2 logits offset1'],'data.pkl_and_expert_storage_not_read':True},
        all_initial_zero_and_state_chains_verified=True,all_no_expert_states_unchanged=True))
    put(P/'ACTUAL_COSTS.json',dict(CPU_online_measure_seconds=bar['CPU_wall_seconds'],CPU_arrival_update_seconds=sum(v['CPU_arrival_seconds']for v in r['rows']),
        CPU_score_seconds=read(D/'CPU_COMPLETION.json')['CPU_score_seconds'],external_optimizer_steps=bar['external_optimizer_steps'],
        external_fit_slots=128,scheduled_expert_queries=32,valid_expert_queries=sum(v['valid_expert']for v in stats.values()),
        nominal_planned_expert_frames=128,new_expert_frames=0,new_backbone_forwards=0,backbone_updates=0,source_training=0,
        warm_cache_cost_not_first_online_end_to_end_latency=True))
    root=D/'root_review';root.mkdir(exist_ok=True)
    contrasts=['A_no_learning','C_sparse_OPD','F_fixed_rollout','B_Direct'];fig,axs=plt.subplots(2,2,figsize=(12,7.5));fig.subplots_adjust(left=.1,right=.97,top=.85,bottom=.15,hspace=.42,wspace=.3)
    for i,ds in enumerate(['hc2','vidstg']):
        for j,ep in enumerate(['v','s']):
            ax=axs[i,j]
            for k,c in enumerate(contrasts):
                q=r['groups'][ds]['no_expert']['paired']['D_dense_OPD_minus_'+c][ep];m=q['mean'];lo,hi=q['CI95'];ax.errorbar(k,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#327f85')
            ax.axhline(0,color='.5',lw=.8);ax.set_xlim(-.5,3.5);ax.set_xticks(range(4),['D−A','D−C','D−F','D−B']);ax.set_ylabel('No-expert paired change (pp)');ax.set_title(ds+' · '+('full vIoU'if ep=='v'else'dense GT sIoU'),loc='left');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('External20: transfer to16 no-expert arrivals per domain',fontsize=14);fig.text(.1,.035,'All current expert outputs are identical E. CIs condition on ONE saved online state history.\nA no learning; B Direct; C sparse OPD; D dense OPD; F fixed rollout. No backbone or new expert calls.',fontsize=9)
    fig.savefig(P/'NO_EXPERT_CONTRASTS.svg');fig.savefig(root/'NO_EXPERT_CONTRASTS.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.2));fig.subplots_adjust(bottom=.24,wspace=.25)
    for ax,ds in zip(axs,['hc2','vidstg']):
        rr=[v for v in r['rows']if v['dataset']==ds and v['role']=='no_expert']
        for arm,color in [('B_Direct','#bf7c38'),('D_dense_OPD','#327f85')]:ax.plot(range(1,17),[100*(v['values'][arm]['v']-v['values']['A_no_learning']['v'])for v in rr],marker='o',ms=3,label=arm,color=color)
        ax.axhline(0,color='.5',lw=.8);ax.set_xlabel('No-expert arrival index in fixed chronological stream');ax.set_ylabel('Full vIoU change vs A (pp)');ax.set_title(ds);ax.legend(fontsize=8)
    fig.text(.1,.04,'Every no-expert arrival is shown; no reordering by score. Expert outputs and fitting rewards are not the primary endpoint.',fontsize=9);fig.savefig(P/'NO_EXPERT_HISTORY.svg');fig.savefig(root/'NO_EXPERT_HISTORY.png',dpi=140);plt.close(fig)
    views=[];colors={'A_no_learning':'#2687de','B_Direct':'#e39b32','C_sparse_OPD':'#777777','D_dense_OPD':'#25b660','F_fixed_rollout':'#bf48bb'}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']}
    for ds in ['hc2','vidstg']:
        rec=next(v for v in p['records']if v['dataset']==ds and v['role']=='no_expert');z=load(D/'predictions'/(rec['alias']+'.pt'));assert z['teacher']is None
        row=plans[ds][rec['query_ordinal']];frames,ids,_=frames_for(ds,row,collections.OrderedDict());assert ids==z['frame_ids'];assert sha(Path(row['input']['video_path']))==row['input']['video_sha256'];panels=[]
        for slot,pos in enumerate(z['planned_positions']):
            im=Image.fromarray(frames[pos]).convert('RGB');w,h=im.size;dr=ImageDraw.Draw(im)
            for arm,color in colors.items():
                b=z['outputs'][arm][pos];dr.rectangle(((b[0]-b[2]/2)*w,(b[1]-b[3]/2)*h,(b[0]+b[2]/2)*w,(b[1]+b[3]/2)*h),outline=color,width=3)
            im=im.resize((480,max(1,round(h*480/w))));panel=Image.new('RGB',(480,im.height+62),'white');panel.paste(im,(0,62));pd=ImageDraw.Draw(panel);pd.text((5,3),f'{rec["alias"]} NO EXPERT arrival, Source-planned slot{slot}',fill='black');pd.text((5,23),'Blue A | Orange B | Gray C | Green D | Magenta F',fill='black');pd.text((5,43),'No current DINO or E read. No GT overlay.',fill='black');panels.append(panel)
        sheet=Image.new('RGB',(480*len(panels),max(v.height for v in panels)),'white')
        for k,v in enumerate(panels):sheet.paste(v,(480*k,0))
        out=root/(ds+'_no_expert_fixed_input_output.png');sheet.save(out);views.append(dict(alias=rec['alias'],dataset=ds,role='no_expert',frames=len(panels),path=str(out.relative_to(ROOT)),sha256=sha(out),original_video_sha256=row['input']['video_sha256'],decoded_frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),current_teacher_read=False))
    put(D/'PIXEL_PANEL_GENERATION.json',dict(status='generated_pending_actual_root_view',count=sum(v['frames']for v in views),views=views,new_semantic_labels=0))
    lines=['# External20 fixed-stream OPD transfer64','','Only external W4x5 updates. Backbone frozen, no new forward/expert call. Original HC32/Vid32 roster is split into16 expert and16 no-expert arrivals/domain by a score-independent locked rule, then interleaved. Main results below use NO-CURRENT-EXPERT arrivals only. Every expert arrival outputs identical E in all five arms.','','## Primary no-expert results','','| Arm | HC vIoU % | HC sIoU % | Vid vIoU % | Vid sIoU % |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        vals=[100*r['groups'][ds]['no_expert']['metrics'][a][ep]for ds in ['hc2','vidstg']for ep in ['v','s']];lines.append('| '+a+' | '+' | '.join(f'{v:.6f}'for v in vals)+' |')
    lines+=['','| No-expert contrast | HC Δv pp | HC95% CI | Vid Δv pp | Vid95% CI |','|---|---:|---|---:|---|']
    for c in contrasts:
        cols=[]
        for ds in ['hc2','vidstg']:
            q=r['groups'][ds]['no_expert']['paired']['D_dense_OPD_minus_'+c]['v'];cols.extend([f'{q["mean"]:+.6f}',f'[{q["CI95"][0]:+.6f},{q["CI95"][1]:+.6f}]'])
        lines.append('| D−'+c+' | '+' | '.join(cols)+' |')
    lines+=['','## Interpretation','','D−A: HC−0.073316pp, Vid+0.677554pp full vIoU. There is a Vid saved-stream transfer signal; no two-domain positive full-v screen. B−A means are HC+0.557365pp and Vid+0.548177pp, so the experiment does NOT show that all external-state learning fails. B and D also have positive spatial means in both domains, but overall tube benefit differs.','','D−C full v is HC−0.071877pp/Vid−0.074713pp: dense E coverage did not outperform the same teacher at four planned points. D−F is HC+0.141065pp/Vid−0.014959pp; it does not establish consistent benefit of on-policy resampling. D−B is HC−0.630680pp/Vid+0.129377pp; OPD is not established as superior to Direct. No automatic expansion or parameter sweep follows. Retain deterministic E and all limited positive cases; hold this fixed dense external-OPD variant. This is not a route-level impossibility claim.','','These paired bootstrap10000 intervals are CONDITIONAL ON A SINGLE SAVED ONLINE HISTORY. They do not rerun the state sequence under resampled parents, and are not independent stream/order confidence intervals. Cohort has historical development exposure; the no-expert condition is an inference-access constraint, not untouched-data status. No target scores choose step, W or output.','','## Mechanism and access verification','','psi reads only frozen Source trajectory means. Existing opposite-source geometry s scales a shared unit Gaussian correction; LR.01/tau.25/5steps/M32 are common across domains. W alone carries; Adam resets per expert. C/D resample around current Wpsi each step; F freezes the first actions/rewards/weights/rollout mean. No-current-expert arrival never opens an expert input and never updates W. A same-budget baseline emits Source there. All expert outputs are exact E even if a learner fits poorly, so current expert gains cannot contaminate the primary comparisons.','','Old Source archives also contain expert objects. The exporter reads only the first Source/logits raw storages and byteorder, not the pickle or expert storage; recomputes the same MBR/Source uniform positions. Thirty-two allowed expert cases independently verify raw Source/time/planned positions against their sealed E artifact; no teacher input is created for no-expert cases. The online process denies old mixed artifacts, GT/results/weights and teacher input except at the current expert arrival. See INPUT_ACCESS_AUDIT.json.','','## Cost, failures and limits','',f'CPU online run {bar["CPU_wall_seconds"]:.6f}s; actual external Adam steps {bar["external_optimizer_steps"]} of maximum640, including no-reference and exact-flat reward no-update cases. Zero backbone forward/update, expert or source-training calls; cached runtime is not first online end-to-end latency.']
    for ds,c in stats.items():lines.append(f'- {ds}:16 expert/16 no-expert; valid expert support{c["valid_expert"]}/16.')
    lines+=['','All64 outputs seal before target GT. No-reference/no-update/unknown and adverse outcomes remain in denominators. Five outputs have identical tIoU for every arrival; expert outputs match exactly. Complete no-expert/expert/all-arrival metrics, conditional CIs, negative tails, state/step displacement diagnostics and history are in RESULTS.json.','','Independent audits cover chronological state links, no-expert isolation/no-update, original fixed actions vs actual resampling, reward/likelihood gradients, Direct objective and Adam moments, metrics and aggregates. An audit-only correction preserves an overly strict float64 NumPy-vs-Torch dot-product bit-equality failure; same-call recorded means remain bit-exact and independent differences are checked numerically. No input, update, prediction or metric changed.','','Twenty variables and five trajectory statistics may be insufficient to transfer general corrections; this test does not identify that as the unique cause. E pseudo boxes introduce no new unobserved-frame visual evidence. Geometry feedback is not event-boundary supervision. Previous LN experiment remains user-paused; second backbone remains deferred.','','[All results and diagnostics](RESULTS.json) · [Protocol](ACTUAL_PROTOCOL.json) · [Access audit](INPUT_ACCESS_AUDIT.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Costs](ACTUAL_COSTS.json) · [Decision](DECISION.json) · [Primary CIs](NO_EXPERT_CONTRASTS.svg) · [Chronological history](NO_EXPERT_HISTORY.svg).']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# External20 online transfer64\n\n[Report](REPORT.md) and [all signed results](RESULTS.json). Run prepare/measure in run_stvg_external_opd20_online64_v1.py; then score, independent audit, report, actual fixed input/output view and privacy review; publish_stvg_external_opd20_online64_v1.py stage then connector ref updates and verify. Private original Source/E cache receipts are required and not redistributed. No network weight, LN, query-residual or new expert update.\n')
    put(P/'VERIFICATION.json',dict(status='pass',count=64,independent=audit,actual_pixel_view_pending=True,no_current_teacher_at_noexpert=True,all_expert_outputs_identical=True,all_time_outputs_identical=True,private_media_query_GTcoordinates_weights_states_excluded=True))
    files=['vg_tta/stvg_external_opd20_v1.py','scripts/run_stvg_external_opd20_online64_v1.py','scripts/score_stvg_external_opd20_online64_v1.py','scripts/audit_stvg_external_opd20_online64_v1.py','scripts/report_stvg_external_opd20_online64_v1.py','scripts/publish_stvg_external_opd20_online64_v1.py','protocols/stvg_external_opd20_online64_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS.json','REPORT.md','README.md','ACTUAL_PROTOCOL.json','ACTUAL_COSTS.json','INPUT_ACCESS_AUDIT.json','ROOT_INDEPENDENT_AUDIT.json','VERIFICATION.json','DECISION.json','NO_EXPERT_CONTRASTS.svg','NO_EXPERT_HISTORY.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(status='report_generated',files=len(files),actual_view_pending=True)))


if __name__=='__main__':main()

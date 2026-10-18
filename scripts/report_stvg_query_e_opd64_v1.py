"""Frozen-decoder QR256 evidence, costs, fixed image panels and public export."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,collections,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_query_e_opd64_v1 import D,P,ARMS,verify


def main():
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from scripts.run_decota_paper_main_v1 import frames_for
    p=verify();audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass';r=read(P/'RESULTS.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    stats={ds:dict(queries=32,valid_support=sum(v['active']for v in r['rows']if v['dataset']==ds),optimizer_steps={a:sum(v['fits'][a]['gradient_calls']for v in r['rows']if v['dataset']==ds)for a in ARMS[2:]})for ds in ['hc2','vidstg']}
    put(P/'ACTUAL_PROTOCOL.json',dict(status='complete',count=64,per_domain=32,arms=ARMS,active_variables=256,active_parameter='spatial.query_residual',configs=p['configs'],
        source_checkpoint=read(ROOT/'results/stvg_error_state/2026-10-18/source_expanded256/ACTUAL_PROTOCOL.json')['source_checkpoints'],
        cohort='Exact previously locked HC32+Vid32 from Source-expanded256; no reselection',initial='Frozen Source every query and arm; u=0 and empty Adam; episodic',
        backbone_LN_attention_bbox_head_frozen=True,teacher='Exact prior sealed E trajectory; float32 training reference shared by all fits; no refresh',
        sparse='Same E at all four original planned positions; not GT-known filtering',dense='All actual decoder frames, per-frame mean; no extra sparse term',
        original_Gaussian_control_variate_likelihood=True,native_endpoint_logit_extension=True,current_policy_resampling=['Sparse_OPD','Dense_OPD'],
        fixed_rollout='First samples/rewards/weights/rollout mean reused; likelihood remains evaluated at current Student',
        direct_loss='mean[5 L1 + 2(1-GIoU)]',all_arms_final_preplanned_step=True,final_student_not_E_overlay=True,
        temporal='Exact frozen Source final-distribution MBR for all six outputs; no temporal adaptation',
        source_prefix='Exact frozen-prefix reproduction where decoder input cache unavailable; only selected64; no historical Before',
        all64_sealed_before_GT=True,known_development_exposure=True,independent_confirmation=False,counts=stats,
        original_paused_LN_fit_outputs_unused=True,second_backbone_not_started=True,CURRENT_MAIN_PAPER_unchanged=True,audit_only_precision_recovery=read(D/'recovery/audit_precision002/RECOVERY.json')))
    costs=dict(GPU_lease_seconds=bar['GPU_lease_seconds'],Source_capture_seconds=sum(v['compute']['Source_capture_seconds']for v in r['rows']),
        fit_seconds={a:sum(v['compute']['fit_seconds'][a]for v in r['rows'])for a in ARMS[2:]},case_seconds=sum(v['compute']['case_seconds']for v in r['rows']),
        optimizer_steps=sum(sum(v['optimizer_steps'].values())for v in stats.values()),new_Source_prefix_captures=bar['new_Source_prefix_captures'],
        new_DINO_calls=0,backbone_weight_updates=0,LN_updates=0,new_source_training=0,max_query_variables=256,
        CPU_score_seconds=read(D/'CPU_COMPLETION.json')['CPU_score_seconds'],pilot2=read(D/'PILOT2_TIMING.json'),
        reused_E_and_DINO_first_acquisition_cost_not_counted_in_new_run=True)
    put(P/'ACTUAL_COSTS.json',costs)
    root=D/'root_review';root.mkdir(exist_ok=True)
    comparisons=['Source_MBR','E_MBR','Sparse_OPD','Fixed_rollout','Dense_Direct']
    fig,axs=plt.subplots(2,2,figsize=(13,7.6));fig.subplots_adjust(left=.09,right=.98,top=.88,bottom=.15,hspace=.4,wspace=.27)
    for i,ds in enumerate(['hc2','vidstg']):
        for j,ep in enumerate(['v','unobserved_actual_s']):
            ax=axs[i,j]
            for k,c in enumerate(comparisons):
                q=r['summary'][ds]['paired']['Dense_OPD_minus_'+c][ep];m=q['mean'];lo,hi=q['CI95'];ax.errorbar(k,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#267f83')
            ax.axhline(0,color='.5',lw=.8);ax.set_xticks(range(5),['Source','E','Sparse','Fixed','Direct']);ax.set_xlim(-.5,4.5);ax.set_ylabel('Dense OPD minus comparator (pp)');ax.set_title(ds+' / '+('full vIoU'if ep=='v'else'unobserved actual decoder sIoU'),loc='left');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Current-video adaptation: only query residual256 updates',fontsize=14);fig.text(.09,.035,'32 historical development parents per domain; fixed Source reset, teacher E and final MBR.\nPaired parent bootstrap10000. No pretrained/LN update, no current query E overlay.',fontsize=9)
    fig.savefig(P/'QUERY_ONLY_CONTRASTS.svg');fig.savefig(root/'QUERY_ONLY_CONTRASTS.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.2));fig.subplots_adjust(bottom=.23,wspace=.25)
    for ax,ds in zip(axs,['hc2','vidstg']):
        rr=[v for v in r['rows']if v['dataset']==ds]
        for a,col in [('E_MBR','#315ece'),('Dense_Direct','#d38a36'),('Dense_OPD','#279052')]:ax.plot(range(1,33),[100*(v['values'][a]['v']-v['values']['Source_MBR']['v'])for v in rr],marker='o',ms=2,label=a,color=col)
        ax.axhline(0,color='.5',lw=.8);ax.set_title(ds);ax.set_xlabel('Locked cohort order (not score sorted)');ax.set_ylabel('Full vIoU change vs Source (pp)');ax.legend(fontsize=8)
    fig.text(.09,.04,'Every parent shown, including no-reference/no-update and adverse outcomes. No cross-query state inheritance.',fontsize=9);fig.savefig(P/'QUERY_ONLY_ALL_PARENTS.svg');fig.savefig(root/'QUERY_ONLY_ALL_PARENTS.png',dpi=140);plt.close(fig)
    colors={'Source_MBR':'#2687de','E_MBR':'#e23d42','Sparse_OPD':'#777777','Dense_OPD':'#25b660','Fixed_rollout':'#bf48bb','Dense_Direct':'#e39b32'};views=[]
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']}
    for ds in ['hc2','vidstg']:
        rec=next(v for v in p['records']if v['dataset']==ds);z=load(D/'predictions'/(rec['alias']+'.pt'));row=plans[ds][rec['query_ordinal']]
        frames,ids,_=frames_for(ds,row,collections.OrderedDict());assert ids==z['frame_ids'];assert sha(Path(row['input']['video_path']))==row['input']['video_sha256'];panels=[]
        unobs=[k for k in range(len(ids))if k not in z['planned_positions']]
        extra=[unobs[k]for k in np.linspace(0,len(unobs)-1,min(4,len(unobs)),dtype=int)]
        positions=[('planned',k)for k in z['planned_positions']]+[('unobserved',k)for k in extra]
        for slot,(kind,pos) in enumerate(positions):
            im=Image.fromarray(frames[pos]).convert('RGB');w,h=im.size;dr=ImageDraw.Draw(im)
            for arm,col in colors.items():
                b=z['outputs'][arm][pos];dr.rectangle(((b[0]-b[2]/2)*w,(b[1]-b[3]/2)*h,(b[0]+b[2]/2)*w,(b[1]+b[3]/2)*h),outline=col,width=3)
            im=im.resize((480,max(1,round(h*480/w))));panel=Image.new('RGB',(480,im.height+62),'white');panel.paste(im,(0,62));pd=ImageDraw.Draw(panel);pd.text((5,3),f'{rec["alias"]} {kind} decoder position{pos}',fill='black');pd.text((5,23),'Blue Source | Red E | Gray Sparse | Green Dense',fill='black');pd.text((5,43),'Magenta Fixed | Orange Direct | No GT overlay',fill='black');panels.append(panel)
        ph=max(v.height for v in panels);sheet=Image.new('RGB',(480*4,ph*((len(panels)+3)//4)),'white')
        for k,v in enumerate(panels):sheet.paste(v,(480*(k%4),ph*(k//4)))
        out=root/(ds+'_fixed_input_output.png');sheet.save(out);views.append(dict(alias=rec['alias'],dataset=ds,frames=len(panels),path=str(out.relative_to(ROOT)),sha256=sha(out),original_video_sha256=row['input']['video_sha256'],decoded_frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest()))
    put(D/'PIXEL_PANEL_GENERATION.json',dict(status='generated_pending_actual_root_view',count=sum(v['frames']for v in views),views=views,new_semantic_labels=0))
    lines=['# Frozen-decoder query-only E adaptation64','','Exact historical HC32/Vid32, frozen Source reset and QR256=0 for every arm/query. E and original four planned positions are fixed. Only external query residual differentiates through frozen decoder; no pretrained or LN parameter changes. Final outputs are the Student, with no E overlay. This is spatial adaptation, not temporal or cross-video learning.','','## Results','','| Arm | HC vIoU % | HC unobserved actual sIoU % | Vid vIoU % | Vid unobserved actual sIoU % |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        vals=[100*r['summary'][ds]['query_mean'][a][ep]for ds in ['hc2','vidstg']for ep in ['v','unobserved_actual_s']];lines.append('| '+a+' | '+' | '.join(f'{v:.6f}'for v in vals)+' |')
    lines+=['','| Dense OPD minus | HC delta v pp | HC95% CI | Vid delta v pp | Vid95% CI |','|---|---:|---|---:|---|']
    for comp in comparisons:
        vals=[]
        for ds in ['hc2','vidstg']:
            q=r['summary'][ds]['paired']['Dense_OPD_minus_'+comp]['v'];vals.extend([f'{q["mean"]:+.6f}',f'[{q["CI95"][0]:+.6f},{q["CI95"][1]:+.6f}]'])
        lines.append('| '+comp+' | '+' | '.join(vals)+' |')
    lines+=['','## Scope and interpretation','','Each comparison changes the declared support, resampling or objective only. Sparse and Dense use the SAME E teacher; fixed rollout uses initial actions with the original likelihood evaluated at current policy. Direct uses the same E and mean5L1+2GIoU. All use identical domain-specific inherited learning rate and predetermined rounds, but actual parameter/output movements differ and are reported.','','Dense-vs-Sparse tests coverage; Dense-vs-Fixed tests resampling; Dense-vs-Direct tests OPD objective; Dense-vs-E tests whether adaptation earns current-query deployment benefit. Source benefit alone does not establish superiority over directly applying E. See DECISION.json for per-comparator two-domain endpoint checks. No automatic tuning, expansion or promotion follows this development panel.','','This cohort has historical exposure and source calibration overlap already disclosed in prior reports. Parent bootstrap intervals describe this fixed episodic panel; they do not establish independent confirmation. E pseudoboxes contain no new unobserved-frame expert information. Geometry IoU is not event-boundary feedback. Negative tails, no-update and unknown-support samples remain in denominators.','','## Cost and verification','',f'GPU lease {costs["GPU_lease_seconds"]:.6f}s; {costs["new_Source_prefix_captures"]} exact frozen Source prefix captures; {costs["optimizer_steps"]} query-only Adam updates; zero new DINO calls, zero pretrained/LN updates and zero source training. Existing E/DINO acquisition cost is reused, not erased. Pilot timing is included in the original64, not extra samples.','','All64 six-output predictions sealed before new target GT scoring. Original Source/E metrics exact; all time readouts/tIoU identical. Independent audit checks QR zero/reset, unchanged LN, original action/reward/gradient/Adam math, actual resampling vs frozen reuse, full two-pass readout, metrics and parent intervals. Fixed first-parent planned4 plus uniformly chosen unobserved4 actual-pixel panels per domain are generated for root inspection; no new semantic labels claimed. Previous LN experiment remains paused; second backbone not started.','','[All signed results](RESULTS.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Costs](ACTUAL_COSTS.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Decision](DECISION.json) · [Contrasts](QUERY_ONLY_CONTRASTS.svg) · [All parents](QUERY_ONLY_ALL_PARENTS.svg).']
    sens=[]
    for ds in ['hc2','vidstg']:
        for comp in ['Source_MBR','Sparse_OPD']:
            pair=r['summary'][ds]['paired']['Dense_OPD_minus_'+comp]['v'];vv=pair['per_parent_pp'];rr=[x for x in r['rows']if x['dataset']==ds];imax=int(np.argmax(vv));imin=int(np.argmin(vv))
            sens.append(f'- {ds}, Dense minus {comp}: largest positive {rr[imax]["alias"]} {vv[imax]:+.6f}pp; largest negative {rr[imin]["alias"]} {vv[imin]:+.6f}pp; leave-one-parent mean range [{pair["leave_one_saved_mean_range"][0]:+.6f},{pair["leave_one_saved_mean_range"][1]:+.6f}]pp.')
    lines+=['','## Post-hoc sensitivity and audit precision','','The primary means and all parents above remain unchanged. Leave-one-out sensitivity diagnoses concentration and is not a selection or exclusion rule.']+sens+['','The audit originally compared saved FP32 Adam moments with independently accumulated FP64 moments using an absolute-only tolerance. Its original failure is preserved. The corrected audit records absolute/relative errors and uses rtol2e-6 plus atol2e-5 for moments; parameter-step checks remain unchanged (maximum error below2e-8). No scientific inputs, update paths, predictions or scores changed or reran.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# Frozen decoder query-only E64\n\n[Report](REPORT.md) and [complete results](RESULTS.json). Run prepare/measure, then score, audit, report, actual image review, explicit publication and archive closure. Original private cache/media/weights required and excluded from this export. Only QR256 updates through the frozen decoder.\n')
    put(P/'VERIFICATION.json',dict(status='pass',count=64,independent=audit,actual_pixel_view_pending=True,all_pretrained_and_LN_weights_unchanged=True,all_time_readouts_identical=True,private_media_query_GTcoordinates_weights_states_excluded=True))
    files=['vg_tta/stvg_dense_e_opd64_v1.py','vg_tta/stvg_query_e_opd64_v1.py','scripts/run_stvg_query_e_opd64_v1.py','scripts/continue_stvg_query_e_opd64_v1.py','scripts/score_stvg_query_e_opd64_v1.py','scripts/audit_stvg_query_e_opd64_v1.py','scripts/report_stvg_query_e_opd64_v1.py','scripts/publish_stvg_query_e_opd64_v1.py','protocols/stvg_query_e_opd64_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS.json','REPORT.md','README.md','ACTUAL_PROTOCOL.json','ACTUAL_COSTS.json','ROOT_INDEPENDENT_AUDIT.json','VERIFICATION.json','DECISION.json','QUERY_ONLY_CONTRASTS.svg','QUERY_ONLY_ALL_PARENTS.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(status='report_generated',files=len(files),actual_view_pending=True)))


if __name__=='__main__':main()

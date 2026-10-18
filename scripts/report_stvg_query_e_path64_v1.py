"""Publish full cached paths and fixed-case figures, with no new optimization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load
from scripts.analyze_stvg_query_e_path64_v1 import D,P,OLD,ARMS,CASES,verify


def response(rows,plan):
    stats={ds:{arm:[]for arm in ARMS[:3]}for ds in ['hc2','vidstg']};by={x['alias']:x for x in rows}
    for rec in plan['records']:
        z=load(OLD/'predictions'/(rec['alias']+'.pt'));ds=rec['dataset']
        for arm in ARMS[:3]:
            f=z['fits'][arm]
            for k,(path,tr)in enumerate(zip(f['path'][:-1],by[rec['alias']]['fits'][arm]['transitions'])):
                rr=f['rounds'][k].get('rollout')
                if not tr['updated']or rr is None:continue
                mu=path['chart']['mean'].double().numpy();dw=rr['weights'].double().numpy()-1/32;ac=rr['samples'].double().numpy();old=rr['mean'].double().numpy()
                coef=1+dw.sum(1);assert (coef>0).all();opt=(old+(dw[:,:,None]*ac).sum(1))/coef[:,None]
                target_norm=float(np.linalg.norm(opt-mu));actual_norm=tr.get('actual_logit_mean_step_norm')
                ratio=actual_norm/target_norm if actual_norm is not None and target_norm>0 else None
                stats[ds][arm].append(dict(alias=rec['alias'],step_to=k+1,local_free_mean_optimum_step_norm=target_norm,
                    actual_decoder_logit_step_norm=actual_norm,response_to_local_target_ratio=ratio,ESS=f['rounds'][k]['ESS'],
                    first_order_parameter_loss_change=tr['predicted_first_order_loss_change'],same_rollout_loss_change=tr.get('same_rollout_loss_change'),
                    E_feedback_change_pp=tr['E_feedback_change_pp'],v_change_pp=tr['v_change_pp']))
    summary={ds:{arm:dict(actual_updates=len(items),finite_ratio_steps=sum(x['response_to_local_target_ratio']is not None for x in items),
        median_response_to_local_target_ratio=float(np.median([x['response_to_local_target_ratio']for x in items if x['response_to_local_target_ratio']is not None])),
        ratio_quantiles=np.quantile([x['response_to_local_target_ratio']for x in items if x['response_to_local_target_ratio']is not None],[.1,.5,.9]).tolist(),
        mean_ESS=float(np.mean([x['ESS']for x in items])),first_order_descent_but_surrogate_rise=sum(x['first_order_parameter_loss_change']< -1e-12 and x['same_rollout_loss_change']is not None and x['same_rollout_loss_change']>1e-5 for x in items))for arm,items in arms.items()}for ds,arms in stats.items()}
    return dict(formula='For each frozen rollout: c=1+sum(w-1/M); mu_opt=(mu_rollout+sum((w-1/M)*z))/c. Free-output Gaussian surrogate optimum, not attainable-output guarantee or deployment action.',
        interpretation='Ratio is a local scale diagnostic, not a GT-quality guarantee; high ratios can coexist with positive E/GT changes. No additional output or optimizer is generated.',summary=summary,steps=stats)


def main():
    import matplotlib.pyplot as plt
    p=verify();assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';r=read(P/'RESULTS.json');by={x['alias']:x for x in r['rows']}
    aux=response(r['rows'],p)
    if(P/'LOCAL_RESPONSE.json').exists():assert read(P/'LOCAL_RESPONSE.json')==aux
    else:put(P/'LOCAL_RESPONSE.json',aux)
    public={k:v for k,v in r.items()if k!='rows'};public['rows']=[];pathfiles=[]
    for ds in ['hc2','vidstg']:
        for arm in ARMS:
            name=f'PATHS_{ds}_{arm}.json';data=[dict(alias=x['alias'],dataset=ds,**x['fits'][arm])for x in r['rows']if x['dataset']==ds]
            (P/name).write_text(json.dumps(dict(count=32,arm=arm,dataset=ds,rows=data),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');pathfiles.append(name)
    for x in r['rows']:
        public['rows'].append({**{k:v for k,v in x.items()if k!='fits'},'fits':{a:{**{k:v for k,v in f.items()if k not in ['states','transitions']},'path_file':f'PATHS_{x["dataset"]}_{a}.json'}for a,f in x['fits'].items()}})
    public['all_state_paths_files']=pathfiles;put(P/'RESULTS_PUBLIC.json',public)
    put(P/'CASES.json',dict(fixed_before_analysis=CASES,rows=[by[a]for a in CASES],no_GT_step_selection=True))
    put(P/'ACTUAL_COSTS.json',dict(CPU_path_analysis_seconds=read(D/'CPU_COMPLETION.json')['CPU_seconds'],states=6656,transitions=6400,
        new_models=0,new_model_forwards=0,new_backwards=0,new_fits=0,new_DINO_calls=0,new_source_training=0,
        original_TTA_GPU_seconds=620.2086517240386,original_GPU_cost_not_new_diagnostic_cost=True))
    root=D/'root_review';root.mkdir(exist_ok=True)
    colors={'Sparse_OPD':'#777777','Dense_OPD':'#258a83','Fixed_rollout':'#ad5eaa','Dense_Direct':'#d4872b'}
    fig,axs=plt.subplots(5,3,figsize=(15,16.5));fig.subplots_adjust(top=.94,bottom=.06,hspace=.5,wspace=.25)
    for i,a in enumerate(CASES):
        row=by[a]
        for arm,col in colors.items():
            st=row['fits'][arm]['states'];k=[s['step']for s in st]
            for j,key in enumerate(['E_IoU_all','v','output_L2_from_Source']):axs[i,j].plot(k,[s[key]*(100 if j<2 else 1)for s in st],color=col,label=arm,lw=1.5)
        axs[i,1].axhline(100*row['original_final_values']['E_MBR']['v'],color='#c94545',ls=':',lw=1,label='E output')
        axs[i,1].axhline(100*row['original_final_values']['Source_MBR']['v'],color='#3d68b8',ls='--',lw=1,label='Source output')
        for j,title in enumerate(['mean overlap with E (%)','full vIoU (%)','actual output L2 from Source']):
            axs[i,j].set_title(a+' / '+title,loc='left',fontsize=10);axs[i,j].set_xlabel('Saved prescribed step');axs[i,j].spines[['top','right']].set_visible(False)
        if i==0:axs[i,0].legend(fontsize=7);axs[i,1].legend(fontsize=7)
    fig.suptitle('Fixed cases: E fitting, real localization, actual decoder response',fontsize=15)
    fig.text(.07,.015,'All saved steps shown. E/Source lines are fixed baselines. No GT-selected deployment step or new model call.',fontsize=10)
    fig.savefig(P/'CASE_PATHS.svg');fig.savefig(root/'CASE_PATHS.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(bottom=.23,wspace=.28)
    for ax,ds in zip(axs,['hc2','vidstg']):
        rows=[x for x in r['rows']if x['dataset']==ds]
        for x in rows:
            d=x['fits']['Dense_OPD']['diagnosis'];ax.scatter(d['E_change_pp'],d['v_change_pp'],c='#277f84'if x['active']else'#aaaaaa',s=24,alpha=.7)
            if x['alias']in CASES:ax.annotate(x['alias'],(d['E_change_pp'],d['v_change_pp']),fontsize=8,xytext=(4,3),textcoords='offset points')
        ax.axhline(0,color='.5',lw=.7);ax.axvline(0,color='.5',lw=.7);ax.set_title(ds+' Dense OPD / all32 parents');ax.set_xlabel('Final minus initial E overlap (pp)');ax.set_ylabel('Final minus initial full vIoU (pp)')
    fig.text(.08,.035,'Upper-right: E and GT improve; lower-right: fitting E yet real localization worsens.\nLower-left: E fitting and real localization both worsen. Gray points retain unsupported queries.',fontsize=9)
    fig.savefig(P/'ALL64_E_V_CHANGE.svg');fig.savefig(root/'ALL64_E_V_CHANGE.png',dpi=140);plt.close(fig)
    lines=['# Query-only E64: sealed update-path diagnosis','','CPU-only retrospective readback of every saved step from288f2876. No model forward/backward, fit, expert call, parameter change, new cohort or GT-selected deployment step. MBR, teacher E and query-only interface remain fixed.','','## Main findings','','The evidence supports prioritizing the actual update response before blaming all severe failures on E. H128 moves farther from E while real localization collapses; H008 sparse first improves its four-point objective, then loses both geometric and real quality and becomes flat-reward/no-update. Dense avoids that collapse but does not reproduce the full E benefit.','','Feedback alignment remains a distinct problem: in HC9 cases, Dense ends closer to E yet has lower real vIoU;4 cases lose both E overlap and vIoU. The latter4 account for39.208147 of50.427315 negative parent-pp (77.752%). Vid2 negative cases both have net E-overlap gain. These are descriptive categories, not pure causal allocations or identity labels. E overlap averages all decoder frames and differs from GT/event/interpolation evaluation support; this mismatch does not uniquely identify a bad teacher or wrong identity.','','## All64 final-state diagnosis','','| Domain / arm | Mean E overlap change pp | Mean vIoU change pp | E up / v down | E down / v down | Net E loss queries |','|---|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for arm in ARMS:
            g=r['summary'][ds][arm]['all'];cells=g['final_E_v_sign_counts'];lines.append(f'| {ds} / {arm} | {g["mean_final_E_change_pp"]:+.6f} | {g["mean_final_v_change_pp"]:+.6f} | {cells["up_down"]}/32 | {cells["down_down"]}/32 | {g["queries_with_net_E_loss"]}/32 |')
    lines+=['','Full64 and active56 summaries are separate in RESULTS_PUBLIC.json. Every path is available in the eight PATHS files; no-reference/no-update/unknown and positive cases remain. Signs use fixed1e-10 tolerance; a small numerical decrease is not itself a scientifically meaningful failure.','','## Fixed cases','','| Case / arm | E overlap initial -> final % | vIoU initial -> final % | Actual updates |','|---|---|---|---:|']
    for a in CASES:
        for arm in (['Sparse_OPD','Dense_OPD']if a=='H008'else['Dense_OPD','Dense_Direct']):
            f=by[a]['fits'][arm];s0,s1=f['states'][0],f['states'][-1];lines.append(f'| {a} / {arm} | {100*s0["E_IoU_all"]:.6f} -> {100*s1["E_IoU_all"]:.6f} | {100*s0["v"]:.6f} -> {100*s1["v"]:.6f} | {f["diagnosis"]["actual_updates"]} |')
    lines+=['','H008 sparse four-point E overlap67.286268% ->67.764? at step2 (exact trace retained), then50.9937% at step3 and0 at step6; vIoU60.524579% ->60.5841% ->48.2245% ->2.238566%, then unchanged through step40 because rewards become exactly flat. Dense final E overlap69.868977%, vIoU60.627595%. This is not simply a teacher-quality failure.','','H128 Dense E overlap28.854595% ->10.369711% and vIoU44.669483% ->10.389052%; Direct ends at E overlap27.604250% and vIoU45.203972%. Thus the severe Dense result is not evidence that faithfully fitting E alone caused this loss.','','V038 Dense first shows real improvement through early/middle saved steps but later loses it: E overlap75.438773% ->75.728148%, final vIoU56.983348% ->56.142706%. This combines a weak net E response with late deterioration. No intermediate step becomes a deployment rule.','','H119 remains a positive control: Dense E overlap47.153178% ->47.564909% while vIoU28.586077% ->36.957890%; a small all-frame E-overlap change can accompany a large task benefit, so small mean E gains cannot alone establish a response-capacity limit. V016 moves closer to a globally poorer E yet improves real vIoU, further showing that partial fitting need not reproduce the complete teacher result.','','## Local update response','','| Domain / Dense OPD | Actual updates | First-order parameter loss descent | Same-rollout surrogate rises | Negative mean-gradient/actual-output cosine |','|---|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        g=r['summary'][ds]['Dense_OPD']['all'];lines.append(f'| {ds} | {g["actual_updates"]} | {g["first_order_param_loss_descent_updates"]} | {g["same_rollout_surrogate_rise_updates"]} | {g["output_descent_cos_negative_updates"]} |')
    lines+=['','For each SAME frozen rollout, the original Gaussian likelihood/control-variate surrogate is recomputed before and after the actual saved decoder update. Parameter gradient dot actual step often predicts descent while finite decoder output raises that surrogate. H008 sparse step3 predicts−0.189964 but actual same-rollout loss rises about1350.908119, with output L2 step1.773065 and four-point E overlap down16.771291pp. H128 Dense step1 predicts−3.074680 but the same-rollout loss rises389.807091, output step1.320034 and E overlap down1.408254pp. These isolate an actual local-update mismatch rather than an incorrect GT checkpoint choice.','','The auxiliary local free-mean optimum is analytically derived from the existing frozen-rollout objective; its scale ratio is diagnostic only. It is neither a new output readout nor proof that every large step hurts. The surrogate includes a KL baseline and is not true task loss: H008 Dense can improve E and retain real localization even though this local surrogate rises. Do not turn these counts into an automatic gate or claim all Adam/sigma/temperature mechanisms have been causally isolated.','','## Decision scope','','Keep the same frozen-decoder query-only interface, fixed E/MBR and original predictions. Prioritize a future matched test of actual output-step/response control before modifying the teacher or merely increasing rounds/temperature. No such test is launched by this diagnosis. HC E-up/GT-down cases separately require feedback-use/metric-support investigation; optimization alone cannot be assumed to fix them. Do not splice cached paths, pick GT-best steps, infer identity errors from IoU, expand data or introduce temporal adaptation.','','[Summary and all64 rows](RESULTS_PUBLIC.json) · [Fixed case paths](CASES.json) · [Local response diagnostics](LOCAL_RESPONSE.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [All64 scatter](ALL64_E_V_CHANGE.svg) · [Case plots](CASE_PATHS.svg).']
    lines=[s.replace('67.764? at step2','67.765027% at step2')for s in lines]
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# Sealed query-only path64 diagnosis\n\n[Report](REPORT.md); [all64 summary and eight full-path file references](RESULTS_PUBLIC.json). Original fit/prediction files are immutable private inputs and excluded. No new model/fit or GT-best-step selection.\n')
    put(P/'DECISION.json',dict(status='complete',priority='actual_decoder_update_response_first_with_separate_feedback_alignment_diagnosis',
        no_new_update_or_deployment_rule_executed=True,no_GT_selected_steps=True,no_automatic_expansion=True,not_a_universal_OPD_failure=True))
    put(P/'VERIFICATION.json',dict(status='pass',count=64,independent=read(P/'ROOT_INDEPENDENT_AUDIT.json'),actual_chart_view_pending=True,
        old_inputs_predictions_steps_scores_unchanged=True,private_GT_coordinates_media_weights_raw_states_excluded=True,new_semantic_labels=0))
    files=['scripts/analyze_stvg_query_e_path64_v1.py','scripts/audit_stvg_query_e_path64_v1.py','scripts/report_stvg_query_e_path64_v1.py','scripts/publish_stvg_query_e_path64_v1.py','protocols/stvg_query_e_path64_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS_PUBLIC.json','CASES.json','LOCAL_RESPONSE.json','ACTUAL_COSTS.json','ROOT_INDEPENDENT_AUDIT.json','VERIFICATION.json','DECISION.json','REPORT.md','README.md','CASE_PATHS.svg','ALL64_E_V_CHANGE.svg']+pathfiles]
    src=ROOT/'external/TA-STVG/models/grounding_model/query_decoder.py';ln=next(i+1 for i,l in enumerate(src.read_text().splitlines())if 'pred_boxes = new_pred_boxes.detach()'in l)
    put(P/'STATIC_SOURCE_EVIDENCE.json',dict(status='read_only_current_source_fact',source_path=str(src.relative_to(ROOT)),sha256=sha(src),line=ln,statement='Intermediate predicted boxes are detached before entering the next spatial decoder layer. Finite recomputation refreshes them, while that dependency is absent from the autograd path.',causal_scope='Competing explanation only; no new finite-difference/model evaluation and no historical source reconstruction performed.',new_model_calls=0))
    with(P/'REPORT.md').open('a')as fh:fh.write('\n## Additional current-code boundary\n\nThe current PosDecoder refreshes intermediate predicted boxes via `pred_boxes = new_pred_boxes.detach()` before subsequent layers. Thus saved autograd directions do not include that reference-update dependency, while a real decoder recomputation refreshes the references. This is a competing explanation for finite-response mismatch; it does not establish how much came from stop-gradient, curvature, Adam momentum or step size. No new finite-difference/model test was run. [Current source evidence](STATIC_SOURCE_EVIDENCE.json).\n')
    files.append(str((P/'STATIC_SOURCE_EVIDENCE.json').relative_to(ROOT)))
    put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(status='generated',files=len(files),all64_full_paths_exported=True,actual_view_pending=True)))


if __name__=='__main__':main()

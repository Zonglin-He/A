"""Anonymous measured response report and standalone scientific figures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_query_e_response64_v1 import D,P,verify


def main():
    lock=read(D/'ANALYSIS_IMPLEMENTATION_LOCK.json')
    for f,h in lock['pins'].items():assert sha(ROOT/f)==h,f
    import matplotlib.pyplot as plt
    p=verify();assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';r=read(P/'RESULTS.json');by={x['alias']:x for x in r['rows']}
    review=D/'root_review';review.mkdir(exist_ok=True)
    # Export complete anonymous local outcomes and all scheduled controlled paths.
    public={k:v for k,v in r.items()if k!='rows'};public['rows']=[];shards=[]
    for ds in ['hc2','vidstg']:
        name='PATHS_'+ds+'.json';data=[dict(alias=x['alias'],dataset=ds,local=x['local'],fits=x['fits'])for x in r['rows']if x['dataset']==ds]
        (P/name).write_text(json.dumps(dict(count=32,rows=data),separators=(',',':'),allow_nan=False)+'\n');shards.append(name)
    for x in r['rows']:public['rows'].append({**{k:v for k,v in x.items()if k not in ['local','fits']},'path_file':'PATHS_'+x['dataset']+'.json'})
    put(P/'RESULTS_PUBLIC.json',public)
    cases=['H008','H128','H119','V038','V016'];put(P/'CASES.json',dict(predeclared_cases=cases,rows=[by[x]for x in cases]))
    old=read(ROOT/'results/stvg_error_state/2026-10-18/query_e_path64/RESULTS.json');oldby={x['alias']:x for x in old['rows']}
    fig,axs=plt.subplots(5,3,figsize=(14,16));fig.subplots_adjust(top=.95,bottom=.05,hspace=.5,wspace=.3)
    colors={'Sparse_control':'#a46d27','Dense_control':'#19847f'}
    for i,a in enumerate(cases):
        for arm,color in colors.items():
            st=by[a]['fits'][arm]['states'];original='Sparse_OPD'if arm.startswith('Sparse')else'Dense_OPD';ost=oldby[a]['fits'][original]['states']
            for j,key in enumerate(['v','E_IoU_all','output_L2_from_Source']):
                mult=100 if j<2 else 1
                axs[i,j].plot([v['step']for v in st],[mult*v[key]for v in st],label=arm,color=color)
                axs[i,j].plot([v['step']for v in ost],[mult*v[key]for v in ost],label=original,color=color,ls='--',alpha=.65)
        for j,title in enumerate(['full vIoU (%)','mean overlap with E (%)','actual output L2 from Source']):
            axs[i,j].set_title(a+' / '+title,fontsize=10,loc='left');axs[i,j].set_xlabel('Prescribed step');axs[i,j].spines[['top','right']].set_visible(False)
        if i==0:axs[i,0].legend(fontsize=7)
    fig.suptitle('Native query-only proposals with bounded actual-forward control',fontsize=14)
    fig.savefig(P/'CONTROL_PATHS.svg');fig.savefig(review/'CONTROL_PATHS.png',dpi=125);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(bottom=.2,wspace=.25)
    for ax,ds in zip(axs,['hc2','vidstg']):
        rr=[x for x in r['rows']if x['dataset']==ds]
        for arm,c in colors.items():
            orig='Sparse_OPD'if arm.startswith('Sparse')else'Dense_OPD'
            x=[100*(v['values'][orig]['v']-v['values']['Source_MBR']['v'])for v in rr];y=[100*(v['values'][arm]['v']-v['values']['Source_MBR']['v'])for v in rr]
            ax.scatter(x,y,label=arm,color=c,s=22,alpha=.7)
            for vv,xx,yy in zip(rr,x,y):
                if vv['alias']in cases:ax.annotate(vv['alias'],(xx,yy),fontsize=8)
        ax.axhline(0,color='.6',lw=.7);ax.axvline(0,color='.6',lw=.7);ax.plot([-40,15],[-40,15],color='.5',ls=':');ax.set_title(ds+' / all32');ax.set_xlabel('Original minus Source vIoU (pp)');ax.set_ylabel('Controlled minus Source vIoU (pp)');ax.legend(fontsize=8)
    fig.text(.08,.045,'Fixed cohort, fixed E/MBR, actual final scheduled step. Better surrogate optimization is not assumed to mean better GT.',fontsize=9)
    fig.savefig(P/'ALL64_CONTROL.svg');fig.savefig(review/'ALL64_CONTROL.png',dpi=140);plt.close(fig)
    lines=['# Query-only E64: actual decoder response and bounded step control','','All64 original development queries. Frozen pretrained network, QR256 only, fixed E and final-MBR. No new expert call, source training, cohort, backbone, temporal adaptation or hyperparameter sweep. Historical outputs stay unchanged. This is a matched mechanism experiment, not independent confirmation.','','## Complete adaptation results','','| Domain | Source | E | Sparse original | Sparse control | Dense original | Dense control | Fixed rollout | Dense Direct |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        g=r['summary'][ds];names=['Source_MBR','E_MBR','Sparse_OPD','Sparse_control','Dense_OPD','Dense_control','Fixed_rollout','Dense_Direct']
        lines.append('| '+ds+' | '+' | '.join(f'{100*g["means"][a]["v"]:.6f}'for a in names)+' |')
    lines+=['','All numbers above are full vIoU percent at the predefined last step. Complete t/s, exact unobserved decoder support, positive/negative/neutral and severe-tail counts are in RESULTS_PUBLIC.json. Every path, trial scale, displacement, local factor output and FD value is in PATHS_hc2.json/PATHS_vidstg.json. Unsupported8 Vid queries remain in all denominators.','','| Domain / paired comparison | Full vIoU delta pp [parent 95% CI] | Unobserved sIoU delta pp [95% CI] |','|---|---:|---:|']
    for ds in ['hc2','vidstg']:
        for name,g in r['summary'][ds]['paired'].items():
            def show(v):return 'unknown'if v['mean_pp']is None else f'{v["mean_pp"]:+.6f} [{v["CI95_pp"][0]:+.6f}, {v["CI95_pp"][1]:+.6f}]'
            lines.append(f'| {ds} / {name} | {show(g["v"])} | {show(g["unobserved_actual_s"])} |')
    lines+=['','## Matched local factorial','','192 predetermined Dense states: three nominal early/middle/late pre-rounds per query. Each of eight proposals shares the SAME historical QR/LN state, teacher, actions, weights and original Adam second-moment metric. First compare historical versus current-only first moment and native-detach versus full-chain gradient at equal QR norm; then original versus half norm. Main effects average all other factors, then three states within parent. They describe one-step responses, NOT full adaptation variants. Two extra H008 Sparse states are separate descriptive checks.','','| Domain / local factor | vIoU effect pp [parent 95% CI] | E overlap effect pp | Same-rollout surrogate change |','|---|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for factor in ['half_minus_full','current_minus_history','chain_minus_detach']:
            g=r['local_factor_effects'][ds][factor];v=g['v'];lines.append(f'| {ds} / {factor} | {v["mean_pp"]:+.6f} [{v["CI95_pp"][0]:+.6f}, {v["CI95_pp"][1]:+.6f}] | {g["E"]["mean_pp"]:+.6f} | {g["loss"]["mean"]:+.6f} |')
    lines+=['','The full-chain intervention removes only the internal reference stop-gradient in one decoder instance. Its forward boxes are bit-identical; the vendor source and pretrained tensors are unchanged. Current-only first moment is a direction control: the original second moment and total proposal norm stay fixed, so this is not a new Adam tuning recipe. Factor interactions can remain; no additive causal allocation of all failures is claimed.','','## Finite-difference check','','| Domain | Direction/h checks | Median absolute native-autograd error | Median absolute chain-autograd error | Native / chain sign disagreements |','|---|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        fd=r['local_factor_effects'][ds]['FD'];lines.append(f'| {ds} | {fd["count"]} | {fd["native_abs_error_median"]:.6f} | {fd["chain_abs_error_median"]:.6f} | {fd["native_wrong_sign"]} / {fd["chain_wrong_sign"]} |')
    lines+=['','Both fixed central-difference scales (.001 and .0005 QR-unit displacement) are reported, including numerical inconsistency. These are checks of the actual deterministic forward surrogate at that state, not GT gradients or proof that a full-chain optimizer would improve final localization. Sign counts near numerical zero should not be interpreted as substantive errors.','','## Bounded native-gradient control','','Original Adam proposes the QR step. Up to four actual decoder trials use scales1,.5,.25,.125, accepting the FIRST same-rollout FP64 surrogate nonincrease within tolerance1e-7*(1+abs(before)). No candidate/weight refresh within retries. Accepted scaled steps keep the proposed Adam moments; complete rejection restores QR and all optimizer state. Next original scheduled round resamples around the actual accepted Student. No extra rounds, teacher refresh, GT checkpoint or event-boundary rule. Actual output displacement is measured, not used as an independently tuned quality gate.','','| Domain / arm | Accepted | Rejected/restored | Flat reward | Accepted scale counts | Mean final output L2 |','|---|---:|---:|---:|---|---:|']
    for ds in ['hc2','vidstg']:
        for a,g in r['summary'][ds]['control'].items():lines.append(f'| {ds} / {a} | {g["accepted"]} | {g["restored"]} | {g["flat_rewards"]} | {json.dumps(g["scales"],sort_keys=True)} | {g["mean_final_output_L2"]:.6f} |')
    lines+=['','## Predeclared cases','','| Case | Source | E | Dense original | Dense control | Sparse original | Sparse control |','|---|---:|---:|---:|---:|---:|---:|']
    for a in cases:lines.append('| '+a+' | '+' | '.join(f'{100*by[a]["values"][arm]["v"]:.6f}'for arm in ['Source_MBR','E_MBR','Dense_OPD','Dense_control','Sparse_OPD','Sparse_control'])+' |')
    lines+=['','H008/H128 test prevention of abrupt failures; H119 is the positive-control preservation check. V038 and V016 retain feedback/task disagreement. A nearly frozen output may protect Source while discarding useful updates: report both the output displacement and task gains, rather than declaring all rejection protective. Proxy improvement and teacher overlap are not sufficient conditions for real task benefit.','','## Scope and audit','','All64 local/control predictions sealed before new GT evaluation. tIoU is invariant. Independent audit verifies all reported metric states, known unobserved support, original gradients/next outputs, matched QR norms, accepted-step geometry, Adam moment recurrences and exact QR/optimizer restoration. Rejected trial loss records are checked against their rule, not independently re-forwarded on GPU; this limitation is explicit. No new identity/role annotation. The two scientific charts receive actual root review.','','This test does not authorize an automatic sweep, full run, new student or teacher. Keep E/MBR as fixed baselines and choose further work from the measured split between direction/response control and feedback alignment; do not promote a development result to CURRENT.','','[All64 summary](RESULTS_PUBLIC.json) · [All HC paths](PATHS_hc2.json) · [All Vid paths](PATHS_vidstg.json) · [Fixed cases](CASES.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Controlled paths](CONTROL_PATHS.svg) · [All64 outcomes](ALL64_CONTROL.svg).']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# Fixed E/MBR QR256 response64\n\n[Report](REPORT.md). Full anonymous results, all local outcomes and complete scheduled controlled paths. Private coordinates/raw states/media/text are excluded.\n')
    put(P/'ACTUAL_COSTS.json',dict(GPU_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_seconds'],CPU_score_seconds=read(D/'CPU_COMPLETION.json')['CPU_seconds'],
        Source_captures=64,new_local_slots=194,new_fit_slots=128,new_DINO_calls=0,new_source_training=0,new_Temporal_updates=0,
        control_forwards=sum(f['compute']['forwards']for x in r['rows']for f in x['fits'].values()),control_backwards=sum(f['compute']['backwards']for x in r['rows']for f in x['fits'].values()),
        local_forwards=sum(l['compute']['forwards']for x in r['rows']for l in x['local']),local_backwards=sum(l['compute']['backwards']for x in r['rows']for l in x['local']),
        Source_capture_seconds=sum(x['compute']['Source_capture_seconds']for x in r['rows']),fit_seconds={a:sum(x['compute']['fit_seconds'][a]for x in r['rows'])for a in ['Sparse_control','Dense_control']},not_complete_first_online_expert_latency=True))
    put(P/'VERIFICATION.json',dict(status='pass',count=64,independent_audit=read(P/'ROOT_INDEPENDENT_AUDIT.json'),actual_chart_review_pending=True,
        registries_and_original_predictions_unchanged=True,new_semantic_labels=0))
    put(P/'DECISION.json',dict(status='complete',automatic_expansion=False,production_promotion=False,local_factor_not_full_method_comparison=True,
        surrogate_descent_not_GT_success=True,summary=r['summary']))
    files=['vg_tta/stvg_query_e_response64_v1.py','scripts/run_stvg_query_e_response64_v1.py','scripts/score_stvg_query_e_response64_v1.py','scripts/audit_stvg_query_e_response64_v1.py','scripts/report_stvg_query_e_response64_v1.py','scripts/publish_stvg_query_e_response64_v1.py','protocols/stvg_query_e_response64_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS_PUBLIC.json','CASES.json','ROOT_INDEPENDENT_AUDIT.json','VERIFICATION.json','ACTUAL_COSTS.json','DECISION.json','REPORT.md','README.md','CONTROL_PATHS.svg','ALL64_CONTROL.svg']+shards]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(dict(status='generated',files=len(files)))


if __name__=='__main__':main()

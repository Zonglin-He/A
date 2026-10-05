"""Complete paired readout, acquisition and pre-update memory report."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_three_scope_common_v1 import *
import numpy as np

def fmt(z,scale=100):return f"{z['mean']*scale:+.4f} [{z['ci95'][0]*scale:+.4f}, {z['ci95'][1]*scale:+.4f}]" if z.get('mean') is not None and z.get('ci95') is not None else 'not estimable'

def write_identical_or_new(path,value):
    if path.exists():assert read(path)==value,'A report revision cannot change sealed decisions or case rows'
    else:write(path,value)

def main():
    verify();assert read(PUB/'ROOT_AUDIT.json')['status']=='pass'
    s=read(PUB/'scope/SUMMARY.json');b=read(PUB/'boundary/SUMMARY.json');m=read(PUB/'memory/SUMMARY.json')
    bd=read(PUB/'boundary/DECISION.json');sd=read(PUB/'scope/DECISION.json');md=read(PUB/'memory/DECISION.json')
    conditional=dict(temporal514='qualified_pending_actual_runtime' if bd['temporal_gradient_qualified'] else 'skipped_P0_did_not_qualify',
        retrieval='qualified_pending_actual_runtime' if md['memory_qualified'] else 'skipped_preupdate_P0_did_not_qualify',
        independent_scope_streams_executed=True,production_promoted=False)
    write_identical_or_new(BASE/'CONDITIONAL_STAGE_STATUS.json',conditional);write_identical_or_new(PUB/'CONDITIONAL_STAGE_STATUS.json',conditional)
    lines=['# Native temporal scope, dual-offset P0 and pre-update LN compatibility','',
        'The three tests are separate. TTS changes acquisition only; boundary P0 changes temporal readout only; memory P0 predicts previously measured isolated utility with pre-update features. A conditional NO-GO is a decision for this locked mechanism, not a proof that all temporal adaptation or memory is impossible.','',
        'Actual fixed panel: each dataset 32 development + 16 source-disjoint confirmation, one query/source, two orders, clean + five original 5% corruptions. All have historical exposure. Confirmation is not fresh. Match the prior Top1 working point with a 100% scheduled-expert rate in both episodic and independent online streams; admission/eligibility may still produce no update. Official same-domain checkpoints, one frozen DINO, original admission + Top1 singleton energy, Adam .03, joint1792, ten steps/own-loss first minimum, Native time, reset query/Adam, selected LN delta/16. Deployed CURRENT_METHOD unchanged.','',
        '## TTS acquisition: current and actual online outputs','',
        'Uniform4 reference is the previous verified actual stream. TTS uses independent evolving LN; it never reuses Uniform post-update state. All2304 TTS logical outputs were sealed before scoring. Frozen uses identical Native. Unit pp, source-macro and 10000 paired source-bootstrap.','',
        '| Dataset/panel | Stream | Uniform−Frozen Δv | TTS−Frozen Δv | TTS−Uniform Δv | TTS−Uniform Δs | TTS >5/>20 harm vs Frozen |','|---|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            for stream in ['episodic','online100']:
                z=s[ds][sp][stream]['corruption'];a=z['metrics'];u=a['vs_frozen_v']['mean']-a['vs_uniform_v']['mean'];t=z['tails']
                lines.append(f"| {ds}/{sp} | {stream} | {u*100:+.4f} | {fmt(a['vs_frozen_v'])} | {fmt(a['vs_uniform_v'])} | {fmt(a['vs_uniform_s'])} | {t['vs_frozen_harm_gt5pp']}/{t['vs_frozen_harm_gt20pp']} |")
    lines+=['','TTS scope qualified on both confirmation corruption panels: **'+str(sd['scope_benefit_established'])+'**. No automatic promotion. Per-order/condition/clean, Before and matched net-memory results are all retained in scope/SUMMARY.json. Current gain does not establish memory gain.','',
        '| Dataset/confirm | TTS Before−Frozen | TTS current After−Before | TTS online−matched episodic | Uniform online−matched episodic |','|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        a=s[ds]['confirm']['online100']['corruption']['metrics'];lines.append(f"| {ds} | {fmt(a['before_vs_frozen_v'])} | {fmt(a['current_v'])} | {fmt(a['net_memory_v'])} | {fmt(a['uniform_net_memory_v'])} |")
    sr=read(PUB/'scope/ROWS.json');lines+=['','The safety comparison counts logical corruption arrivals, not independent sources; each source has five conditions and two orders. Confirmation has160 such arrivals per dataset, development320.','',
        '| Dataset/panel, online | Uniform >5/>20 pp harm vs Frozen | TTS >5/>20 pp harm vs Frozen |','|---|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            rr=[q for q in sr if q['dataset']==ds and q['split']==sp and q['condition']!='clean' and q['stream']=='online100']
            u=[sum(q['uniform_v']-q['frozen_v'] < -n/100 for q in rr) for n in [5,20]];t=s[ds][sp]['online100']['corruption']['tails']
            lines.append(f"| {ds}/{sp} | {u[0]}/{u[1]} | {t['vs_frozen_harm_gt5pp']}/{t['vs_frozen_harm_gt20pp']} |")
    lines+=['','TTS retains positive confirmation gains over Frozen, but its incremental confirmation gains over Uniform include zero in both datasets. The hoped-for common reduction of development severe tails is not established: Vid worsens while HC improves in the logical >20pp count. The matched online-minus-episodic effect is small, uncertain in Vid and positive in this HC panel; the Before effect alone cannot substitute for that net comparison. Keep Uniform4 as the working baseline, without claiming that every target-aware acquisition is ineffective.','']
    ev=read(PUB/'scope/EVIDENCE_SUMMARY.json');lines+=['','## Evidence quality','',
        'Rates divide by requested slots, including unavailable/empty evidence. IoU mass is admitted Top1 GT-IoU summed over exactly GT-scored observations divided by slots; it combines coverage and quality and is not conditional identity accuracy. Conditional means, valid/scored/admitted counts, best/worst and every observation are in EVIDENCE_ROWS.json. No scored GT at a selected frame stays missing.','',
        '| Dataset/panel | Δvalid rate | Δadmission rate | Δobserved-event-frame rate | Δadmitted/event GT-IoU mass |','|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            a=ev[ds][sp]['corruption']['metrics'];lines.append(f"| {ds}/{sp} | {fmt(a['delta_valid_rate'])} | {fmt(a['delta_admission_rate'])} | {fmt(a['delta_event_rate'])} | {fmt(a['delta_admitted_event_GT_mass'])} |")
    lines+=['','The event-frame rate counts actual observations in the GT event divided by requested slots; unavailable observations do not count as event hits. Conditional observation quality is descriptive: a different acquisition rule also changes which frames have scored GT. It is not an identity-accuracy label.','',
        '| Dataset/panel | Acquisition | valid/admitted/scored-valid | Mean scored Top1 GT-IoU | Mean admitted/scored GT-IoU |','|---|---|---:|---:|---:|']
    er=read(PUB/'scope/EVIDENCE_ROWS.json')
    for ds in DATASETS:
        for sp in ['search','confirm']:
            rr=[q for q in er if q['dataset']==ds and q['split']==sp and q['condition']!='clean']
            for arm in ['uniform','tts']:
                aa=[q[arm] for q in rr];nv=sum(q['valid'] for q in aa);na=sum(q['admitted'] for q in aa);ns=sum(q['scored_valid'] for q in aa);nas=sum(q['admitted_GT_scored'] for q in aa)
                raw=f"{100*sum(q['top1_GT_IoU_sum'] for q in aa)/ns:.2f}%" if ns else 'not estimable'
                admitted=f"{100*sum(q['admitted_GT_IoU_sum'] for q in aa)/nas:.2f}%" if nas else 'not estimable'
                lines.append(f'| {ds}/{sp} | {arm} | {nv}/{na}/{ns} | {raw} | {admitted} |')
    lines+=['','Confirmation evidence does not show a shared improvement: admitted/event GT-IoU mass has a negative point difference and a CI spanning zero in both datasets. HC obtains more observed event frames at the point estimate while its conditional admitted observation IoU decreases; event coverage and referent quality are separate. Conditional means above describe changed observation populations, not a causal same-frame identity test.','',
        '## Dual-offset boundary P0','',
        'Physical alignment interpolates discrete probability values on existing sampled frames, floors1e−12 and renormalizes. It is not a calibrated density or independent expert likelihood. Geometric endpoint means and strict i<j MAP; only evaluation interval changes, fixed Uniform Top1 spatial output. All1152 old online100 inputs covered and new readouts sealed before GT.','',
        '| Dataset/panel | Consensus−Native Δt | Δv | >5/>20 v harm |','|---|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            z=b[ds][sp]['consensus']['corruption'];t=z['tails'];lines.append(f"| {ds}/{sp} | {fmt(z['metrics']['delta_t'])} | {fmt(z['metrics']['delta_v'])} | {t['vs_reference_harm_gt5pp']}/{t['vs_reference_harm_gt20pp']} |")
    lines+=['','Temporal514 qualification: **'+str(bd['temporal_gradient_qualified'])+'**. Status: '+conditional['temporal514']+'. No gradient result is invented for a skipped stage. JS/error association is diagnostic only (boundary/JS_DIAGNOSTIC.json); all four source-bootstrap intervals for its association with Native error include zero.','',
        '## Pre-update memory P0','',
        'All939 exact isolated donor/recipient pairs are retained, including zero recipient-correction cases. Utility is actual selected donor LN write/16, source-initialized recipient Before−Frozen before a second nonzero write. It is not a full delta intervention or accumulated long-stream effect. Features: frozen raw-query RoBERTa masked mean768, Native final spatial latent mean256, box/time/TTS summaries. No recipient correction, future gradient or GT quality enters keys.','',
        'The ridge predictor uses existing GT-derived utility labels in development, alpha1 fixed. Both held-out donor and recipient nodes are excluded across all roles/conditions; confirmation fits development only. This is explicitly a supervised compatibility diagnostic, not an unlabeled deployed quality head. Similarity controls use no labels. Bootstrap uses common source-node weights in both roles and recipient-source-equal base weights; exact-zero utility is retained except binary AUC.','',
        '| Dataset/panel | Pre-update signal | corr(Uhat,U) | help/harm AUC | pairs/nodes |','|---|---|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            for a in ['query_cos','spatial_cos','ridge','constant']:
                z=m[ds][sp]['corruption'][a];lines.append(f"| {ds}/{sp} | {a} | {fmt(z['correlation'],1)} | {fmt(z['AUC'],1)} | {z['pairs']}/{z['source_nodes']} |")
    lines+=['','Memory qualification: **'+str(md['memory_qualified'])+'**; passing shared signals '+str(md['passing_signals'])+'. Status: '+conditional['retrieval']+'. Three predeclared signals were tested; qualification only authorizes a separate matched memory trial, not production. Clean results and all pair predictions remain public.','',
        'Only14/15 independent confirmation source nodes contribute to the Vid/HC corruption pair panels, despite200/206 pair rows. Wide node-bootstrap intervals matter. Positive spatial-cosine point correlations do not establish a deployable predictor, and the NO-GO does not prove that transfer relations or conditional memory never exist. Development constant predictions differ across held-out folds because training means differ; confirmation constant predictions are truly constant and their correlation is undefined.','',
        '## Costs, cases and limits','',
        '576 unique native inputs reuse encoder H; no video backbone forward. Frozen TTS heads and text encoder plus cached native decoder are real compute. Newly chosen frames need the same DINO; overlapping observations are reused only with identical RGB/text/context receipts. Worker time is wall time, not isolated GPU kernel time. Exact calls/reuse/backward counts are in scope/COST.json.','',
        'Independent root audit checks physical selections, all939 ridge predictions with a separate Cholesky calculation, double-role source exclusion, all optimizer paths/state chains and official dense scores. Public audit recomputes anonymous aggregates and source/node-bootstrap CIs, with an independent weighted Mann-Whitney AUC calculation; it cannot re-run private GT or private feature extraction. No harmful or failed row is discarded. Historical good confirmation and harmful development cases are preserved. An initial additional root-readback assertion exposed mixed float32/float64 normalization in the auditor; its saved engineering repair uses float64 throughout and preserves strict tolerances, all predictions, gradients and GT metrics. See ENGINEERING_RECOVERY.json.','',
        'Results concern these native TTS classifiers, this physical-grid geometric barycenter, and these pre-update keys with fixed ridge; they do not establish failure of every routing, temporal self-supervision or conditional memory. No new full-query job, external expert, scorer sweep, latent readout repair or production promotion was executed.','',
        '## Actual integration','',
        'Frozen / S-only / Tscope+S are the actual independent per-query and online100 results above. Conditional T+S and retrieval arms are run only if their locked P0 qualifies; skipped statuses are explicit. Scope candidate performance is retained even if it fails qualification.','']
    rows=read(PUB/'scope/ROWS.json');cases={}
    for ds in DATASETS:
        for sp in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean' and r['stream']=='online100'];cases[ds+'_'+sp]=dict(worst_vs_uniform=sorted(rr,key=lambda q:q['vs_uniform_v'])[:8],best_vs_uniform=sorted(rr,key=lambda q:-q['vs_uniform_v'])[:8])
    write_identical_or_new(PUB/'CASES.json',cases)
    lines+=['## Representative paired negative and recovery cases','', '| Dataset/panel | source/condition/order | TTS−Uniform v pp | TTS−Frozen v pp |','|---|---|---:|---:|']
    for key,g in cases.items():
        for label in ['worst_vs_uniform','best_vs_uniform']:
            q=g[label][0];lines.append(f"| {key}/{label} | {q['source_id']}/{q['condition']}/{q['order']} | {q['vs_uniform_v']*100:+.3f} | {q['vs_frozen_v']*100:+.3f} |")
    (ROOT/'docs/TA_DECOTA_THREE_SCOPE_REVIEW.md').write_text('\n'.join(lines)+'\n')
    plot(s,b,m)

def plot(s,b,m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12.3,3.5),layout='constrained');dsnames=['VidSTG','HC-STVG-v2'];colors=['#4477AA','#EE6677']
    for j,ds in enumerate(DATASETS):
        for a,arm in enumerate(['episodic','online100']):
            z=s[ds]['confirm'][arm]['corruption']['metrics']['vs_uniform_v'];x=a+j*.23-.115
            axes[0].errorbar(x,100*z['mean'],yerr=np.array([[z['mean']-z['ci95'][0]],[z['ci95'][1]-z['mean']]])*100,fmt='o',color=colors[j],capsize=3,label=dsnames[j] if a==0 else None)
        for k,field in enumerate(['delta_t','delta_v']):
            z=b[ds]['confirm']['consensus']['corruption']['metrics'][field];x=k+j*.23-.115
            axes[1].errorbar(x,100*z['mean'],yerr=np.array([[z['mean']-z['ci95'][0]],[z['ci95'][1]-z['mean']]])*100,fmt='o',color=colors[j],capsize=3)
        for k,a in enumerate(['query_cos','spatial_cos','ridge']):
            z=m[ds]['confirm']['corruption'][a]['correlation'];x=k+j*.23-.115
            if z['mean'] is not None and z['ci95'] is not None:axes[2].errorbar(x,z['mean'],yerr=np.array([[max(z['mean']-z['ci95'][0],0)],[max(z['ci95'][1]-z['mean'],0)]]),fmt='o',color=colors[j],capsize=3)
    titles=['TTS4 − Uniform4','Offset consensus − Native','Pre-update key vs transfer utility']
    for ax,title in zip(axes,titles):ax.axhline(0,color='#999999',lw=.8);ax.set_title(title);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    axes[0].set_xticks([0,1],['Episodic','Online100']);axes[0].set_ylabel('Paired vIoU change (pp)');axes[0].legend(frameon=False,fontsize=8)
    axes[1].set_xticks([0,1],['tIoU','vIoU']);axes[1].set_ylabel('Paired change (pp)')
    axes[2].set_xticks([0,1,2],['Query cosine','Spatial cosine','Ridge']);axes[2].set_ylabel('Source-node correlation')
    for ext in ['png','pdf']:fig.savefig(PUB/('three_scope_confirm.'+ext),dpi=200)
    plt.close(fig)

if __name__=='__main__':main()

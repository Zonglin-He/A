"""Report sealed CPU scores without selecting parameters or running models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,gzip,time,csv,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_paper_baseline_scoring_common_v1 import BASE,PUB,read,write,sha,verify
from scripts.audit_decota_paper_baseline_scoring_v1 import independent_statistics

def source_analysis(rows):
    q=collections.defaultdict(list)
    for r in rows:q[r['query_id']].append(r)
    queries=[]
    fields=['After_v','Before_v','Frozen_v','delta_total_v','delta_inherited_v','delta_current_v',
            'inherited_boxes_v','inherited_time_v','current_boxes_v','current_time_v','GT_time_readout_headroom_v',
            'After_t','After_s']
    for i,rr in sorted(q.items()):
        z={k:rr[0][k] for k in ['source_id','query_type','development_source_exposed','GT_duration_seconds',
             'GT_track_motion_per_second','admitted_known_GT_frames','admitted_unknown_GT_frames','admitted_expert_GT_iou','admitted_frames']}
        z.update(query_id=i,**{f:float(np.mean([r[f] for r in rr])) for f in fields});queries.append(z)
    strata={}
    for label,key in [('duration','GT_duration_seconds'),('motion','GT_track_motion_per_second'),('expert_quality','admitted_expert_GT_iou')]:
        vals=[r[key] for r in queries if r[key] is not None];edges=np.quantile(vals,[1/3,2/3]).tolist() if vals else []
        for r in queries:r['_'+label]='unknown' if r[key] is None else ('low' if r[key]<=edges[0] else 'middle' if r[key]<=edges[1] else 'high')
        strata[label]={'tercile_cutpoints':edges,'groups':stratum(queries,'_'+label,fields)}
    for label,key in [('query_type','query_type'),('development_exposure','development_source_exposed'),('admission_frames','admitted_frames')]:
        strata[label]={'groups':stratum(queries,key,fields)}
    tail=[]
    for source in sorted(set(r['source_id'] for r in queries)):
        subset=[r for r in queries if r['source_id']==source]
        tail.append(dict(source_id=source,queries=len(subset),**{f:float(np.mean([r[f] for r in subset])) for f in fields}))
    values=np.array([r['delta_total_v'] for r in tail]);current=np.array([r['delta_current_v'] for r in tail])
    cases=sorted(queries,key=lambda r:r['delta_total_v'])
    return dict(strata=strata,source_rows=tail,
        distribution=dict(source_positive=int((values>1e-12).sum()),source_negative=int((values< -1e-12).sum()),
          source_zero=int((np.abs(values)<=1e-12).sum()),current_positive=int((current>1e-12).sum()),current_negative=int((current< -1e-12).sum()),
          delta_quantiles_pp=(np.quantile(values,[0,.01,.05,.25,.5,.75,.95,.99,1])*100).tolist()),
        cases=dict(failure=cases[:8],positive=list(reversed(cases[-8:])),
            selection='post-score diagnostic extremes; anonymous scalar signal chains, not online selection or promotion'))

def stratum(queries,key,fields):
    out={}
    for group in sorted(set(str(r[key]) for r in queries)):
        rows=[r for r in queries if str(r[key])==group];sources=sorted(set(r['source_id'] for r in rows))
        out[group]=dict(queries=len(rows),parent_sources=len(sources),
            metrics={f:dict(query_macro=float(np.mean([r[f] for r in rows])),
                source_macro=float(np.mean([np.mean([r[f] for r in rows if r['source_id']==s]) for s in sources]))) for f in fields})
    return out

def cost(rows):
    cached=all(r['compute'].get('cached_readout',False) for r in rows)
    reused=all(r['compute'].get('stateless_prediction_reused_three_orders',False) or r['compute'].get('cached_readout',False) for r in rows)
    actual=[r for r in rows if r['order']=='order1'] if reused else rows
    recorded=[float(r['compute']['model_seconds']) for r in actual if 'model_seconds' in r['compute']]
    expert=[float(r['compute'].get('shared_expert_seconds',r['compute'].get('DINO_seconds',0))) for r in actual]
    return dict(logical_rows=len(rows),unique_recorded_computations=len(actual),
       stateless_reused_three_orders=reused,recorded_GPU_model_wall_seconds=sum(recorded) if recorded else None,
       recorded_model_seconds_per_actual_arrival=float(np.mean(recorded)) if recorded else None,
       recorded_model_seconds_median=float(np.median(recorded)) if recorded else None,
       recorded_model_seconds_p95=float(np.quantile(recorded,.95)) if recorded else None,
       cached_control_new_model_forwards=0 if cached else None,
       shared_frozen_DINO_recorded_seconds=sum(expert),
       optimizer_steps=sum(r['optimizer_steps'] for r in rows),backward_steps=sum(r.get('backward_steps',0) for r in rows),
       arrivals_with_optimizer_step=sum(r['optimizer_steps']>0 for r in rows),
       recoveries=sum(r['recoveries'] for r in rows),
       recorded_states_changed=sum(r.get('recorded_state_changed',False) for r in rows),
       skip_counts=dict(sum((collections.Counter(r.get('skip_counts',{})) for r in rows),collections.Counter())),
       limitation='Recorded GPU inference/update wall times are not isolated CUDA kernel timings. Source native inference cost was not retained here; cached readout zero does not imply zero deployment cost. Shared DINO work and Fisher preparation are not newly executed or added to online fit wall time.')

def pp(s):return f"{100*s:.3f}"
def run():
    verify();start=time.time();summary=read(PUB/'SUMMARY.json');audit=read(PUB/'ROOT_STATISTICS_AUDIT.json');assert audit['status']=='pass'
    diagnostic={};costs={};allrows={};tables=[]
    text=['# Sealed baseline evaluation, 2026-10-08','',
      'The human authorized this limited post-seal evaluation after pausing EATA. No inference, Fisher preparation, source-media download, OPD full-roster evaluation or parameter search was performed in this stage.',
      '', 'VidSTG test uses 10,303 official queries from 732 videos. HC2 validation uses 3,482 official clips/queries from 237 parent movies. Each query is averaged across three original orders; the primary source macro then averages queries within parent sources and weights parents equally. The official query macro weights every query equally. All values below are percentages; differences are percentage points.',
      '', 'EATA is a single completed direction supplement. Target-trained reference uses supervised same-dataset source training and is reported separately from TTA. The registered OPD main method has no complete prediction roster here; development tuning scores are not inserted. Historical target sources have been exposed, so this is not untouched confirmation.', '']
    for ds,methods in summary['datasets'].items():
        diagnostic[ds]={};costs[ds]={};allrows[ds]={}
        text += [f'## {"HC-source → VidSTG test" if ds=="vidstg" else "Vid-source → HC2 validation"}','',
          '| Method | Source vIoU | Source Δv (95% CI) | Source tIoU | Source sIoU | Source R@.3 | Source R@.5 | Query vIoU | Query R@.3 | Query R@.5 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method,s in methods.items():
            with gzip.open(PUB/ds/s['row_file'],'rt') as f:rows=[json.loads(line) for line in f]
            allrows[ds][method]=rows;diagnostic[ds][method]=source_analysis(rows);costs[ds][method]=cost(rows)
            m=s['metrics'];delta=m['delta_total_v'];ci=delta['ci95_source']
            label=method+' (one-direction supplement)' if method=='EATA-STVG' else method
            text.append('| '+label+' | '+pp(m['After_v']['source_macro'])+' | '+pp(delta['source_macro'])+f' [{pp(ci[0])}, {pp(ci[1])}] | '+' | '.join(pp(m[f]['source_macro']) for f in ['After_t','After_s','After_R0.3','After_R0.5'])+' | '+' | '.join(pp(m[f]['query_macro']) for f in ['After_v','After_R0.3','After_R0.5'])+' |')
            for agg in ['source_macro','query_macro']:
                tables.append(dict(dataset=ds,method=method,aggregation=agg,queries=s['queries'],parent_sources=s['sources'],
                  **{f:100*m[f][agg] for f in m},delta_v_ci95_low_pp=100*ci[0] if agg=='source_macro' else 100*delta['ci95_query_clustered'][0],
                  delta_v_ci95_high_pp=100*ci[1] if agg=='source_macro' else 100*delta['ci95_query_clustered'][1]))
        text+=['','### State transfer and negative tails','',
          '| Method | Inherited Δv | Current Δv | Gross source gains | Gross source losses | Source harm >5pp | Source harm >20pp | Updates / arrivals | Recoveries |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method,s in methods.items():
            if method not in ['TENT-STVG','SAR-STVG','EATA-STVG','DINO-Refine']:continue
            m=s['metrics'];z=m['delta_total_v'];c=costs[ds][method]
            text.append('| '+method+' | '+pp(m['delta_inherited_v']['source_macro'])+' | '+pp(m['delta_current_v']['source_macro'])+f" | {z['gross_gain_pp']:.3f} | {z['gross_loss_pp']:.3f} | {z['harm_gt5pp_sources']} | {z['harm_gt20pp_sources']} | {c['arrivals_with_optimizer_step']} / {c['logical_rows']} | {c['recoveries']} |")
        text+=['','The inherited effect compares saved Before with frozen Source; the current effect compares saved After with Before. Spatial and temporal terms telescope exactly to the total difference. Reference checkpoint differences are not interpreted as adaptation or inherited state.','']
    paired={}
    for ds,arms in allrows.items():
        paired[ds]={}
        for a,b in [('TENT-STVG','SAR-STVG'),('TENT-STVG','DINO-Refine'),('SAR-STVG','DINO-Refine')]+([('EATA-STVG','TENT-STVG')] if ds=='hc2' else []):
            lookup={(r['query_id'],r['order']):r for r in arms[b]}
            rows=[dict(query_id=r['query_id'],source_id=r['source_id'],order=r['order'],difference=r['After_v']-lookup[(r['query_id'],r['order'])]['After_v']) for r in arms[a]]
            z=independent_statistics(rows,['difference']);paired[ds][a+' minus '+b]=dict(source_delta_pp=100*float(z['source_macro'][0]),ci95_source_pp=(100*z['ci'][:,0]).tolist(),query_delta_pp=100*float(z['query_macro'][0]),ci95_query_pp=(100*z['query_ci'][:,0]).tolist())
    write(PUB/'DIAGNOSTICS.json',diagnostic);write(PUB/'RECORDED_COST.json',dict(methods=costs,
      new_GPU_seconds=0,scoring_CPU_seconds=summary['actual_scoring_CPU_seconds'],independent_statistics_CPU_seconds=audit['CPU_seconds'],
      report_CPU_seconds=time.time()-start,source_Fisher_preparation='previous stage only, separately excluded from deployment-arrival costs'))
    write(PUB/'PAIRED_COMPARISONS.json',paired)
    with (PUB/'TABLE.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    text += ['## Verification and interpretation','',
      'All 120,726 selected prediction payloads and 13,785 shared source inputs were checked against independent GT-free byte receipts before this stage read labels. Original barriers and science/runtime locks remain unchanged. The new evaluation barrier records the limited authorized scope, including the unfinished EATA direction.',
      '', 'Every metric readout was checked with an independent dense geometry implementation. Extracted official evaluator functions were also compared on ordinal multiples of 101 in both datasets, and synthetic contracts cover no extrapolation, clipping, half-open temporal endpoints and strict thresholds. Recorded parameter and payload chains were checked for every online arrival; recorded optimizer snapshots and qualification gradient arithmetic were verified. Full production gradients and model Jacobians were not retained and were not replayed.',
      '', 'The independent root auditor reproduced query-then-source means, source-clustered official query means, 10,000 paired source bootstrap intervals, negative tails, and scalar telescoping identities. Per-order, quality, duration, track motion, query-type, source-exposure, positive/failure chains and cost details are in the accompanying anonymous scalar files. Quality strata use only admitted frames with known GT support; unknown frames stay explicit.',
      '', 'TENT/SAR/EATA are fixed STVG ports with decoder LayerNorm scope (19,968 affine coordinates), not unchanged classification implementations. Qualification arithmetic and complete saved-state integrity do not prove scientific usefulness; interpret measured effects and actual update coverage together.',
      '', 'EATA inference remains user-paused. Evaluation completion does not imply the original complete baseline table, full OPD results or all paper stages have completed. No scores were used to select a new configuration.', '']
    (PUB/'REPORT.md').write_text('\n'.join(text))
    plots(summary,diagnostic)
    write(BASE/'REPORT_COMPLETION.json',dict(status='report_written_pending_actual_visual_and_publication',
      report_sha256=sha(PUB/'REPORT.md'),figures=['baseline_scores.png','baseline_state_effects.png','baseline_negative_tails.png'],time=time.time()))

def plots(summary,diagnostic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    dslist=['vidstg','hc2'];colors={'Source Only':'#888888','DINO-Refine':'#4c78a8','TENT-STVG':'#f58518','SAR-STVG':'#54a24b','EATA-STVG':'#e45756','Target-trained reference':'#b279a2'}
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ds,ax in zip(dslist,axes):
        arms=summary['datasets'][ds];names=list(arms);x=np.arange(len(names));v=np.array([arms[n]['metrics']['After_v']['source_macro']*100 for n in names]);ci=np.array([arms[n]['metrics']['After_v']['ci95_source'] for n in names])*100
        ax.bar(x,v,color=[colors[n] for n in names]);ax.errorbar(x,v,yerr=np.maximum(0,np.vstack([v-ci[:,0],ci[:,1]-v])),fmt='none',color='black',capsize=3)
        ax.set_xticks(x,names,rotation=28,ha='right');ax.set_ylabel('Parent-source macro vIoU (%)');ax.set_title('HC source → VidSTG test' if ds=='vidstg' else 'Vid source → HC2 validation')
        for i,y in enumerate(v):ax.text(i,y+.6,f'{y:.2f}',ha='center',fontsize=9)
    fig.suptitle('Sealed baseline scores · 95% parent-source bootstrap CI · OPD full-roster scores unavailable');fig.savefig(PUB/'baseline_scores.png');fig.savefig(PUB/'baseline_scores.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ds,ax in zip(dslist,axes):
        arms=summary['datasets'][ds];names=[n for n in arms if n in ['DINO-Refine','TENT-STVG','SAR-STVG','EATA-STVG']];x=np.arange(len(names))
        for offset,f,label,c in [(-.24,'delta_inherited_v','Inherited Before − Source','#4c78a8'),(0,'delta_current_v','Current After − Before','#f58518'),(.24,'delta_total_v','Total After − Source','#54a24b')]:
            y=np.array([arms[n]['metrics'][f]['source_macro']*100 for n in names]);ci=np.array([arms[n]['metrics'][f]['ci95_source'] for n in names])*100
            ax.bar(x+offset,y,width=.23,label=label,color=c);ax.errorbar(x+offset,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='none',color='black',capsize=2)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,names,rotation=20);ax.set_ylabel('Parent-source macro ΔvIoU (pp)');ax.set_title(ds+' · three complete orders')
    axes[0].legend(fontsize=8);fig.suptitle('State transfer and current-query effects · fixed recorded trajectories');fig.savefig(PUB/'baseline_state_effects.png');fig.savefig(PUB/'baseline_state_effects.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ds,ax in zip(dslist,axes):
        for method,z in diagnostic[ds].items():
            if method not in ['DINO-Refine','TENT-STVG','SAR-STVG','EATA-STVG']:continue
            y=np.sort([r['delta_total_v']*100 for r in z['source_rows']]);ax.plot(y,np.arange(1,len(y)+1)/len(y),label=method,color=colors[method])
        ax.axvline(0,color='black',lw=.7);ax.axvline(-5,color='gray',ls='--',lw=.7);ax.set_xlabel('Per-parent ΔvIoU (pp)');ax.set_ylabel('Cumulative source fraction');ax.set_title(ds);ax.legend(fontsize=8)
    fig.suptitle('Complete positive and negative source distribution · no result-driven retuning');fig.savefig(PUB/'baseline_negative_tails.png');fig.savefig(PUB/'baseline_negative_tails.pdf');plt.close(fig)

if __name__=='__main__':run()

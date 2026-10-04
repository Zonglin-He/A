"""Mechanism contrasts on sealed anonymous rows; no new GT or inference."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_optimizer_posterior_common_v1 import *
from scripts.score_decota_optimizer_posterior_v1 import stats

def run():
    rows=read(PUB/'SPATIAL_ROWS.json');di=read(PUB/'SPATIAL_DIAGNOSTICS.json');contrasts={}
    pairs={'admission_all_minus_admit':('all_adam','admit_adam'),'mixing_admit_minus_top1':('admit_adam','top1_adam'),
        'objective_top1_minus_direct':('top1_adam','direct_adam'),'actuation_all_sgd_minus_adam':('all_sgd','all_adam'),
        'actuation_direct_sgd_minus_adam':('direct_sgd','direct_adam'),'authority_lr_minus_adam':('all_lr_authority','all_adam'),
        'authority_post_minus_adam':('all_post_authority','all_adam')}
    lookup={(r['dataset'],r['split'],r['condition'],r['order'],r['arrival'],r['arm']):r for r in rows}
    for ds in DATASETS:
        contrasts[ds]={}
        for split in ['search','confirm']:
            out={}
            for name,(a,b) in pairs.items():
                rr=[]
                for r in rows:
                    if r['dataset']!=ds or r['split']!=split or r['condition']=='clean' or r['arm']!=a:continue
                    s=lookup[(ds,split,r['condition'],r['order'],r['arrival'],b)]
                    rr.append(dict(source_id=r['source_id'],delta_v=r['v']-s['v']))
                out[name]=stats(rr,['delta_v'])
            contrasts[ds][split]=out
    cases=[r for r in rows if r['dataset']=='vidstg' and r['split']=='confirm' and r['source_id']==35 and r['condition']=='exposure_5' and r['order']=='order1']
    dd=[d for d in di if d['dataset']=='vidstg' and d['split']=='confirm' and d['source_id']==35 and d['condition']=='exposure_5' and d['order']=='order1']
    diagnostics={}
    for ds in DATASETS:
        diagnostics[ds]={}
        for arm in sorted({r['arm'] for r in di}):
            rr=[r for r in di if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['arm']==arm]
            observed=[dict(source_id=r['source_id'],observed=r['observed_GT_IoU_delta']) for r in rr if r['observed_GT_IoU_delta'] is not None]
            unobs=[dict(source_id=r['source_id'],unobserved=r['unobserved_GT_IoU_delta']) for r in rr if r['unobserved_GT_IoU_delta'] is not None]
            diagnostics[ds][arm]=dict(proxy_improved_GT_harmed=sum(r['proxy_improved_GT_harmed'] for r in rr),
                proposal_proxy_improved_GT_harmed=sum(r['proposal_proxy_improved_GT_harmed'] for r in rr),
                zero_gradient_queries=sum(bool(r['gradient_norms']) and max(r['gradient_norms'])==0 for r in rr),
                no_backward_queries=sum(not r['gradient_norms'] for r in rr),
                observed_GT_delta=stats(observed,['observed']),unobserved_GT_delta=stats(unobs,['unobserved']),
                parameter_interpolation_functional_overshoot=sum(r['used_functional_exceeds_full_proposal'] for r in rr))
    write(PUB/'MECHANISM_CONTRASTS.json',contrasts);write(PUB/'ACTUATION_DIAGNOSTIC_SUMMARY.json',diagnostics)
    write(PUB/'SOURCE35_CASE.json',dict(rows=cases,diagnostics=dd))
    resources={ds:read(BASE/ds/'SPATIAL_PREDICTION_BARRIER.json') if (BASE/ds/'SPATIAL_PREDICTION_BARRIER.json').exists() else read(BASE/ds/'SPATIAL_BARRIER.json') for ds in DATASETS}
    write(PUB/'RESOURCE_RECEIPT.json',dict(workers={ds:{k:v for k,v in p.items() if k!='files'} for ds,p in resources.items()},
        logical_spatial_arm_arrivals=11520,reused_all_adam_arrivals=1152,new_DINO=0,new_backbone=0,temporal_backwards=0,
        worker_wall_is_not_pure_GPU_kernel=True))
    write(PUB/'CPU_AUDIT_REVISION_RECEIPT.json',dict(revisions=[read(f) for f in sorted((BASE/'cpu_revisions').glob('*.json'))],
        engineering_failure_preserved=True,predictions_changed_after_GT=False,optimizer_changed=False,SGD_addition_audited_by_per_element_float32_ULP_bound=True))
    f=ROOT/'docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md';text=f.read_text();text+='\n## Matched mechanism readback\n\n'
    text+='All−Admit changes admission; Admit−Top1 changes multi-proposal support on the same admitted observations; Top1−Direct changes the objective with matched admitted top-one boxes. SGD−Adam uses a development-matched first-step median functional displacement, not per-query equality or a matched ten-step trajectory. These comparisons can rank evidence for mechanisms; none alone proves a unique cause.\n\n'
    text+='| Confirm corruption | Contrast | ΔvIoU pp [paired95%] |\n|---|---|---:|\n'
    for ds in DATASETS:
        for name,z in contrasts[ds]['confirm'].items():
            m=z['metrics']['delta_v'];text+=f"| {ds} | {name} | {100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f},{100*m['ci95'][1]:+.4f}] |\n"
    text+='\nSource35/exposure/order1 starts from the same actual P1 prestate in every arm. The following are actual used outputs, including post-authority interpolation; optimizer proposal best-step metrics remain separately stored.\n\n| Arm | vIoU% | Δbefore pp | >20pp current harm |\n|---|---:|---:|---|\n'
    for r in cases:text+=f"| {r['arm']} | {100*r['v']:.4f} | {100*r['vs_before_v']:+.4f} | {r['vs_before_v']<-.2} |\n"
    text+='\nThe FP64 beta-zero native readout is independently retained as a numerical control; posterior qualification requires positive paired evidence against both the original and this matched control. Negative, clean and order-specific results are retained in complete summaries. Wall times and backward counts refer to worker execution and cached decoder/backprop work, not pure GPU-kernel time.\n'
    text+='\nThe first post-seal CPU audit stopped on a valid-looking SGD theoretical-delta discrepancy of 3.2781e-6 against a fixed3e-6 tolerance. The failure was preserved; resumed auditing checks each actual float32 parameter addition against an explicit ULP rounding bound, retaining exact saved state-to-state deltas. Predictions, gradients, optimization and GT-exposure chronology were unchanged.\n'
    f.write_text(text)
    # Report binding is an updated inspectable receipt, distinct from immutable prediction pins.
    write(PUB/'ROOT_REPORT_REVIEW_BINDING.json',dict(report='docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md',sha256=sha(f),time=time.time(),prediction_outputs_unchanged=True))
    write(PUB/'REPORT_BINDING_INITIAL.json',read(PUB/'REPORT_BINDING.json'))
    status(PUB/'REPORT_BINDING.json',dict(report='docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md',sha256=sha(f),root_review='ROOT_REPORT_REVIEW_BINDING.json',initial_receipt='REPORT_BINDING_INITIAL.json'))

if __name__=='__main__':run()

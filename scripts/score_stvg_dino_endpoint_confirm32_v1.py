"""All32-seal then official metrics, common admission and cost accounting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_endpoint_confirm32_v1 import D,P,verify
from vg_tta.stvg_dino_endpoint_confirm32_v1 import ARMS
from scripts.run_stvg_dino_local_residual32_v1 import mean,metrics
def aggregate(rows):
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    endpoints=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s','common_admitted_s']
    pairs=[(a,'E_MBR')for a in ARMS[1:]]+[('Samples24_endpoint','Center_full_local'),('Samples24_endpoint','Center_p_local'),('Center_p_local','Center_full_local')]
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];g=dict(queries=len(rr),parents=len(rr),active=sum(r['active']for r in rr))
        g['metrics']={a:{k:mean([r['values'][a][k]for r in rr])for k in endpoints}for a in ARMS}
        g['paired']={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else
            100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in endpoints if k!='t'}for a,b in pairs}
        g['displacement']={a:{scope:{k:mean([r['geometry'][a][scope][k]for r in rr])for k in rr[0]['geometry'][a][scope]}
                            for scope in ['all','observed','unobserved']}for a in ARMS}
        g['v_effect_contributions_pp']={a:{scope:paired([r['interpolation_effects'][a][scope]['v_contribution_pp']for r in rr])
                    for scope in ['observed_actual','unobserved_actual','benchmark_interpolated']}for a in ARMS}
        g['known_observed_query_count']=sum(r['known_observed_actual']>0 for r in rr)
        g['known_unobserved_query_count']=sum(r['known_unobserved_actual']>0 for r in rr)
        groups[ds]=g
    return groups

def main():
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
    import torch
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];start=time.perf_counter()
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];z=load(D/'predictions'/(a+'.pt'));row=plans[ds][q]
        g,span=gt[ds][0][q],gt[ds][1][q];assert list(z['frame_ids'])==list(row['frame_ids'])
        m=metrics(z,row,g,span);common=[pos for pos,yes in zip(z['planned_positions'],z['common_admitted'])if yes]
        known=[pos for pos in common if int(z['frame_ids'][pos])in g]
        for arm,b in z['outputs'].items():m['values'][arm]['common_admitted_s']=mean([iou_gt(b[pos],g[int(z['frame_ids'][pos])],row,ds)for pos in known])
        center=z['center_feedback'];samples=z['sample_feedback']['details']
        coverage=dict(planned_nodes=4,active=z['active'],center_legal=center['legal'],center_accepted_nodes=int(z['center_accepted'].sum()),
            sample_any_accepted_nodes=int((z['accepted_counts']>0).sum()),sample_accepted_instances=int(z['accepted_counts'].sum()),
            common_admitted_nodes=len(common),common_known_GT_nodes=len(known),sample_legal_states=sum(v['legal']for v in samples),
            no_support_query=not z['active'],p_values=z['p'].tolist())
        for arm in ARMS[1:]:assert m['values'][arm]['unobserved_actual_s']==m['values']['E_MBR']['unobserved_actual_s']
        rows.append(dict(alias=a,dataset=ds,active=z['active'],coverage=coverage,compute=z['compute'],**m))
    groups=aggregate(rows)
    for ds,g in groups.items():
        rr=[r for r in rows if r['dataset']==ds]
        g['coverage']={k:sum(r['coverage'][k]for r in rr)for k in ['planned_nodes','center_accepted_nodes','sample_any_accepted_nodes','sample_accepted_instances','common_admitted_nodes','common_known_GT_nodes','sample_legal_states','no_support_query']}
        g['common_known_queries']=sum(r['coverage']['common_known_GT_nodes']>0 for r in rr)
        g['cost']={k:sum(r['compute'][k]for r in rr)for k in ['case_seconds','encoding_frames','encoding_native_seconds','center_decoder_seconds','sample_decoder_seconds','center_logical_queries','sample_logical_queries','unique_sample_queries']}
    compute=dict(GPU_total_allocation_seconds=bar['GPU_total_allocation_seconds'],GPU_teacher_allocation_seconds=bar['GPU_teacher_allocation_seconds'],
        Source={ds:{k:read(D/('SOURCE_BARRIER_'+ds+'.json'))[k]for k in ['count','GPU_allocation_seconds','peak_allocated_bytes']}for ds in ['hc2','vidstg']},
        GPU_teacher_peak_allocated_bytes=bar['peak_allocated_bytes'],new_Source_captures=32,new_original_DINO_calls=0,
        shared_encoding=True,old_original_expert_and_E_generation_cost_not_first_use_latency=True)
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,compute=compute,all_outputs_before_new_GT=True,
        roster_disjoint_from_recorded_selection_pools=True,historically_full_exposed=True,globally_unseen=False,
        p_not_calibrated_correctness_probability=True,center_p_depends_on24_queries=True,same_p_not_equal_displacement_norm=True,
        no_dynamic_OPD=True,no_automatic_expansion=True))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,score_seconds=time.perf_counter()-start,GT_after_fullseal=True))
    print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in groups.items()},flush=True)

if __name__=='__main__':main()

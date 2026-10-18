"""Only after complete32 seal: official metrics, real-node propagation and costs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_query_closedloop32_v1 import D,P,PRE,verify
from vg_tta.stvg_dino_query_closedloop32_v1 import ARMS,FIT_ARMS
from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
from scripts.score_stvg_joint_robust_correction32_v1 import paired

def main():
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all_outputs_before_GT']and bar['count']==32
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint_confirm32/RESULTS_PUBLIC.json')['rows']};rows=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q])
        for arm in ['E_MBR','Center_full_local']:
            for k in ['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']:
                x,y=m['values'][arm][k],prior[a]['values'][arm][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
        assert len(set(v['t']for v in m['values'].values()))==1
        fits={arm:dict(updates=sum(v['updated']for v in f['rounds']),parameter_norm=float(f['state']['spatial.query_residual'].norm()),
            loss=[v['loss']for v in f['rounds']],accepted=[int(v['mask'].sum())for v in f['rounds']],QR_step_norm=[float(v['QR_step'].norm())for v in f['rounds']],
            output_step_L2=[float((f['path'][i+1]['boxes']-f['path'][i]['boxes']).norm())for i in range(2)],
            observed_abs_step_mean=[float((f['path'][i+1]['boxes'][o['planned_positions']]-f['path'][i]['boxes'][o['planned_positions']]).abs().mean())for i in range(2)],
            conditional_queries=f['count']['conditional_logical_queries'],gradient_calls=f['count']['backwards'])for arm,f in o['fits'].items()}
        rows.append(dict(alias=a,dataset=ds,active=o['active'],fits=fits,compute=o['compute'],**m))
    endpoints=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];pairs=[(a,b)for a in FIT_ARMS for b in ['E_MBR','Center_full_local']]+[('On_policy_KL','Fixed_center_KL'),('On_policy_KL','Fixed_sample_KL')];groups={}
    for ds in ['hc2','vidstg']:
        rs=[r for r in rows if r['dataset']==ds]
        groups[ds]=dict(count=16,active=sum(r['active']for r in rs),metrics={a:{k:mean([r['values'][a][k]for r in rs])for k in endpoints}for a in ARMS},
            paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rs])for k in endpoints if k!='t'}for a,b in pairs},
            v_contributions={a:{s:mean([r['interpolation_effects'][a][s]['v_contribution_pp']for r in rs])for s in ['observed_actual','unobserved_actual','benchmark_interpolated']}for a in FIT_ARMS},
            cost=dict(new_Source_captures=sum(r['compute']['new_Source_captures']for r in rs),DINO_encoding_frames=sum(r['compute']['DINO_encoding_frames']for r in rs),
                Source_capture_seconds=sum(r['compute']['Source_capture_seconds']for r in rs),new_conditional_queries=sum(r['compute']['new_conditional_queries']for r in rs),
                new_conditional_decoder_seconds=sum(r['compute']['new_conditional_decoder_seconds']for r in rs),fit_seconds={a:sum(r['compute']['fit_seconds'][a]for r in rs)for a in FIT_ARMS}))
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,all32_sealed_before_GT=True,query_only256=True,pretrained_weights_frozen=True,final_step=2,
        Gaussian_endpoint_forward_KL_not_original_IoU_reward_OPD=True,historical_development_exposure=True,independent_confirmation=False,
        compute={k:bar[k]for k in ['GPU_allocation_seconds','peak_allocated_bytes']},no_parameter_search=True))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_after_fullseal=True));print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in groups.items()},flush=True)
if __name__=='__main__':main()
